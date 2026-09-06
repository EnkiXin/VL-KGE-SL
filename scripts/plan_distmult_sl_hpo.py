"""Prepare reproducible, validation-only HPO manifests; NEVER launch training.

No torch dependency, server connection, scheduler or paid compute is used.
Expansion is a prepared plan, not an assertion that its trials have run.
The runner retains author training and scoring for baseline and extends only
the model for the two residual branches. See experiments/SL_HPO_PLAN.md.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "euclidean", "sl8")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_spec(path):
    spec = json.loads(Path(path).read_text())
    if tuple(spec["methods"]) != METHODS:
        raise ValueError("This matched plan requires baseline, euclidean and sl8")
    for space in (spec["shared_space"], spec["extension_space"]):
        for name, values in space.items():
            if not values or len(set(values)) != len(values):
                raise ValueError(f"Empty or duplicated search values: {name}")
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Search values must be positive finite numbers: {name}")
            if name in ("batch_size", "num_neg_samples") and any(not isinstance(v, int) for v in values):
                raise ValueError(f"Search values must be integers: {name}")
    for anchor in spec["anchors"]:
        if set(anchor) != set(spec["shared_space"]):
            raise ValueError("Anchor keys must match shared space")
        if any(value not in spec["shared_space"][key] for key, value in anchor.items()):
            raise ValueError("Anchor outside search space")
    if len({canonical(a) for a in spec["anchors"]}) != len(spec["anchors"]):
        raise ValueError("Duplicate anchors")
    if spec["policy"]["test_used_to_select_hyperparameters"]:
        raise ValueError("Test-based tuning is forbidden")
    return spec


def candidate_pool(spec, count=48):
    """Unique matched shared configs with coverage; no redundant full grid.

    Each extra branch changes at most one geometry factor from defaults.
    Shared training settings are identical for the three methods in each row.
    The Euclidean and SL branch settings are identical in each matched pair.
    """
    keys = list(spec["shared_space"])
    all_shared = [dict(zip(keys, values)) for values in
                  itertools.product(*(spec["shared_space"][key] for key in keys))]
    if count < len(spec["anchors"]) or count > len(all_shared):
        raise ValueError("count must fit all anchors and the unique shared pool")
    rng = random.Random(spec["search_seed"])
    rng.shuffle(all_shared)
    rows = [dict(a) for a in spec["anchors"]]
    seen = {canonical(row) for row in rows}
    # Fill uncovered values before purely random combinations. This makes the
    # declared broad search truthful even when using only 48 of 140 settings.
    for key in keys:
        for value in spec["shared_space"][key]:
            if not any(row[key] == value for row in rows):
                if len(rows) == count:
                    raise ValueError("count too small to cover the declared space")
                row = next(row for row in all_shared if row[key] == value
                           and canonical(row) not in seen)
                rows.append(row)
                seen.add(canonical(row))
    for row in all_shared:
        if len(rows) == count:
            break
        if canonical(row) not in seen:
            rows.append(row)
            seen.add(canonical(row))

    defaults = spec["extension_default"]
    factors = [{}]
    for key, values in spec["extension_space"].items():
        factors += [{key: value} for value in values if value != defaults[key]]
    candidates = []
    for index, shared in enumerate(rows):
        delta = {} if index < len(spec["anchors"]) else factors[(index - len(spec["anchors"])) % len(factors)]
        extension = {**defaults, **delta}
        candidates.append({"candidate_id": f"c{index:03d}", "shared": shared,
                           "extension": extension, "extension_factor_change": delta,
                           "author_anchor": index == 0})
    return candidates


def planned_jobs(spec, stage):
    definitions = {s["name"]: s for s in spec["stages"]}
    if stage not in ("profile", "anchors", "coarse"):
        raise ValueError("Only preregistered non-adaptive validation stages can be materialized")
    definition = definitions[stage]
    candidates = candidate_pool(spec, definitions["coarse"]["configs_per_method"])
    candidates = candidates[:definition["configs_per_method"]]
    jobs = []
    # Interleave matched groups, so a bounded tranche cannot spend all its
    # budget on one method before either control has been exercised.
    for candidate in candidates:
        for method in METHODS:
            overrides = {**candidate["shared"], "epochs": definition["epochs"],
                         "patience": spec["fixed"]["patience"], "seed": spec["training_seed"]}
            jobs.append({"job_id": f"{stage}-{candidate['candidate_id']}-{method}-s{spec['training_seed']}",
                         "candidate_id": candidate["candidate_id"], "method": method,
                         "author_anchor": candidate["author_anchor"],
                         "author_overrides": overrides,
                         "extension_config": {} if method == "baseline" else candidate["extension"],
                         "validation_only": True, "status": "not_started"})
    return jobs


def promote_validation_records(records, limit, anchor_id="c000"):
    """Rank ONE method's completed, test-free records with protected slots.

    Records need candidate_id, best_validation_mrr, author_overrides and
    common method/stage/seed/spec_sha256 provenance.
    An optional validation_improvement_last_10 field can rescue a slow starter.
    Early-stage scores are selection evidence, NOT final effect estimates.
    """
    if limit < 2 or limit > len(records):
        raise ValueError("Promotion count must be between 2 and record count")
    provenance_keys = ("method", "stage", "seed", "spec_sha256")
    if any(key not in records[0] for key in provenance_keys):
        raise ValueError("Missing method/stage/seed/spec provenance")
    provenance = {key: records[0][key] for key in provenance_keys}
    if provenance["method"] not in METHODS:
        raise ValueError("Unknown method in promotion")
    if provenance["stage"] not in ("anchors", "coarse", "refine", "full_validation"):
        raise ValueError("Unknown source stage in promotion")
    if not isinstance(provenance["spec_sha256"], str) or len(provenance["spec_sha256"]) != 64:
        raise ValueError("Missing valid specification hash")
    if isinstance(provenance["seed"], bool) or not isinstance(provenance["seed"], int):
        raise ValueError("Invalid seed provenance")
    ids = set()
    for record in records:
        if any(record.get(key) != value for key, value in provenance.items()):
            raise ValueError("Cannot mix method/stage/seed/spec provenance")
        if record["candidate_id"] in ids:
            raise ValueError("Duplicate candidate result")
        ids.add(record["candidate_id"])
        if record.get("status") != "completed_validation_only" or record.get("test_evaluated") is not False:
            raise ValueError("Only completed validation-only records may select candidates")
        if record.get("final_test") is not None:
            raise ValueError("Test results are not accepted by promotion")
        mrr = record.get("best_validation_mrr")
        if isinstance(mrr, bool) or not isinstance(mrr, (float, int)) or not math.isfinite(mrr) or not 0 <= mrr <= 1:
            raise ValueError("Invalid validation MRR")
        growth = record.get("validation_improvement_last_10", 0.0)
        if isinstance(growth, bool) or not isinstance(growth, (float, int)) or not math.isfinite(growth):
            raise ValueError("Invalid validation learning-curve improvement")
    if anchor_id not in ids:
        raise ValueError("Author anchor must remain available for promotion")
    ranked = sorted(records, key=lambda r: (-r["best_validation_mrr"], r["candidate_id"]))
    chosen = []

    def take(record, reason):
        if record["candidate_id"] not in {x["candidate_id"] for x in chosen}:
            chosen.append({"candidate_id": record["candidate_id"], "reason": reason,
                           **provenance,
                           "best_validation_mrr": record["best_validation_mrr"]})

    take(next(r for r in records if r["candidate_id"] == anchor_id), "protected_author_anchor")
    slow = max((r for r in records if r["candidate_id"] != anchor_id),
               key=lambda r: (r.get("validation_improvement_last_10", 0.0),
                                      -r["author_overrides"]["lr"],
                                      r["author_overrides"]["batch_size"]))
    take(slow, "protected_late_improver_or_low_lr_large_batch")
    for record in ranked:
        if len(chosen) == limit:
            break
        take(record, "validation_mrr")
    return chosen


def write_plan(spec, destination, stage):
    destination = Path(destination).absolute()
    # Never overwrite or follow an existing/broken symlink, even for a dry run.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    jobs = planned_jobs(spec, stage)
    destination.mkdir(parents=True, exist_ok=False)
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for job in jobs:
        config_dir = destination / "configs" / job["job_id"]
        config_dir.mkdir(parents=True)
        override_path = config_dir / "author_overrides.json"
        extension_path = config_dir / "extension.json"
        write_json(override_path, job["author_overrides"])
        write_json(extension_path, job["extension_config"])
        # Relative paths allow the whole manifest to be moved to the server.
        job["author_overrides_path"] = str(override_path.relative_to(destination))
        job["extension_config_path"] = str(extension_path.relative_to(destination))
    manifest = {"schema_version": 1, "experiment": spec["experiment"], "stage": stage,
                "status": "prepared_not_launched", "selection_metric": "best_validation_mrr",
                "spec_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
                "gpu_hours_authorized": None, "auto_launch": False,
                "jobs": jobs, "candidate_pool": candidate_pool(spec, next(
                    stage["configs_per_method"] for stage in spec["stages"] if stage["name"] == "coarse"))}
    write_json(destination / "spec.json", spec)
    write_json(destination / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "experiments/distmult-sl-hpo-v1.json")
    parser.add_argument("--stage", choices=("profile", "anchors", "coarse"), default="anchors")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = checked_spec(args.spec)
    manifest = write_plan(spec, args.out, args.stage)
    print(json.dumps({"manifest": str(args.out.absolute() / "manifest.json"),
                      "jobs_prepared": len(manifest["jobs"]), "training_launched": False}, indent=2))


if __name__ == "__main__":
    main()
