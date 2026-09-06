"""Run the untouched author trainer with an optional DistMult geometric residual.

The original WN9 YAML and author trainer/loss/sampler/ranker stay in place.
Only a package-level model export is replaced. Validation-only search exits at
the author's final test call, AFTER its best-checkpoint reload but BEFORE any
test scores are computed. No synthetic metrics or substitute test data are used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


AUTHOR_CONFIG = "vlkge/configs/wn9_img/distmult_clip.yaml"
AUTHOR_OVERRIDE_KEYS = {"lr", "batch_size", "num_neg_samples", "epochs", "patience", "seed"}
MAX_GREGORY_REMAINDER_BOUND = 1e-3
EXTENSION_DEFAULTS = {
    "residual_weight": 1.0,
    "coordinate_scale": 0.1,
    "chart_radius": 0.5,
    "score_chunk": 1024,
    "checkpoint_blocks": True,
}


class _SkipFinalTest(Exception):
    """Private control-flow signal; never represents a computed test result."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json_object(value, *, file_only=False):
    if value is None:
        return {}, None
    if not file_only and value.lstrip().startswith("{"):
        result, source = json.loads(value), None
    else:
        source = Path(value).resolve()
        result = json.loads(source.read_text())
    if not isinstance(result, dict):
        raise ValueError("configuration JSON must be an object")
    return result, source


def validate_author_overrides(overrides):
    unknown = set(overrides) - AUTHOR_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Forbidden author overrides {sorted(unknown)}; allowed actual author names: {sorted(AUTHOR_OVERRIDE_KEYS)}")
    for name, value in overrides.items():
        if name == "lr":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("lr must be finite and positive")
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        elif name == "seed":
            if not 0 <= value < 2**32:
                raise ValueError("seed must be between 0 and 2**32-1")
        elif value <= 0:
            raise ValueError(f"{name} must be positive")
    return dict(overrides)


def validate_extension_config(method, config):
    unknown = set(config) - set(EXTENSION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown or forbidden extension settings: {sorted(unknown)}")
    if method == "baseline":
        if config:
            raise ValueError("baseline uses the original DistMult; extension config must be empty")
        return {}
    resolved = {**EXTENSION_DEFAULTS, **config}
    for name in ("residual_weight", "coordinate_scale", "chart_radius"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name != "residual_weight" and value == 0):
            raise ValueError(f"invalid {name}")
    if isinstance(resolved["score_chunk"], bool) or not isinstance(resolved["score_chunk"], int) or resolved["score_chunk"] <= 0:
        raise ValueError("score_chunk must be a positive integer")
    if not isinstance(resolved["checkpoint_blocks"], bool):
        raise ValueError("checkpoint_blocks must be boolean")
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", default=AUTHOR_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--method", choices=("baseline", "sl8", "euclidean"), required=True)
    parser.add_argument("--extension-config", help="JSON file containing only residual-head settings")
    parser.add_argument("--author-overrides", help="JSON object or JSON file using whitelisted author argument names")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--calibration-positives", type=int, default=1024,
                        help="Training-only calibration positives, each paired with one negative; at most 1024")
    args = parser.parse_args(argv)
    try:
        args.author_overrides, args.author_overrides_file = read_json_object(args.author_overrides)
        args.author_overrides = validate_author_overrides(args.author_overrides)
        raw_config, args.extension_config_file = read_json_object(args.extension_config, file_only=True)
        args.extension_config = validate_extension_config(args.method, raw_config)
        if args.smoke_epochs is not None and args.smoke_epochs <= 0:
            raise ValueError("smoke-epochs must be positive")
        if args.smoke_epochs is not None and "epochs" in args.author_overrides:
            raise ValueError("do not combine --smoke-epochs with an epochs author override")
        if not 1 <= args.calibration_positives <= 1024:
            raise ValueError("calibration-positives must be between 1 and 1024")
        args.repo = args.repo.resolve()
        args.run_dir = args.run_dir.resolve()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def validate_locked_author_args(resolved, repo):
    """A different YAML cannot silently alter this controlled WN9 protocol."""
    expected = {
        "dataset": "wn9_img", "model": "DistMult", "embedding_dim": 768,
        "fusion_mode": "average", "use_structural": True, "use_visual": True,
        "use_textual": True, "use_relation_features": False,
        "freeze_visual": True, "freeze_textual": True, "visual_proj": False,
        "textual_proj": False, "normalize_before_fusion": False,
        "normalize_visual": False, "normalize_textual": False,
        "use_per_relation_candidates": False, "bidirectional_eval": True,
        "inductive": False, "modality_asymmetry": False, "use_bernoulli": False,
        "use_scheduler": False, "evaluate_every": 1, "top_k": [1, 3, 10],
        "resume_from": None, "exclude_relations": None, "exclude_relations_eval": None,
        "add_inverse_relations": None, "downsample_relation": None,
    }
    for key, value in expected.items():
        if not hasattr(resolved, key) or getattr(resolved, key) != value:
            raise ValueError(f"Locked author setting {key} must remain {value!r}")
    if getattr(resolved, "shared_projection", None) not in (None, []):
        raise ValueError("shared_projection must remain disabled")
    expected_paths = {
        "data_path": "vlkge/data/wn9_img/wn9_img_triples.csv",
        "visual_features_path": "vlkge/data/wn9_img/features/wn9_img_vf_clip.pkl",
        "textual_features_path": "vlkge/data/wn9_img/features/wn9_img_tf_clip.pkl",
    }
    for key, relative in expected_paths.items():
        if Path(getattr(resolved, key)).resolve() != (repo / relative).resolve():
            raise ValueError(f"Locked WN9 input path changed: {key}")


def _validate_health(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_health(child, path + "/" + str(key))
    elif isinstance(value, bool):
        if path.endswith("parameters_finite") and not value:
            raise FloatingPointError(f"Unsafe diagnostic {path}")
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite diagnostic {path}")
        if ("branch_risk_fraction" in path or "nonfinite_fraction" in path) and value > 0:
            raise FloatingPointError(f"Unsafe diagnostic {path}: {value}")
        if path.endswith("max_gregory_frobenius_remainder_bound") and value > MAX_GREGORY_REMAINDER_BOUND:
            raise FloatingPointError(f"Gregory remainder bound exceeds {MAX_GREGORY_REMAINDER_BOUND}: {path}={value}")


def _json_safe(value):
    """Keep invalid diagnostics readable without breaking failure recording."""
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def calibrate_from_training(model, train_loader, helpers, *, seed, max_positives, torch):
    """Use no validation/test object, no training-loader iteration, and no shared RNG."""
    frame = getattr(train_loader.dataset, "data", None)
    columns = ["head_id", "relation_id", "tail_id"]
    if frame is None or any(column not in frame for column in columns):
        raise ValueError("Calibration requires the original KGDataset with mapped training IDs")
    train = torch.tensor(frame[columns].to_numpy(), dtype=torch.long)
    if not len(train):
        raise ValueError("Cannot calibrate on empty training data")
    # Deliberately train-only. Preserve the author's combined-key convention
    # for this diagnostic sampling; no all-split filter is used here.
    train_filter = defaultdict(set)
    for head, relation, tail in train.tolist():
        train_filter[(head, relation)].add(tail)
        train_filter[(tail, relation)].add(head)
    calibration_seed = (int(seed) + 7919) % 2**32
    generator = torch.Generator().manual_seed(calibration_seed)
    indices = torch.randperm(len(train), generator=generator)[:min(max_positives, len(train))]
    positives = train[indices]
    for head, relation, tail in positives.tolist():
        if len(train_filter[(head, relation)]) >= model.num_entities or len(train_filter[(tail, relation)]) >= model.num_entities:
            raise ValueError("Calibration has no available corrupted entity for a selected positive")
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            nh, nr, nt = helpers.negative_sampling_uniform(
                positives[:, 0], positives[:, 1], positives[:, 2], model.num_entities,
                1, train_filter, {}, generator, use_bernoulli=False,
            )
            negatives = torch.stack((nh, nr, nt), dim=1).cpu()
            calibration_triples = torch.cat((positives, negatives), dim=0)
            model.eval()
            details = model.calibrate(calibration_triples)
    finally:
        model.train(was_training)
    if not isinstance(details, dict) or details.get("calibrated") is not True or details.get("frozen") is not True:
        raise ValueError("Model calibration must return a frozen, successful calibration record")
    _validate_health(details)
    return {
        **details, "calibration_seed": calibration_seed,
        "training_positive_count": len(positives), "training_corrupted_count": len(negatives),
        "training_split_size": len(train), "positive_row_indices": indices.tolist(),
        "calibration_triples_sha256": hashlib.sha256(calibration_triples.numpy().tobytes()).hexdigest(),
        "sampling": "independent_cpu_generator; train-only combined-direction rejection filter; one unique negative per positive",
        "validation_or_test_used": False, "global_rng_preserved": True,
    }


def validate_negative_capacity(train_loader, filter_map, num_entities, num_neg_samples):
    """Fail instead of entering the author's unbounded rejection loop."""
    frame = getattr(train_loader.dataset, "data", None)
    columns = ["head_id", "relation_id", "tail_id"]
    if frame is None or any(column not in frame for column in columns):
        raise ValueError("Negative-capacity check requires the author's mapped training dataset")
    minimum = num_entities
    for head, relation, tail in frame[columns].to_numpy().tolist():
        for fixed in (int(head), int(tail)):
            available = num_entities - len(filter_map.get((fixed, int(relation)), set()))
            minimum = min(minimum, available)
            if available < num_neg_samples:
                raise ValueError(f"Author negative sampler would hang: requested {num_neg_samples} distinct negatives, "
                                 f"but only {available} available for training filter key {(fixed, int(relation))}")
    return {"requested_unique_negatives": num_neg_samples, "minimum_available_for_training_query": minimum,
            "filter": "unchanged_author_all_split_combined_direction_filter"}


def extension_source_hashes(root, core=None):
    files = {
        "project/scripts/run_distmult_sl_author.py": Path(__file__).resolve(),
        "project/scripts/run_author.py": root / "scripts/run_author.py",
    }
    for relative in ("geometry/distmult_sl.py", "geometry/__init__.py"):
        path = root / relative
        if path.is_file():
            files["project/" + relative] = path
    if core is not None:
        for path in sorted((core / "sl_manifold").glob("*.py")):
            files["shared_sl/sl_manifold/" + path.name] = path
    return {label: digest(path) for label, path in files.items()}


def run_with_hooks(args, runner, models, helpers, entry, torch, *, extension_class=None, source_hashes=None):
    """Inject only the model factory, observation hooks, and final-test sentinel.

    The explicit module arguments allow routing tests without CUDA or training.
    All patched globals, the working directory, arguments, and signal handlers
    are restored even if the original runner fails before reaching training.
    """
    old = {"model": models.DistMult, "train": helpers.train, "evaluate": helpers.evaluate_kge,
           "parse": entry.parse_args, "save": runner.save_json, "active": runner.ACTIVE_RUN_DIR,
           "argv": sys.argv[:], "cwd": Path.cwd()}
    context = {"test_loader": None, "validation_loader": None, "test_evaluated": False,
               "skipping_final_test": False, "author_args_original": None,
               "calibration": None, "last_diagnostics": None, "model": None,
               "negative_sampling_capacity": None}
    started = time.monotonic()
    metadata = {
        "method": args.method,
        "track": "author_release_protocol_distmult_baseline" if args.method == "baseline" else "author_release_protocol_distmult_residual",
        "runtime_model_class": "vlkge.models.distmult.DistMult" if args.method == "baseline" else "geometry.distmult_sl.DistMultSL",
        "runtime_export_replaced": args.method != "baseline",
        "unmodified_author_model": args.method == "baseline",
        "uses_original_author_trainer": True,
        "extension_config": dict(args.extension_config),
        "author_overrides": dict(args.author_overrides),
        "extension_source_sha256": source_hashes or {},
        "validation_only": args.validation_only,
        "selection_metric": "best_validation_mrr",
        "diagnostic_limits": {"max_gregory_frobenius_remainder_bound": MAX_GREGORY_REMAINDER_BOUND},
        "warning": "Original author mixed-direction/all-split filtering retained. Runtime residual extensions are not pure author-model reproductions. All HPO changes are explicitly recorded.",
    }
    for key in ("extension_config_file", "author_overrides_file"):
        path = getattr(args, key, None)
        if path is not None:
            metadata[key] = {"path": str(path), "sha256": digest(path)}

    def decorate(value):
        record = dict(value)
        if "kind" in record:
            record["author_observer_kind"] = record["kind"]
        record.update(metadata)
        record.update(test_evaluated=context["test_evaluated"],
                      author_args_before_overrides=context["author_args_original"],
                      calibration=context["calibration"], last_diagnostics=context["last_diagnostics"],
                      negative_sampling_capacity=context["negative_sampling_capacity"])
        if args.validation_only:
            record.pop("final_test", None)
            record["kind"] = "validation_only"
        return record

    def save_hook(path, value):
        name = path.name
        if name == "status.json" and context["skipping_final_test"] and str(value.get("error", "")).startswith("_SkipFinalTest("):
            # The untouched observer catches every exception. This private
            # sentinel is successful validation completion, not a failed run.
            return
        if name in ("status.json", "result.json"):
            value = decorate(value)
        elif name == "resolved_config.json":
            value = {**value, "runtime_extension": metadata,
                     "author_args_before_overrides": context["author_args_original"]}
        elif name == "source_sha256.json":
            value = {**value, **{"extension/" + key: sha for key, sha in metadata["extension_source_sha256"].items()}}
        old["save"](path, value)

    @functools.wraps(old["parse"])
    def parse_hook():
        resolved = old["parse"]()
        validate_locked_author_args(resolved, args.repo)
        context["author_args_original"] = dict(vars(resolved))
        for name, value in args.author_overrides.items():
            if not hasattr(resolved, name):
                raise ValueError(f"Author parser has no setting {name}")
            setattr(resolved, name, value)
        context["effective_seed"] = resolved.seed
        return resolved

    @functools.wraps(old["train"])
    def train_hook(*train_args, **train_kwargs):
        bound = inspect.signature(old["train"]).bind(*train_args, **train_kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values["test_loader"] is values["val_loader"] or values["test_loader"] is values["train_loader"]:
            raise ValueError("Test loader must remain a distinct original author object")
        context.update(test_loader=values["test_loader"], validation_loader=values["val_loader"], model=values["model"])
        if "filter_map" in values and "num_neg_samples" in values:
            context["negative_sampling_capacity"] = validate_negative_capacity(
                values["train_loader"], values["filter_map"], values["model"].num_entities, values["num_neg_samples"])
        if args.method != "baseline":
            context["calibration"] = calibrate_from_training(
                values["model"], values["train_loader"], helpers,
                seed=context["effective_seed"], max_positives=args.calibration_positives, torch=torch,
            )
            old["save"](args.run_dir / "calibration.json", context["calibration"])
        return old["train"](*train_args, **train_kwargs)

    @functools.wraps(old["evaluate"])
    def evaluate_hook(*evaluate_args, **evaluate_kwargs):
        bound = inspect.signature(old["evaluate"]).bind(*evaluate_args, **evaluate_kwargs)
        values = bound.arguments
        loader = values.get("dataloader")
        if loader is None:
            loader = evaluate_args[1] if len(evaluate_args) > 1 else next(v for k, v in values.items() if "loader" in k)
        if loader is context["test_loader"] and context["test_loader"] is not None:
            if args.validation_only:
                if not (args.run_dir / "best.pt").is_file():
                    raise RuntimeError("Final-test interception reached without the author's best checkpoint")
                context["skipping_final_test"] = True
                print("[SL launcher] Validation-only: intercepted final test before scoring; author already reloaded its best checkpoint. No test metrics computed.", flush=True)
                raise _SkipFinalTest("validation-only: final test scoring intentionally omitted")
            result = old["evaluate"](*evaluate_args, **evaluate_kwargs)
            context["test_evaluated"] = True
        else:
            result = old["evaluate"](*evaluate_args, **evaluate_kwargs)
        if args.method != "baseline" and (loader is context["validation_loader"] or loader is context["test_loader"]):
            diagnostics = context["model"].diagnostics()
            context["last_diagnostics"] = _json_safe(diagnostics)
            split = "validation" if loader is context["validation_loader"] else "test"
            context["diagnostic_events"] = context.get("diagnostic_events", 0) + 1
            with (args.run_dir / "diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": utc(), "split": split,
                                         "evaluation_index": context["diagnostic_events"],
                                         "diagnostics": context["last_diagnostics"]}, allow_nan=False) + "\n")
            _validate_health(diagnostics)
        return result

    def model_factory(**kwargs):
        if extension_class is None:
            raise RuntimeError("Missing DistMultSL class for residual method")
        return extension_class(**kwargs, geometry=args.method, **args.extension_config)

    def write_failure(error):
        if runner.ACTIVE_RUN_DIR is None or Path(runner.ACTIVE_RUN_DIR).resolve() != args.run_dir:
            return
        path = args.run_dir / "status.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      ended_at=utc(), error=repr(error), traceback=traceback.format_exc())
        save_hook(path, record)

    def finalize_validation_only():
        checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
        epoch, score = checkpoint.get("epoch"), checkpoint.get("score")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise RuntimeError("Author checkpoint is missing a valid best epoch")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Author checkpoint is missing a valid best validation MRR")
        events = [json.loads(line) for line in (args.run_dir / "evaluations.jsonl").read_text().splitlines() if line.strip()]
        validation = [event for event in events if event["split"] == "validation"]
        if not validation or any(event["split"] == "test" for event in events) or context["test_evaluated"]:
            raise RuntimeError("Validation-only completion has missing validation or unexpected test metrics")
        match = [event for event in validation if event["evaluation_index"] == epoch]
        if len(match) != 1 or not math.isclose(float(match[0]["mrr"]), score, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Checkpoint validation score does not match the recorded best epoch")
        record = json.loads((args.run_dir / "status.json").read_text())
        for key in ("error", "traceback", "final_test"):
            record.pop(key, None)
        record.update(status="completed_validation_only", ended_at=utc(),
                      best_epoch=epoch, best_validation_mrr=score, completed_epochs=len(validation),
                      elapsed_seconds=time.monotonic() - started,
                      cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      final_test_interception={"stage": "after_author_best_checkpoint_reload_before_test_scoring",
                                               "synthetic_metrics_used": False,
                                               "author_trainer_returned_normally": False})
        if args.method != "baseline":
            diagnostics = context["model"].diagnostics()
            context["last_diagnostics"] = _json_safe(diagnostics)
            _validate_health(diagnostics)
        save_hook(args.run_dir / "result.json", record)
        save_hook(args.run_dir / "status.json", record)
        print(f"[SL launcher] COMPLETED VALIDATION ONLY: best MRR={score:.8f} at epoch {epoch}; test not evaluated", flush=True)

    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        for sig in original_handlers:
            signal.signal(sig, interrupted)
        runner.ACTIVE_RUN_DIR = None
        runner.save_json = save_hook
        entry.parse_args, helpers.train, helpers.evaluate_kge = parse_hook, train_hook, evaluate_hook
        if args.method != "baseline":
            models.DistMult = model_factory
        sys.argv = [str(Path(runner.__file__).resolve()), "--repo", str(args.repo),
                    "--config", args.config, "--run-dir", str(args.run_dir)]
        if args.smoke_epochs is not None:
            sys.argv.extend(("--smoke-epochs", str(args.smoke_epochs)))
        try:
            runner.main()
            if args.validation_only:
                raise RuntimeError("Validation-only run returned without the required final-test interception")
            if not context["test_evaluated"]:
                raise RuntimeError("Formal author run returned without an observed test evaluation")
        except _SkipFinalTest:
            if not args.validation_only or not context["skipping_final_test"]:
                raise
            finalize_validation_only()
    except BaseException as error:
        write_failure(error)
        raise
    finally:
        models.DistMult = old["model"]
        helpers.train, helpers.evaluate_kge, entry.parse_args = old["train"], old["evaluate"], old["parse"]
        runner.save_json, runner.ACTIVE_RUN_DIR = old["save"], old["active"]
        sys.argv = old["argv"]
        os.chdir(old["cwd"])
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    args = parse_args(argv)
    repo = args.repo
    root = repo.parent.parent
    requested_config = (repo / args.config).resolve()
    if requested_config != (repo / AUTHOR_CONFIG).resolve():
        raise ValueError("This launcher requires the original WN9 DistMult YAML; use explicit --author-overrides for HPO")
    expected = subprocess.check_output(["git", "show", f"HEAD:{AUTHOR_CONFIG}"], cwd=repo)
    if requested_config.read_bytes() != expected:
        raise ValueError("Author YAML has local changes; refusing to label it as the original configuration")
    before_path = sys.path[:]
    try:
        for path in (repo, root):
            sys.path.insert(0, str(path))
        core = next((path for path in (root / "vendor/sl-manifold-core/src", root.parent / "sl-manifold-core/src")
                     if (path / "sl_manifold/core.py").is_file()), None)
        if core is not None:
            sys.path.insert(0, str(core))
        import torch
        from vlkge import helpers, models
        from vlkge.scripts import train as entry
        extension_class = None
        if args.method != "baseline":
            if core is None:
                raise FileNotFoundError("Missing shared SL core")
            from geometry.distmult_sl import DistMultSL
            extension_class = DistMultSL
        spec = importlib.util.spec_from_file_location("_sl_launcher_author_observer", root / "scripts/run_author.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        run_with_hooks(args, runner, models, helpers, entry, torch,
                       extension_class=extension_class, source_hashes=extension_source_hashes(root, core))
    finally:
        sys.path[:] = before_path


if __name__ == "__main__":
    main()
