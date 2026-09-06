"""Bounded GPU preflight, NOT a formal training run or full validation result.

Author entry.main constructs data, model, optimizer and RNG as usual. A hook
replaces training with exactly three first-batch diagnostic updates using the
author sampler and logistic loss, then invokes the unchanged author evaluator
on a small validation subset. No test loader is iterated or scored.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("baseline", "euclidean", "sl8"), required=True)
    parser.add_argument("--score-chunk", type=int, default=1024)
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--compare-chunks", action="store_true")
    parser.add_argument("--validation-queries", type=int, default=32)
    args = parser.parse_args(argv)
    if args.score_chunk < 1:
        parser.error("score-chunk must be positive")
    if args.validation_queries < 2 or args.validation_queries > 128 or args.validation_queries % 2:
        parser.error("validation-queries must be even and between 2 and 128")
    if args.compare_chunks and args.method == "baseline":
        parser.error("baseline has no geometric chunks to compare")
    return args


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_inputs_and_source(root):
    repo = root / "upstream/vl-kge"
    spec = importlib.util.spec_from_file_location("_profile_verify_pinned_source", root / "scripts/setup_upstream.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    checked = setup.verify_checkout(repo, root / "patches/author-utils-import.patch")
    if checked["patch_state"] != "already_patched":
        raise ValueError("approved author import fix has not been applied")
    hashes = {path.relative_to(repo).as_posix(): sha(path)
              for path in sorted((repo / "vlkge").rglob("*.py"))}
    manifest = json.loads((root / "artifacts/wn9-inputs.json").read_text())
    for record in manifest["files"]:
        path = (repo / record["path"]).resolve()
        path.relative_to(repo.resolve())
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("input integrity failure: " + record["path"])
    return {"upstream_commit": checked["upstream_commit"], "checkout_verification": checked,
            "author_source_sha256": hashes, "input_manifest": manifest}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def compare_score_chunks(model, triples, torch):
    """No optimizer or RNG changes; compare stable full-candidate rankings."""
    state = (model.training, model.score_chunk, model.checkpoint_blocks)
    records, outputs, discrepancies = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            for chunk in (1024, 8192):
                model.score_chunk = chunk
                model.checkpoint_blocks = False
                torch.cuda.synchronize()
                before = time.monotonic()
                scores, delta = [], []
                for h, r, t in triples:
                    candidates = torch.arange(model.num_entities, device=h.device)
                    for ids in ((h.expand_as(candidates), r.expand_as(candidates), candidates),
                                (candidates, r.expand_as(candidates), t.expand_as(candidates))):
                        scores.append(model(*ids))
                        delta.append(model.geometric_discrepancy(*ids))
                torch.cuda.synchronize()
                outputs.append(torch.stack(scores))
                discrepancies.append(torch.stack(delta))
                records.append({"score_chunk": chunk, "checkpoint_blocks": False,
                                "full_candidate_query_count": len(scores),
                                "timing_includes_additional_raw_discrepancy_pass": True,
                                "elapsed_seconds": time.monotonic() - before})
            difference = (outputs[0] - outputs[1]).abs()
            q = [torch.tanh((model.calibration_center - delta) / model.calibration_scale)
                 for delta in discrepancies]
            # Actual initial gate is zero: combined-score equality alone would
            # hide EVERY residual bug. Also compare raw delta/q and a virtual
            # nonzero-gate combination, without modifying any model parameter.
            virtual = [score + model.residual_weight * (0.5 - model.gate) * compat
                       for score, compat in zip(outputs, q)]
            ranks_equal = torch.equal(torch.argsort(outputs[0], dim=1, descending=True, stable=True),
                                      torch.argsort(outputs[1], dim=1, descending=True, stable=True))
            return {"before_optimizer_updates": True, "scope": "small_train_only_query_set",
                    "records": records, "scores_bitwise_equal": torch.equal(outputs[0], outputs[1]),
                    "max_absolute_score_difference": float(difference.max()),
                    "all_unfiltered_candidate_orders_equal": ranks_equal,
                    "initial_gate_zero_score_parity_alone_is_vacuous": float(model.gate) == 0.0,
                    "raw_discrepancies_bitwise_equal": torch.equal(discrepancies[0], discrepancies[1]),
                    "raw_discrepancy_max_absolute_difference": float((discrepancies[0] - discrepancies[1]).abs().max()),
                    "compatibility_q_max_absolute_difference": float((q[0] - q[1]).abs().max()),
                    "virtual_gate_for_nontrivial_comparison": 0.5,
                    "virtual_nonzero_gate_score_max_absolute_difference": float((virtual[0] - virtual[1]).abs().max()),
                    "virtual_nonzero_gate_all_candidate_orders_equal": torch.equal(
                        torch.argsort(virtual[0], dim=1, descending=True, stable=True),
                        torch.argsort(virtual[1], dim=1, descending=True, stable=True))}
    finally:
        model.score_chunk, model.checkpoint_blocks = state[1:]
        model.train(state[0])


def main(argv=None):
    args = parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "method": args.method,
              "kind": "bounded_gpu_preflight_not_formal_training",
              "final_test_evaluated": False, "requested_optimizer_updates": 3,
              "requested_validation_queries": args.validation_queries}
    try:
        report.update(verify_inputs_and_source(root))
        repo = root / "upstream/vl-kge"
        core = next((p for p in (root / "vendor/sl-manifold-core/src", root.parent / "sl-manifold-core/src")
                     if (p / "sl_manifold/core.py").is_file()), None)
        sys.path[:0] = [str(root / "scripts"), str(root), str(repo)] + ([str(core)] if core else [])
        import torch
        from torch.utils.data import DataLoader, Subset
        from vlkge import helpers, models
        from vlkge.scripts import train as entry
        import run_distmult_sl_author as launcher
        if not torch.cuda.is_available():
            raise RuntimeError("GPU preflight refuses CPU execution")
        report.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda,
                      source_sha256=launcher.extension_source_hashes(root, core),
                      profile_source_sha256=sha(Path(__file__)),
                      score_chunk=args.score_chunk, checkpoint_blocks=not args.no_checkpoint_blocks)
        save(output / "profile.json", report)
        old_train, old_model, old_parse = helpers.train, models.DistMult, entry.parse_args
        resolved = {}

        def parsed():
            actual = old_parse()
            launcher.validate_locked_author_args(actual, repo)
            resolved.update(vars(actual))
            report["resolved_author_config"] = dict(resolved)
            return actual

        def train_hook(*positional, **keywords):
            bound = inspect.signature(old_train).bind(*positional, **keywords)
            bound.apply_defaults()
            p = bound.arguments
            model, optimizer, device = p["model"], p["optimizer"], p["device"]
            launcher.validate_negative_capacity(p["train_loader"], p["filter_map"], model.num_entities, p["num_neg_samples"])
            if args.method != "baseline":
                report["calibration"] = launcher.calibrate_from_training(
                    model, p["train_loader"], helpers, seed=resolved["seed"], max_positives=1024, torch=torch)
                report["initial_diagnostics"] = model.diagnostics()
                if args.compare_chunks:
                    frame = p["train_loader"].dataset.data
                    triples = torch.tensor(frame[["head_id", "relation_id", "tail_id"]].iloc[:2].to_numpy(),
                                           dtype=torch.long, device=device)
                    report["chunk_comparison"] = compare_score_chunks(model, triples, torch)
            report["batches"] = []
            model.train()
            for index, batch in enumerate(p["train_loader"]):
                if index >= 3:
                    break
                h, r, t = (batch[key].to(device) for key in ("head_id", "relation_id", "tail_id"))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                positive = model(h, r, t)
                nh, nr, nt = helpers.negative_sampling_uniform(
                    h, r, t, model.num_entities, p["num_neg_samples"], p["filter_map"],
                    p["relation_probs"], p["generator"], use_bernoulli=p["use_bernoulli"])
                negative = model(nh, nr, nt).view(len(h), p["num_neg_samples"])
                loss = helpers.compute_logistic_loss(positive, negative)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite preflight loss")
                optimizer.zero_grad()
                loss.backward()
                if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                       if parameter.grad is not None):
                    raise FloatingPointError("nonfinite preflight gradient")
                gate_gradient = float(model.raw_gate.grad) if args.method != "baseline" else None
                optimizer.step()
                torch.cuda.synchronize()
                event = {"optimizer_update": index + 1, "positive_count": len(h),
                         "negative_count": len(nh), "loss": float(loss),
                         "elapsed_seconds_including_author_negative_sampling_and_gradient_checks": time.monotonic() - before,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                         "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                         "raw_gate_gradient_before_update": gate_gradient}
                if args.method != "baseline":
                    event["diagnostics_after_update"] = model.diagnostics()
                    launcher._validate_health(event["diagnostics_after_update"])
                report["batches"].append(event)
                save(output / "profile.json", report)
                print(json.dumps({"batch_profile": event}, allow_nan=False), flush=True)
            if len(report["batches"]) != 3:
                raise RuntimeError("fewer than three training batches available")
            subset = Subset(p["val_loader"].dataset, list(range(args.validation_queries // 2)))
            loader = DataLoader(subset, batch_size=args.validation_queries // 2, shuffle=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = time.monotonic()
            mrr, hits, _ = helpers.evaluate_kge(model, loader, p["filter_map"], ks=p["top_k"],
                                               bidirectional=True, relation_to_valid_tails=None, device=device)
            torch.cuda.synchronize()
            report["small_validation_profile"] = {
                "scope": "first_validation_triples_subset_not_full_validation_not_test",
                "triple_count": len(subset), "full_candidate_query_count": 2 * len(subset),
                "candidates_per_query": model.num_entities,
                "author_evaluator_unchanged": True,
                "elapsed_seconds": time.monotonic() - before,
                "subset_mrr_not_formal_result": float(mrr), "subset_hits": hits,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            # No final test, checkpoint selection, resumed run or full epoch.

        try:
            if args.method != "baseline":
                from geometry.distmult_sl import DistMultSL
                models.DistMult = lambda **kwargs: DistMultSL(
                    **kwargs, geometry=args.method, score_chunk=args.score_chunk,
                    checkpoint_blocks=not args.no_checkpoint_blocks)
            helpers.train, entry.parse_args = train_hook, parsed
            previous_argv = sys.argv[:]
            sys.argv = ["vlkge.scripts.train", "--config", str(repo / launcher.AUTHOR_CONFIG)]
            try:
                entry.main()
            finally:
                sys.argv = previous_argv
        finally:
            helpers.train, models.DistMult, entry.parse_args = old_train, old_model, old_parse
        report["status"] = "completed"
        save(output / "profile.json", report)
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save(output / "profile.json", report)
        raise


if __name__ == "__main__":
    main()
