"""Observe the author trainer without replacing its math, sampling, or ranking.

Writes full-precision evaluation events, resolved arguments, provenance and CUDA
peaks. Author checkpoint selection and final test remain in helpers.train.
"""
import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback

COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"
ACTIVE_RUN_DIR = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    global ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-epochs", type=int)
    args = parser.parse_args()
    if args.smoke_epochs is not None and args.smoke_epochs <= 0:
        parser.error("--smoke-epochs must be a positive integer")
    repo, out = args.repo.resolve(), args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN_DIR = out
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {commit}")
    config = (repo / args.config).resolve()
    config.relative_to(repo)
    (out / "author_config.yaml").write_bytes(config.read_bytes())
    diff = subprocess.check_output(["git", "diff", "--", "*.py"], cwd=repo, text=True)
    (out / "upstream_runtime.patch").write_text(diff)
    source_hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((repo / "vlkge").rglob("*.py"))}
    save_json(out / "source_sha256.json", source_hashes)
    state = {"status": "starting", "track": "author_release_protocol",
             "kind": "smoke" if args.smoke_epochs is not None else "formal",
             "started_at": utc(), "host": socket.gethostname(), "pid": os.getpid(),
             "upstream_commit": commit, "config": str(config), "run_dir": str(out),
             "python": platform.python_version(), "evaluations": 0,
             "warning": "Preserves upstream combined-direction/all-split filtering. Not the later strict geometry-comparison track."}
    save_json(out / "status.json", state)
    input_manifest = json.loads((repo.parent.parent / "artifacts/wn9-inputs.json").read_text())
    for record in input_manifest["files"]:
        input_path = repo / record["path"]
        h = hashlib.sha256()
        with input_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        if input_path.stat().st_size != record["bytes"] or h.hexdigest() != record["sha256"]:
            raise RuntimeError(f"Input integrity check failed: {record['path']}")
    save_json(out / "input_manifest.json", input_manifest)
    started = time.monotonic()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from vlkge import helpers
    from vlkge.scripts import train as entry

    if not torch.cuda.is_available():
        state.update(status="failed", error="CUDA unavailable; refusing an accidental CPU run", ended_at=utc())
        save_json(out / "status.json", state)
        raise RuntimeError(state["error"])
    state["environment"] = {
        p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "PyYAML", "tqdm"]}
    state["gpu"] = torch.cuda.get_device_name(0)
    state["cuda_runtime"] = torch.version.cuda
    state["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    state["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    save_json(out / "status.json", state)
    original_parse = entry.parse_args
    original_train = helpers.train
    original_evaluate = helpers.evaluate_kge
    loaders = {}
    counts = {}
    test_result = None
    train_entered_at = None

    def parse_observed():
        resolved = original_parse()
        save_json(out / "resolved_config.json", vars(resolved))
        state["resolved_config"] = vars(resolved)
        return resolved

    @functools.wraps(original_evaluate)
    def evaluate_observed(*ev_args, **ev_kwargs):
        nonlocal test_result
        bound = inspect.signature(original_evaluate).bind(*ev_args, **ev_kwargs)
        # evaluate_kge uses test_loader as its generic loader argument.
        loader = ev_args[1] if len(ev_args) > 1 else next(
            value for name, value in bound.arguments.items() if "loader" in name)
        split = loaders.get(id(loader), "unknown")
        counts[split] = counts.get(split, 0) + 1
        torch.cuda.synchronize()
        before = time.monotonic()
        print(f"[observer] {split} evaluation {counts[split]} started", flush=True)
        result = original_evaluate(*ev_args, **ev_kwargs)
        torch.cuda.synchronize()
        mrr, hits, relations = result
        event = {"time": utc(), "split": split, "evaluation_index": counts[split],
                 "elapsed_seconds": time.monotonic() - before,
                 "run_elapsed_seconds": time.monotonic() - started,
                 "mrr": float(mrr), "hits": {str(k): float(v) for k, v in hits.items()},
                 "per_relation": {str(k): {str(a): float(b) for a, b in v.items()} for k, v in relations.items()},
                 "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        with (out / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        state.update(status="running", last_evaluation=event, evaluations=state["evaluations"] + 1)
        save_json(out / "status.json", state)
        print(f"[observer] {split}: MRR={mrr:.8f}, {event['elapsed_seconds']:.2f}s, "
              f"peak={event['cuda_peak_allocated_gib']:.3f} GiB", flush=True)
        if split == "test":
            test_result = event
        return result

    @functools.wraps(original_train)
    def train_observed(*tr_args, **tr_kwargs):
        nonlocal train_entered_at
        bound = inspect.signature(original_train).bind(*tr_args, **tr_kwargs)
        bound.apply_defaults()
        parameters = bound.arguments
        for key, split in [("sampled_train_loader", "sampled_train"), ("val_loader", "validation"), ("test_loader", "test")]:
            loaders[id(parameters[key])] = split
        model = parameters["model"]
        state["model"] = type(model).__name__
        state["num_entities"] = model.num_entities
        state["num_relations"] = model.num_relations
        state["parameters_total"] = sum(p.numel() for p in model.parameters())
        state["parameters_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state["split_sizes"] = {key: len(parameters[key].dataset) for key in [
            "train_loader", "sampled_train_loader", "val_loader", "test_loader"]}
        state["status"] = "running"
        state["training_started_at"] = utc()
        save_json(out / "status.json", state)
        train_entered_at = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        return original_train(*tr_args, **tr_kwargs)

    entry.parse_args = parse_observed
    helpers.train = train_observed
    helpers.evaluate_kge = evaluate_observed
    sys.argv = ["vlkge.scripts.train", "--config", str(config), "--save_path", str(out / "best.pt")]
    if args.smoke_epochs is not None:
        sys.argv.extend(["--epochs", str(args.smoke_epochs)])
    state["author_argv"] = sys.argv
    save_json(out / "status.json", state)
    try:
        entry.main()
        if test_result is None or not (out / "best.pt").exists():
            raise RuntimeError("Trainer returned without a best checkpoint and final test result")
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        state.update(status="completed", ended_at=utc(), final_test=test_result,
                     best_epoch=checkpoint.get("epoch"), best_validation_mrr=checkpoint.get("score"),
                     elapsed_seconds=time.monotonic() - started,
                     training_and_evaluation_seconds=time.monotonic() - train_entered_at,
                     cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        save_json(out / "result.json", state)
        save_json(out / "status.json", state)
        print(f"[observer] COMPLETED: {out / 'result.json'}", flush=True)
    except BaseException as error:
        state.update(status="failed", ended_at=utc(), error=repr(error),
                     traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        save_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_RUN_DIR is not None:
            status_path = ACTIVE_RUN_DIR / "status.json"
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         ended_at=utc(), error=repr(error), traceback=traceback.format_exc())
            save_json(status_path, state)
        raise
