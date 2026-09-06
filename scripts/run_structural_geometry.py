"""Matched structural KGE on pinned WN18RR/FB15k-237; validation only.

No CLIP, feature fusion, projection, DistMult residual or entity biases.
The loss is the author's logistic mean over one positive and N negatives;
Adagrad, negative-sampling distribution, and strict ranking reuse WN9 V2.
These controlled geometry models are not original MuRP/AttH reproductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_wn9_geometry_v2 import (
    TrainNegativeSampler, check_diagnostic_backend, check_health, directional_filters,
    evaluate_strict, initial_score_statistics, select_validation,
)
from run_geometry import digest, save_json, utc

PROTOCOL = "structural_directional_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("WN18RR", "FB15k-237"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("euclidean", "hyperbolic", "sl8"), required=True)
    parser.add_argument("--kind", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--validation-only", action="store_true", default=True,
                        help="Always enabled; there is no test-scoring mode")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Full validation every N epochs and always the last epoch")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--chart-radius", type=float, default=1.5)
    parser.add_argument("--target-init-norm", type=float, default=0.5)
    parser.add_argument("--initial-logit-scale", type=float, default=1.0)
    parser.add_argument("--initial-offset", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--score-chunk", type=int, default=1024)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--candidate-chunk", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-backend", choices=("gregory12", "gauss_legendre"), default="gregory12")
    parser.add_argument("--log-order", type=int, choices=(16, 32), default=16,
                        help="GL order only; Gregory always uses exactly 12 terms")
    parser.add_argument("--validation-seed", type=int, default=260906)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--eval-limit", type=int)
    args = parser.parse_args(argv)
    for name in ("epochs", "eval_every", "patience", "batch_size", "negatives", "score_chunk",
                 "query_batch", "candidate_chunk"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("seed", "validation_seed"):
        if getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    for name in ("lr", "chart_radius", "target_init_norm", "initial_logit_scale", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.target_init_norm >= args.chart_radius:
        parser.error("target_init_norm must be strictly below chart_radius")
    if not math.isfinite(args.initial_offset):
        parser.error("initial_offset must be finite")
    for name in ("max_train_batches", "eval_limit"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or args.kind != "smoke"):
            parser.error(f"{name} is a positive smoke-only option")
    for name in ("root", "data_root", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def should_evaluate(epoch, requested_epochs, eval_every):
    return epoch % eval_every == 0 or epoch == requested_epochs


def logistic_loss(positive_scores, negative_scores):
    """Literal author compute_logistic_loss, without importing VL encoders."""
    import torch
    import torch.nn.functional as functional
    positive_labels = torch.ones_like(positive_scores.unsqueeze(1))
    negative_labels = -torch.ones_like(negative_scores)
    all_scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    all_labels = torch.cat([positive_labels, negative_labels], dim=1)
    return torch.mean(functional.softplus(-all_labels * all_scores))


def setup_imports(root):
    candidates = (root / "vendor/sl-manifold-core/src", root.parent / "sl-manifold-core/src")
    core = next((path for path in candidates if (path / "sl_manifold/core.py").is_file()), None)
    if core is None:
        raise FileNotFoundError("Missing pinned sl-manifold-core/src")
    # geometry/__init__.py imports the old class definitions. No VL model or
    # features are instantiated, but its package must be import-resolvable.
    sys.path[:0] = [str(root), str(core), str(root / "upstream/vl-kge")]
    return core


def training_loop(model, train, valid, ranking_filters, args, state, out, device, *, cpu_test=False):
    """Shared bounded loop. Production calls require CUDA; tests use tiny data."""
    import torch
    device = torch.device(device)
    if device.type != "cuda" and not cpu_test:
        raise RuntimeError("CPU training is allowed only through the explicit unit-test seam")
    started = time.monotonic()
    train_filters = directional_filters(train)
    selected, selection = select_validation(valid, args.eval_limit, args.validation_seed)
    state.update(selection)
    save_json(out / "validation_selection.json", selection)
    expected_batches = math.ceil(len(train) / args.batch_size)
    state.update(training_complete=args.max_train_batches is None or args.max_train_batches >= expected_batches,
                 total_optimizer_steps=0, validation_epochs=[], train_count=len(train))
    if args.kind == "pilot" and (state["validation_scope"] != "full" or not state["training_complete"]):
        raise ValueError("Pilot must train complete epochs and evaluate the full validation split")
    sampler = TrainNegativeSampler(model.num_entities, train_filters, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    probe_positive = train[:min(16, len(train))]
    probe_sampler = TrainNegativeSampler(model.num_entities, train_filters, args.seed + 1000003)
    probe = torch.cat((probe_positive, probe_sampler.sample(probe_positive, 1))).to(device)
    preflight_positive = train[:min(64, len(train))]
    preflight_sampler = TrainNegativeSampler(model.num_entities, train_filters, args.seed + 1000004)
    preflight_negative = preflight_sampler.sample(preflight_positive, min(args.negatives, 8))

    def diagnostics():
        with torch.no_grad():
            value = model.diagnostics(probe[:, 0], probe[:, 1], probe[:, 2])
        check_diagnostic_backend(value, args.log_backend, args.log_order, args.model)
        check_health(value)
        contract = value.get("model_contract", {})
        if contract.get("entity_bias") is not False or contract.get("distance_power") != 1:
            raise RuntimeError("Structural model must have scalar offset only and linear distance")
        return value

    def peak_memory():
        if device.type == "cuda":
            return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}

    def save_checkpoint(path, epoch, score, validation):
        checkpoint = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "score": score, "args": state["config"], "protocol": PROTOCOL,
            "dataset_provenance": state.get("dataset_provenance"),
            "log_backend": args.log_backend, "log_terms": state["log_terms"],
            "validation_selection": selection, "validation": validation,
            "total_optimizer_steps": state["total_optimizer_steps"],
            "shuffle_generator_state": generator.get_state(),
            "negative_rng_state": sampler.rng.bit_generator.state,
            "python_rng_state": random.getstate(), "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    state.update(initial_diagnostics=diagnostics(), initial_scores=initial_score_statistics(
        model, preflight_positive, preflight_negative, device, args.initial_logit_scale, args.initial_offset),
        diagnostics_scope="fixed_train_positive_and_train_negative_probe_not_all_pairs", status="running")
    save_json(out / "status.json", state)
    best, best_epoch = -math.inf, 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        order = torch.randperm(len(train), generator=generator)
        loss_sum, examples, num_batches, max_grad_norm = 0.0, 0, 0, 0.0
        first_negative_hash = None
        for left in range(0, len(order), args.batch_size):
            positive_cpu = train[order[left:left + args.batch_size]]
            negative_cpu = sampler.sample(positive_cpu, args.negatives)
            if first_negative_hash is None:
                first_negative_hash = hashlib.sha256(negative_cpu.numpy().astype("<i8").tobytes()).hexdigest()
            positive, negative = positive_cpu.to(device), negative_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores = model(positive[:, 0], positive[:, 1], positive[:, 2])
            negative_scores = model(negative[:, 0], negative[:, 1], negative[:, 2]).view(len(positive), args.negatives)
            if not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
                raise FloatingPointError("Nonfinite training scores")
            loss = logistic_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            max_grad_norm = max(max_grad_norm, float(gradient_norm))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(positive)
            examples += len(positive)
            num_batches += 1
            if args.max_train_batches is not None and num_batches >= args.max_train_batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - epoch_start
        health = diagnostics()
        validation = None
        if should_evaluate(epoch, args.epochs, args.eval_every):
            raw = evaluate_strict(model, selected, ranking_filters, device, args.query_batch, args.candidate_chunk)
            validation = {key: value for key, value in raw.items() if not key.startswith("ranks_")}
            validation["protocol"] = PROTOCOL
            state["validation_epochs"].append(epoch)
        state["total_optimizer_steps"] += num_batches
        event = {
            "epoch": epoch, "time": utc(), "train_loss": loss_sum / examples,
            "train_examples": examples, "train_batches": num_batches,
            "total_optimizer_steps": state["total_optimizer_steps"],
            "train_seconds": train_seconds, "epoch_seconds": time.monotonic() - epoch_start,
            "validation": validation, "validation_scope": state["validation_scope"],
            "subset_indices_hash": state["subset_indices_hash"], "diagnostics": health,
            "log_backend": args.log_backend, "log_terms": state["log_terms"],
            "max_gradient_norm": max_grad_norm, "first_batch_negative_sha256": first_negative_hash,
            **peak_memory(),
        }
        with (out / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if validation is not None and validation["mrr"] > best:
            best, best_epoch = validation["mrr"], epoch
            save_checkpoint(out / "best.pt", epoch, best, validation)
        save_checkpoint(out / "last.pt", epoch, validation["mrr"] if validation else None, validation)
        state.update(completed_epochs=epoch, best_epoch=best_epoch or None,
                     best_validation_mrr=best if best_epoch else None,
                     last_epoch=event, training_loop_seconds=time.monotonic() - started, **peak_memory())
        save_json(out / "status.json", state)
        metric = f"val_{state['validation_scope']}_mrr={validation['mrr']:.8f}" if validation else "validation=not_scheduled"
        print(f"epoch={epoch} {metric} train={train_seconds:.1f}s total={event['epoch_seconds']:.1f}s "
              f"steps={state['total_optimizer_steps']} peak={event['peak_allocated_gib']:.2f}GiB", flush=True)
        if validation is not None and best_epoch and epoch - best_epoch >= args.patience:
            break
    if not best_epoch:
        raise RuntimeError("No validation checkpoint was produced")
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    state["best_checkpoint_diagnostics"] = diagnostics()
    state.update(status="completed_validation_only", **peak_memory())
    return state


def main(argv=None):
    args = parse_args(argv)
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    state = {
        "status": "starting", "model": args.model, "geometry": args.model, "dataset": args.dataset,
        "kind": args.kind, "seed": args.seed, "started_at": utc(), "pid": os.getpid(),
        "protocol": PROTOCOL, "evaluation_protocol": PROTOCOL, "tie_policy": "realistic_average",
        "training_negative_filter": "train_only_directional", "validation_filter": "all_splits_directional",
        "negative_sampling": "uniform_without_replacement_per_triple",
        "loss": "author_logistic_mean_over_1_positive_and_N_negatives", "optimizer": "Adagrad",
        "score_distance_power": 1, "entity_bias": False, "coordinate_dim": 63,
        "backbone": "none_direct_trainable_entity_and_relation_coordinates", "test_evaluated": False,
        "log_backend": args.log_backend, "log_terms": 12 if args.log_backend == "gregory12" else args.log_order,
        "patience_unit": "epochs_since_best_validation", "args": config, "config": config,
        "completed_epochs": 0, "total_optimizer_steps": 0, "validation_epochs": [],
        "best_epoch": None, "best_validation_mrr": None,
        "warning": "Controlled geometry experiment, not an original MuRP/AttH paper reproduction.",
    }
    save_json(out / "config.json", config)
    save_json(out / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        core = setup_imports(args.root)
        import numpy as np
        import torch
        from structural_kge.data import load_dataset, SOURCE_COMMIT
        from structural_kge.models import StructuralKGE
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing unintended CPU training")
        # Select/initialize CUDA before querying/resetting allocator statistics.
        torch.cuda.set_device(0)
        torch.cuda.init()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        dataset = load_dataset(args.data_root, args.dataset, verify=True)
        train, valid, test = (torch.from_numpy(array.copy()) for array in (dataset.train, dataset.valid, dataset.test))
        filters = directional_filters(torch.cat((train, valid, test)))
        save_json(out / "input_manifest.json", dataset.provenance)
        sources = list((args.root / "structural_kge").rglob("*.py")) + list((args.root / "geometry").rglob("*.py"))
        sources += list((args.root / "upstream/vl-kge/vlkge").rglob("*.py"))
        sources += [Path(__file__).resolve(), Path(__file__).with_name("run_wn9_geometry_v2.py"),
                    Path(__file__).with_name("run_geometry.py")]
        sources += list((core / "sl_manifold").glob("*.py"))
        save_json(out / "source_sha256.json", {str(path): digest(path) for path in sorted(set(sources))})
        torch.cuda.reset_peak_memory_stats(device)
        model = StructuralKGE(
            num_entities=dataset.num_entities, num_relations=dataset.num_relations, geometry=args.model,
            dim=63, init_scale=args.target_init_norm, coordinate_radius=args.chart_radius,
            score_scale=args.initial_logit_scale, initial_offset=args.initial_offset,
            score_chunk=args.score_chunk, checkpoint_blocks=True,
            log_backend=args.log_backend, log_order=args.log_order,
        ).to(device)
        expected_parameters = (dataset.num_entities + dataset.num_relations) * 63 + 2
        actual_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if actual_parameters != expected_parameters:
            raise RuntimeError("Structural model violates the equal 63-D coordinate plus two-scalar parameter budget")
        state.update(dataset_provenance=dataset.provenance, dataset_commit=SOURCE_COMMIT,
                     num_entities=dataset.num_entities, num_relations=dataset.num_relations,
                     split_sizes=[len(train), len(valid), len(test)], train_count=len(train),
                     parameters_trainable=actual_parameters, parameters_total=sum(p.numel() for p in model.parameters()),
                     python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0))
        print(json.dumps({"configuration": config, "parameters_trainable": actual_parameters}), flush=True)
        training_loop(model, train, valid, filters, args, state, out, device)
        state.update(ended_at=utc(), elapsed_seconds=time.monotonic() - start)
        save_json(out / "result.json", state)
        save_json(out / "status.json", state)
        print(f"COMPLETED VALIDATION ONLY {out / 'result.json'}", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=repr(error), traceback=traceback.format_exc(), ended_at=utc(),
                     elapsed_seconds=time.monotonic() - start)
        save_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
