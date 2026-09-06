"""WN9 V2: matched linear-distance geometries, strict filters, validation only.

Retains the pinned author split/IDs, frozen 768-D fusion, logistic loss and
Adagrad. Unlike the older release-protocol experiment, negatives only consult
TRAIN facts, filters are directional, and exact ties receive average ranks.
Consequently the old author-release MRR is not a directly comparable baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_geometry import COMMIT, digest, load_inputs, save_json, setup_paths, utc


PROTOCOL = "strict_directional_v2"
TIE_POLICY = "realistic_average"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("euclidean", "hyperbolic", "sl8"), required=True)
    parser.add_argument("--kind", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--validation-only", action="store_true", default=True,
                        help="Always enabled: this runner never scores the test split")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--coordinate-scale", type=float, default=1.0)
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
    parser.add_argument("--log-backend", choices=("gregory12", "gauss_legendre"), default="gregory12",
                        help="SL log implementation; saved results cannot mix backends")
    parser.add_argument("--log-order", type=int, choices=(16, 32), default=16,
                        help="Gauss-Legendre order only; Gregory always uses exactly 12 terms")
    parser.add_argument("--validation-seed", type=int, default=260906)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--eval-limit", type=int)
    args = parser.parse_args(argv)
    for name in ("epochs", "patience", "batch_size", "negatives", "score_chunk",
                 "query_batch", "candidate_chunk", "log_order"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("lr", "coordinate_scale", "chart_radius", "target_init_norm",
                 "initial_logit_scale", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.target_init_norm >= args.chart_radius:
        parser.error("target_init_norm must be strictly below chart_radius")
    if not math.isfinite(args.initial_offset):
        parser.error("initial_offset must be finite")
    for name in ("max_train_batches", "eval_limit"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or args.kind != "smoke"):
            parser.error(f"{name} is a positive smoke-only option")
    args.root = args.root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def directional_filters(triples):
    """Separate tail(h,r) and head(t,r) indexes; never merge their keys."""
    filters = {"tail": defaultdict(set), "head": defaultdict(set)}
    for h, r, t in triples.tolist():
        filters["tail"][(h, r)].add(t)
        filters["head"][(t, r)].add(h)
    return {direction: dict(index) for direction, index in filters.items()}


class TrainNegativeSampler:
    """Uniform distinct negatives, train-only, with bounded complement sampling.

    One independent 50/50 corruption direction is drawn per positive triple,
    matching the author's sampling distribution (not the old RNG trajectory).
    A rank-to-complement mapping avoids potentially infinite rejection loops.
    """

    def __init__(self, num_entities, train_filters, seed):
        import numpy as np
        self.num_entities = int(num_entities)
        self.rng = np.random.default_rng(seed)
        self.index = {}
        for direction, index in train_filters.items():
            self.index[direction] = {}
            for key, values in index.items():
                ordered = np.asarray(sorted(values), dtype=np.int64)
                if len(ordered) and (ordered[0] < 0 or ordered[-1] >= num_entities):
                    raise ValueError("Training filter has an invalid entity ID")
                self.index[direction][key] = (ordered - np.arange(len(ordered)), len(ordered))

    def sample(self, positives, negatives):
        import numpy as np
        import torch
        if positives.device.type != "cpu" or positives.ndim != 2 or positives.shape[1] != 3:
            raise ValueError("Positive triples must be a CPU [B,3] tensor")
        if negatives <= 0:
            raise ValueError("Negative count must be positive")
        triples = positives.numpy()
        output = np.repeat(triples[:, None, :], negatives, axis=1)
        corrupt_heads = self.rng.random(len(triples)) > 0.5
        for row, (h, r, t) in enumerate(triples):
            direction = "head" if corrupt_heads[row] else "tail"
            key = (int(t if corrupt_heads[row] else h), int(r))
            blocked, count = self.index[direction].get(key, (np.empty(0, dtype=np.int64), 0))
            available = self.num_entities - count
            if available < negatives:
                raise ValueError(f"Only {available} admissible {direction} negatives for {key}, need {negatives}")
            ranks = self.rng.choice(available, size=negatives, replace=False)
            ids = ranks + np.searchsorted(blocked, ranks, side="right")
            output[row, :, 0 if corrupt_heads[row] else 2] = ids
        return torch.from_numpy(output.reshape(-1, 3))


def select_validation(triples, limit=None, seed=260906):
    import numpy as np
    import torch
    if limit is not None and limit <= 0:
        raise ValueError("Validation limit must be positive or None")
    total = len(triples)
    indices = (np.arange(total, dtype=np.int64) if limit is None or limit >= total
               else np.sort(np.random.default_rng(seed).choice(total, limit, replace=False)))
    metadata = {
        "validation_scope": "full" if len(indices) == total else "subset",
        "validation_count": len(indices), "validation_total_count": total,
        "validation_indices": indices.tolist(),
        "subset_indices_hash": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "validation_selection_seed": seed,
    }
    return triples[torch.from_numpy(indices)].clone(), metadata


def evaluate_strict(model, triples, filters, device, query_batch=32, candidate_chunk=256):
    """All-entity, both-direction filtered ranks with average exact-score ties.

    The true candidate is scored in the same pass/shape as every other
    candidate, then explicitly excluded from greater-than and tie counts.
    Only the score matrix (queries x entities) is retained, not geometry graphs.
    """
    import torch
    if query_batch <= 0 or candidate_chunk <= 0:
        raise ValueError("Evaluation chunk sizes must be positive")
    if (not isinstance(triples, torch.Tensor) or triples.device.type != "cpu"
            or triples.dtype != torch.long or triples.ndim != 2 or triples.shape[1] != 3
            or len(triples) == 0):
        raise ValueError("Evaluation needs nonempty CPU int64 [N,3] triples")
    if (triples < 0).any() or (triples[:, (0, 2)] >= model.num_entities).any():
        raise ValueError("Evaluation triple contains invalid IDs")
    if hasattr(model, "num_relations") and (triples[:, 1] >= model.num_relations).any():
        raise ValueError("Evaluation triple contains an invalid relation ID")
    device = torch.device(device)
    devices = ([torch.cuda.current_device() if device.index is None else device.index]
               if device.type == "cuda" else [])
    was_training = model.training
    ranks = {"head": [], "tail": []}
    start = time.monotonic()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            all_ids = torch.arange(model.num_entities, device=device)
            for offset in range(0, len(triples), query_batch):
                batch_cpu = triples[offset:offset + query_batch]
                batch = batch_cpu.to(device)
                size = len(batch)
                for direction in ("tail", "head"):
                    scores = None
                    for left in range(0, model.num_entities, candidate_chunk):
                        candidates = all_ids[left:left + candidate_chunk]
                        width = len(candidates)
                        cid = candidates[None].expand(size, width)
                        rid = batch[:, 1, None].expand(size, width)
                        hid = cid if direction == "head" else batch[:, 0, None].expand(size, width)
                        tid = cid if direction == "tail" else batch[:, 2, None].expand(size, width)
                        values = model(hid.reshape(-1), rid.reshape(-1), tid.reshape(-1))
                        if values.ndim != 1 or values.numel() != size * width:
                            raise ValueError("Model must return one flat score per triple")
                        if not values.is_floating_point() or values.device != batch.device:
                            raise ValueError("Scores must be floating tensors on the evaluation device")
                        if not torch.isfinite(values).all():
                            raise FloatingPointError("Nonfinite score before evaluation filtering")
                        if scores is None:
                            scores = torch.empty((size, model.num_entities), dtype=values.dtype, device=device)
                        elif scores.dtype != values.dtype:
                            raise ValueError("Score dtype changed between chunks")
                        scores[:, left:left + width] = values.reshape(size, width)
                    target_column, fixed_column = ((0, 2) if direction == "head" else (2, 0))
                    targets = batch[:, target_column]
                    target_scores = scores.gather(1, targets[:, None]).clone()
                    for row, triple in enumerate(batch_cpu.tolist()):
                        blocked = filters[direction].get((triple[fixed_column], triple[1]), set())
                        if blocked:
                            excluded = list(blocked)
                            if min(excluded) < 0 or max(excluded) >= model.num_entities:
                                raise ValueError("Evaluation filter contains invalid entity IDs")
                            scores[row, excluded] = -torch.inf
                    # Even if not present in the supplied mask, exclude target self.
                    scores[torch.arange(size, device=device), targets] = -torch.inf
                    better = (scores > target_scores).sum(dim=1)
                    tied_others = (scores == target_scores).sum(dim=1)
                    values = 1.0 + better.to(torch.float64) + 0.5 * tied_others
                    ranks[direction].extend(values.cpu().tolist())
    finally:
        model.train(was_training)
    combined = ranks["tail"] + ranks["head"]
    return {
        "mrr": sum(1.0 / rank for rank in combined) / len(combined),
        "hits": {str(k): sum(rank <= k for rank in combined) / len(combined) for k in (1, 3, 10)},
        "ranks_head": ranks["head"], "ranks_tail": ranks["tail"],
        "evaluated_triples": len(triples), "directional_queries": len(combined),
        "num_entities": model.num_entities, "elapsed_seconds": time.monotonic() - start,
        "protocol": PROTOCOL, "tie_policy": TIE_POLICY,
    }


def check_health(value, prefix="diagnostics"):
    """Finite diagnostics only; mathematical validity is owned by V2 model.

    In particular, the old Cayley/Gregory local-series thresholds are NOT used.
    Model diagnostics must raise on branch/reconstruction failures.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            check_health(child, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            check_health(child, f"{prefix}/{index}")
    elif isinstance(value, bool):
        if prefix.endswith(("parameters_finite", "sampled_health_passed")) and not value:
            raise FloatingPointError(f"Failed health check: {prefix}")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite diagnostic: {prefix}")


def check_diagnostic_backend(value, backend, log_order, geometry):
    """Require actual model diagnostics to identify the requested log track."""
    terms = 12 if backend == "gregory12" else log_order
    contract = value.get("model_contract", {}) if isinstance(value, dict) else {}
    if (not isinstance(value, dict) or value.get("log_backend") != backend
            or value.get("geometry") != geometry or not isinstance(contract, dict)
            or contract.get("log_backend") != backend or contract.get("log_terms") != terms):
        raise RuntimeError("Model diagnostic backend/order differs from the requested configuration")
    if geometry == "sl8":
        principal = value.get("principal_log", {})
        if not isinstance(principal, dict) or principal.get("backend") != backend:
            raise RuntimeError("SL principal-log diagnostic reports a different backend")


def initial_score_statistics(model, positive_cpu, negative_cpu, device, alpha, offset):
    """Read-only, train-only score/gradient preflight before any update."""
    import torch
    result = {"scope": "train_only_sample_before_any_update", "initial_alpha": alpha,
              "initial_offset": offset}
    was_training = model.training
    try:
        with torch.no_grad():
            model.eval()
            for label, triples in (("positive", positive_cpu), ("negative", negative_cpu)):
                triples = triples.to(device)
                scores = model(triples[:, 0], triples[:, 1], triples[:, 2])
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite initial preflight scores")
                distances = (offset - scores) / alpha
                row = {"count": len(triples)}
                for name, values in (("score", scores), ("distance", distances)):
                    row[name] = {"mean": float(values.mean()), "std": float(values.std(unbiased=False)),
                                 "median": float(values.median()), "min": float(values.min()),
                                 "max": float(values.max())}
                probabilities = scores.sigmoid()
                gradient = 1 - probabilities if label == "positive" else probabilities
                row["unreduced_logistic_gradient_magnitude_mean"] = float(gradient.mean())
                row["sigmoid_probability_mean"] = float(probabilities.mean())
                result[label] = row
    finally:
        model.train(was_training)
    return result


def main(argv=None):
    args = parse_args(argv)
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    state = {
        "status": "starting", "model": args.model, "geometry": args.model, "dataset": "WN9-IMG",
        "kind": args.kind, "seed": args.seed, "started_at": utc(), "pid": os.getpid(),
        "protocol": PROTOCOL, "evaluation_protocol": PROTOCOL, "tie_policy": TIE_POLICY,
        "training_negative_filter": "train_only_directional", "validation_filter": "all_splits_directional",
        "negative_sampling": "uniform_without_replacement_per_triple",
        "loss": "author_logistic_mean_over_1_positive_and_N_negatives",
        "score_distance_power": 1, "optimizer": "Adagrad", "test_evaluated": False,
        "log_backend": args.log_backend, "log_terms": 12 if args.log_backend == "gregory12" else args.log_order,
        "warning": "New strict-filter protocol; old author-release scores are not directly comparable.",
        "args": config, "config": config, "completed_epochs": 0,
        "best_epoch": None, "best_validation_mrr": None,
    }
    save_json(out / "config.json", config)
    save_json(out / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        repo, core = setup_paths(args.root)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if commit != COMMIT:
            raise RuntimeError("Pinned author commit changed")
        import torch
        from vlkge import helpers, utils
        from geometry.wn9_geometry_v2 import WN9GeometryV2
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing unintended CPU training")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        utils.set_seed(args.seed)
        _, _, triples, model_args, manifest = load_inputs(args.root, repo)
        save_json(out / "input_manifest.json", manifest)
        source_files = list((args.root / "geometry").rglob("*.py"))
        source_files += [Path(__file__).resolve(), Path(__file__).with_name("run_geometry.py")]
        source_files += list((core / "sl_manifold").glob("*.py"))
        source_files += list((repo / "vlkge").rglob("*.py"))
        save_json(out / "source_sha256.json", {str(path): digest(path) for path in sorted(set(source_files))})
        selected, selection = select_validation(triples[1], args.eval_limit, args.validation_seed)
        state.update(selection)
        state.update(upstream_commit=commit, python=platform.python_version(), torch=torch.__version__,
                     gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                     split_sizes=[len(split) for split in triples], num_entities=model_args["num_entities"])
        save_json(out / "validation_selection.json", selection)
        train_filters = directional_filters(triples[0])
        ranking_filters = directional_filters(torch.cat(triples))
        sampler = TrainNegativeSampler(model_args["num_entities"], train_filters, args.seed)
        shuffle_generator = torch.Generator().manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        model = WN9GeometryV2(
            **model_args, device=device, geometry=args.model, coordinate_scale=args.coordinate_scale,
            chart_radius=args.chart_radius, initial_logit_scale=args.initial_logit_scale,
            initial_offset=args.initial_offset, score_chunk=args.score_chunk,
            checkpoint_blocks=True, log_order=args.log_order, log_backend=args.log_backend,
        ).to(device)
        train_entity_ids = torch.unique(triples[0][:, (0, 2)].reshape(-1)).to(device)
        initialization = model.initialize_geometry(train_entity_ids, target_norm=args.target_init_norm)
        # Probe positives AND train-only sampled negatives without advancing the
        # real training sampler/shuffle RNG. Diagnostics are sampled guarantees.
        probe_positive = triples[0][:min(16, len(triples[0]))]
        probe_sampler = TrainNegativeSampler(model.num_entities, train_filters, args.seed + 1000003)
        probe = torch.cat((probe_positive, probe_sampler.sample(probe_positive, 1))).to(device)
        preflight_positive = triples[0][:min(64, len(triples[0]))]
        preflight_sampler = TrainNegativeSampler(model.num_entities, train_filters, args.seed + 1000004)
        preflight_negative = preflight_sampler.sample(preflight_positive, min(args.negatives, 8))
        initial_scores = initial_score_statistics(model, preflight_positive, preflight_negative, device,
                                                  args.initial_logit_scale, args.initial_offset)

        def diagnostics():
            with torch.no_grad():
                value = model.diagnostics(probe[:, 0], probe[:, 1], probe[:, 2])
            check_diagnostic_backend(value, args.log_backend, args.log_order, args.model)
            check_health(value)
            return value

        state.update(initialization=initialization, initial_diagnostics=diagnostics(), initial_scores=initial_scores,
                     diagnostics_scope="fixed_train_positive_and_train_negative_probe_not_all_pairs",
                     parameters_total=sum(p.numel() for p in model.parameters()),
                     parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     status="running")
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        save_json(out / "status.json", state)
        print(json.dumps({"configuration": config, "parameters_trainable": state["parameters_trainable"]}), flush=True)
        best, best_epoch, stale = -math.inf, 0, 0
        expected_batches = math.ceil(len(triples[0]) / args.batch_size)
        state["training_complete"] = args.max_train_batches is None or args.max_train_batches >= expected_batches
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            order = torch.randperm(len(triples[0]), generator=shuffle_generator)
            loss_sum, examples, num_batches, max_grad_norm = 0.0, 0, 0, 0.0
            first_negative_hash = None
            for left in range(0, len(order), args.batch_size):
                positive_cpu = triples[0][order[left:left + args.batch_size]]
                negative_cpu = sampler.sample(positive_cpu, args.negatives)
                if first_negative_hash is None:
                    first_negative_hash = hashlib.sha256(negative_cpu.numpy().astype("<i8").tobytes()).hexdigest()
                positive, negative = positive_cpu.to(device), negative_cpu.to(device)
                optimizer.zero_grad(set_to_none=True)
                positive_scores = model(positive[:, 0], positive[:, 1], positive[:, 2])
                negative_scores = model(negative[:, 0], negative[:, 1], negative[:, 2]).view(len(positive), args.negatives)
                if not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
                    raise FloatingPointError("Nonfinite training scores")
                loss = helpers.compute_logistic_loss(positive_scores, negative_scores)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                max_grad_norm = max(max_grad_norm, float(norm))
                optimizer.step()
                loss_sum += float(loss.detach()) * len(positive)
                examples += len(positive)
                num_batches += 1
                if args.max_train_batches is not None and num_batches >= args.max_train_batches:
                    break
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - epoch_start
            health = diagnostics()
            validation = evaluate_strict(model, selected, ranking_filters, device, args.query_batch, args.candidate_chunk)
            if args.kind == "pilot" and state["validation_scope"] != "full":
                raise RuntimeError("Pilot validation must use all 1337 validation triples")
            event = {
                "epoch": epoch, "time": utc(), "train_loss": loss_sum / examples,
                "train_examples": examples, "train_batches": num_batches,
                "train_seconds": train_seconds, "epoch_seconds": time.monotonic() - epoch_start,
                "validation": {key: value for key, value in validation.items() if not key.startswith("ranks_")},
                "validation_scope": state["validation_scope"], "subset_indices_hash": state["subset_indices_hash"],
                "log_backend": args.log_backend, "log_terms": state["log_terms"],
                "diagnostics": health, "max_gradient_norm": max_grad_norm,
                "first_batch_negative_sha256": first_negative_hash,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            with (out / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if validation["mrr"] > best:
                best, best_epoch, stale = validation["mrr"], epoch, 0
                checkpoint = {
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "score": best, "args": config, "protocol": PROTOCOL,
                    "log_backend": args.log_backend, "log_terms": state["log_terms"],
                    "validation_selection": selection, "validation": event["validation"],
                    "shuffle_generator_state": shuffle_generator.get_state(),
                    "negative_rng_state": sampler.rng.bit_generator.state,
                    "cpu_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                }
                torch.save(checkpoint, out / "best.pt.tmp")
                os.replace(out / "best.pt.tmp", out / "best.pt")
            else:
                stale += 1
            state.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_mrr=best,
                         last_epoch=event, elapsed_seconds=time.monotonic() - start)
            save_json(out / "status.json", state)
            print(f"epoch={epoch} val_{state['validation_scope']}_mrr={validation['mrr']:.8f} "
                  f"best={best:.8f}@{best_epoch} train={train_seconds:.1f}s "
                  f"total={event['epoch_seconds']:.1f}s peak={event['peak_allocated_gib']:.2f}GiB", flush=True)
            if stale >= args.patience:
                break
        # Recheck the selected state, but NEVER evaluate test in this runner.
        saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        state["best_checkpoint_diagnostics"] = diagnostics()
        state.update(status="completed_validation_only", ended_at=utc(), elapsed_seconds=time.monotonic() - start,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
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
