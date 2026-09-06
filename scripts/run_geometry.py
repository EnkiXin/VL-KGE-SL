"""Separate WN9 geometry experiment; the locked author checkout stays untouched.

The author-release split, mixed-direction filter, loss and negative sampler are
preserved intentionally. This is an adapted-model experiment, not original MuRP
reproduction. Pilots never evaluate test; formal runs test the val-best state once.
"""
import argparse
from datetime import datetime, timezone
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
from types import SimpleNamespace

COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", choices=["euclidean", "mure", "murp", "sl8"], required=True)
    p.add_argument("--kind", choices=["smoke", "pilot", "formal"], default="pilot")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--coordinate-scale", type=float, default=0.1)
    p.add_argument("--chart-radius", type=float, default=0.5)
    p.add_argument("--initial-logit-scale", type=float, default=100.0)
    p.add_argument("--initial-offset", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--negatives", type=int, default=100)
    p.add_argument("--score-chunk", type=int, default=2048)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--candidate-chunk", type=int, default=512)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--eval-limit", type=int)
    args = p.parse_args()
    for name in ["epochs", "patience", "batch_size", "negatives", "score_chunk", "query_batch", "candidate_chunk"]:
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    for name in ["lr", "coordinate_scale", "chart_radius", "initial_logit_scale", "grad_clip"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    if not math.isfinite(args.initial_offset):
        p.error("initial_offset must be finite")
    for name in ["max_train_batches", "eval_limit"]:
        value = getattr(args, name)
        if value is not None and (value <= 0 or args.kind != "smoke"):
            p.error(f"{name} is a positive smoke-only option")
    return args


def setup_paths(root):
    root = root.resolve()
    repo = root / "upstream/vl-kge"
    candidates = [root / "vendor/sl-manifold-core/src", root.parent / "sl-manifold-core/src"]
    core = next((x for x in candidates if (x / "sl_manifold/core.py").is_file()), None)
    if core is None:
        raise FileNotFoundError("Missing shared SL core; deploy its src to vendor/sl-manifold-core/src")
    for x in [repo, root, core]:
        sys.path.insert(0, str(x))
    return repo, core


def load_inputs(root, repo):
    """Pinned author CLIP inputs, author entity IDs, author split and filters."""
    import torch
    from vlkge import utils
    from vlkge.dataloader import KnowledgeGraphDataLoader
    manifest_path = root / "artifacts/wn9-inputs.json"
    if not manifest_path.exists():
        manifest_path = root / "artifacts/wn9-inputs-local.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        path = repo / record["path"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {path}")
    base = repo / "vlkge/data/wn9_img"
    data = KnowledgeGraphDataLoader(base / "wn9_img_triples.csv", "wn9_img", bidirectional_eval=True)
    entity_ids, relation_ids = data.get_entities_and_relations()
    feature_args = SimpleNamespace(use_visual=True, use_textual=True, use_relation_features=False,
                                  normalize_visual=False, normalize_textual=False,
                                  visual_features_path=str(base / "features/wn9_img_vf_clip.pkl"),
                                  textual_features_path=str(base / "features/wn9_img_tf_clip.pkl"))
    visual, textual, _, vmapping, tmapping = utils.load_features(feature_args, entity_ids, relation_ids)
    frames = data.split_data()
    tensors = []
    for frame in frames:
        tensors.append(torch.tensor(list(zip(frame["head"].map(entity_ids),
                                             frame["relation"].map(relation_ids),
                                             frame["tail"].map(entity_ids))), dtype=torch.long))
    if [len(x) for x in tensors] != [11741, 1337, 1319] or len(entity_ids) != 6555:
        raise RuntimeError("Unexpected WN9 split or entity counts")
    model_args = dict(num_entities=len(entity_ids), num_relations=len(relation_ids), embedding_dim=768,
                      visual_features=visual, textual_features=textual,
                      visual_entity_to_index=vmapping, textual_entity_to_index=tmapping,
                      fusion_mode="average", use_structural=True, use_visual=True, use_textual=True,
                      freeze_visual=True, freeze_textual=True, visual_proj=False, textual_proj=False,
                      inductive=False, modality_asymmetry=False, normalize_before_fusion=False)
    return data, frames, tensors, model_args, manifest


def main():
    args = parse_args()
    root, out = args.root.resolve(), args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state = dict(status="starting", model=args.model, kind=args.kind, started_at=utc(),
                 pid=os.getpid(), protocol="author_release_protocol_geometry_v1",
                 warning="Adapted geometry models; author all-split/mixed-direction filter preserved. Not strict-protocol evidence.",
                 args=vars(args), completed_epochs=0)
    save_json(out / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        repo, core = setup_paths(root)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if commit != COMMIT:
            raise RuntimeError("Author commit changed")
        import torch
        from vlkge import helpers, utils
        from vlkge.dataloader import KGDataset
        from geometry.models import VLGeometry
        from geometry.evaluation import evaluate_full
        from torch.utils.data import DataLoader
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing unintended CPU training")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        utils.set_seed(args.seed)
        data, frames, triples, model_args, manifest = load_inputs(root, repo)
        save_json(out / "input_manifest.json", manifest)
        save_json(out / "config.json", vars(args))
        source_files = list((root / "geometry").rglob("*.py")) + [Path(__file__).resolve()]
        source_files += list((core / "sl_manifold").glob("*.py"))
        source_files += list((repo / "vlkge").rglob("*.py"))
        save_json(out / "source_sha256.json", {str(x): digest(x) for x in sorted(source_files)})
        state.update(upstream_commit=commit, python=platform.python_version(), torch=torch.__version__,
                     gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                     split_sizes=[len(x) for x in triples], num_entities=model_args["num_entities"])
        model = VLGeometry(**model_args, device=device, geometry=args.model,
                           coordinate_scale=args.coordinate_scale, chart_radius=args.chart_radius,
                           score_chunk=args.score_chunk, checkpoint_blocks=True,
                           initial_logit_scale=args.initial_logit_scale, initial_offset=args.initial_offset).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        # Author shuffling and negative sampling share the same explicit CPU RNG.
        generator = torch.Generator().manual_seed(args.seed)
        entity_ids, relation_ids = data.get_entities_and_relations()
        loader = DataLoader(KGDataset(frames[0], entity_ids, relation_ids),
                            batch_size=args.batch_size, shuffle=True, generator=generator)
        filter_map, relation_probs = data.compute_filter_map(), data.compute_relation_probs()
        state.update(parameters_total=sum(p.numel() for p in model.parameters()),
                     parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     status="running")
        save_json(out / "status.json", state)
        print(json.dumps({"configuration": vars(args), "parameters_trainable": state["parameters_trainable"]}, default=str), flush=True)
        torch.cuda.reset_peak_memory_stats()
        best, best_epoch, stale = -math.inf, 0, 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            loss_sum, num_batches = 0.0, 0
            max_grad_norm = 0.0
            for batch in loader:
                heads, relations, tails = [batch[k].to(device) for k in ["head_id", "relation_id", "tail_id"]]
                optimizer.zero_grad(set_to_none=True)
                pos_scores = model(heads, relations, tails)
                nh, nr, nt = helpers.negative_sampling_uniform(heads, relations, tails, model.num_entities,
                                                              args.negatives, filter_map, relation_probs,
                                                              generator, use_bernoulli=False)
                neg_scores = model(nh, nr, nt).view(heads.shape[0], args.negatives)
                if not torch.isfinite(pos_scores).all() or not torch.isfinite(neg_scores).all():
                    raise FloatingPointError("Nonfinite training score")
                loss = helpers.compute_logistic_loss(pos_scores, neg_scores)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                max_grad_norm = max(max_grad_norm, float(norm))
                optimizer.step()
                loss_sum += float(loss.detach())
                num_batches += 1
                if args.max_train_batches is not None and num_batches >= args.max_train_batches:
                    break
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - epoch_start
            val = evaluate_full(model, triples[1], filter_map, device,
                                query_batch=args.query_batch, candidate_chunk=args.candidate_chunk,
                                max_queries=args.eval_limit)
            if args.kind != "smoke" and not val["complete"]:
                raise RuntimeError("Pilot/formal validation must be complete")
            diagnostics = model.diagnostics()
            # Concrete risk checks are added by the model diagnostics contract.
            def check_health(value, prefix=""):
                if isinstance(value, dict):
                    for k, v in value.items():
                        check_health(v, prefix + "/" + k)
                elif isinstance(value, bool):
                    if prefix.endswith("parameters_finite") and not value:
                        raise FloatingPointError("Nonfinite model parameters")
                elif isinstance(value, (float, int)):
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Nonfinite diagnostic {prefix}: {value}")
                    if ("nonfinite_fraction" in prefix or "branch_risk_fraction" in prefix) and value > 0:
                        raise FloatingPointError(f"Unsafe geometry diagnostic {prefix}: {value}")
                    if prefix.endswith("max_gregory_frobenius_remainder_bound") and value > 1e-3:
                        raise FloatingPointError(f"Local-log error bound too large {prefix}: {value}")
            check_health(diagnostics)
            event = dict(epoch=epoch, time=utc(), train_loss=loss_sum / num_batches,
                         train_seconds=train_seconds, epoch_seconds=time.monotonic()-epoch_start,
                         validation={k: v for k, v in val.items() if not k.startswith("ranks_")},
                         diagnostics=diagnostics, max_gradient_norm=max_grad_norm,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            with (out / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if val["mrr"] > best:
                best, best_epoch, stale = val["mrr"], epoch, 0
                checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                  epoch=epoch, score=best, args=vars(args),
                                  generator_state=generator.get_state(), cpu_rng_state=torch.get_rng_state(),
                                  cuda_rng_states=torch.cuda.get_rng_state_all())
                torch.save(checkpoint, out / "best.pt.tmp")
                os.replace(out / "best.pt.tmp", out / "best.pt")
            else:
                stale += 1
            state.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_mrr=best,
                         last_epoch=event, elapsed_seconds=time.monotonic()-start)
            save_json(out / "status.json", state)
            print(f"epoch={epoch} loss={event['train_loss']:.6f} val_mrr={val['mrr']:.8f} "
                  f"best={best:.8f}@{best_epoch} train={train_seconds:.1f}s total={event['epoch_seconds']:.1f}s "
                  f"peak={event['peak_allocated_gib']:.2f}GiB", flush=True)
            if stale >= args.patience:
                break
        if args.kind == "formal":
            saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            final_test = evaluate_full(model, triples[2], filter_map, device,
                                       query_batch=args.query_batch, candidate_chunk=args.candidate_chunk)
            save_json(out / "test_ranks.json", final_test)
            state["final_test"] = {k: v for k, v in final_test.items() if not k.startswith("ranks_")}
            state["best_checkpoint_diagnostics"] = model.diagnostics()
            check_health(state["best_checkpoint_diagnostics"])
        state.update(status="completed", ended_at=utc(), elapsed_seconds=time.monotonic()-start,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save_json(out / "result.json", state)
        save_json(out / "status.json", state)
        print(f"COMPLETED {out / 'result.json'}", flush=True)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=repr(exc), traceback=traceback.format_exc(), ended_at=utc(),
                     elapsed_seconds=time.monotonic()-start)
        save_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
