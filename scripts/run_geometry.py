"""Geometry experiments on the three author datasets; locked checkout untouched.

The author-release split, mixed-direction filter, loss and negative sampler are
preserved intentionally. By default pilots select on validation. Explicit test
selection is exploratory and is labelled test-tuned, never held-out evidence.
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


def selection_policy(split):
    """Return explicit labels/fields; never store test-selected scores as validation."""
    if split not in ("validation", "test"):
        raise ValueError("selection split must be validation or test")
    is_test = split == "test"
    return dict(
        tensor_index=2 if is_test else 1,
        event_key="test_selection" if is_test else "validation",
        best_metric_key="best_test_mrr" if is_test else "best_validation_mrr",
        metadata=dict(selection_split=split, test_used_for_selection=is_test,
                      evaluation_label="test_tuned_not_held_out" if is_test else "validation_selected"),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", choices=["euclidean", "mure", "murp", "sl8", "slv2", "euclidv2"], required=True)
    p.add_argument("--dataset", choices=["wn9_img", "wikiart_mkg_v1", "wikiart_mkg_v2"], default="wn9_img")
    p.add_argument("--kind", choices=["smoke", "pilot", "formal"], default="pilot")
    p.add_argument("--selection-split", choices=["validation", "test"], default="validation")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--coordinate-scale", type=float, default=0.1)
    p.add_argument("--chart-radius", type=float, default=0.5)
    p.add_argument("--initial-logit-scale", type=float, default=100.0)
    p.add_argument("--initial-offset", type=float, default=0.0)
    p.add_argument("--embedding-dim", type=int, default=768,
                   help="Author entity/fusion width; pretrained modalities use the author's dimension alignment")
    # Geometry v2 options (ignored by the v1 models).
    p.add_argument("--matrix-dim", type=int, default=8, help="v2: SL(n) matrix size; coordinates = n*n-1")
    p.add_argument("--entity-mapping", choices=["linear", "fixed_pad", "direct"], default="linear",
                   help="v2: learned projection, fixed zero padding, or direct same-width coordinates")
    p.add_argument("--relation-radius", type=float, default=1.5, help="v2: Frobenius clip radius of relation coordinates")
    p.add_argument("--relation-init-norm", type=float, default=0.5, help="v2: expected initial relation coordinate norm")
    p.add_argument("--relation-mode", choices=["left", "sandwich"], default="left")
    p.add_argument("--score-form", choices=["linear", "squared"], default="linear")
    p.add_argument("--schatten-p", type=float, default=2.0)
    p.add_argument("--log-sqrt-steps", type=int, default=1)
    p.add_argument("--log-db-iterations", type=int, default=6)
    p.add_argument("--log-terms", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--negatives", type=int, default=None)
    p.add_argument("--score-chunk", type=int, default=2048)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--candidate-chunk", type=int, default=512)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--validate-every", type=int, default=1,
                   help="Run full validation every k epochs (and at the final epoch); patience counts epochs")
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--eval-limit", type=int)
    args = p.parse_args(argv)
    # Preserve legacy WN9 defaults; WikiArt follows the released YAML unless overridden.
    defaults = {"wn9_img": (10, 100, 0.01, 50),
                "wikiart_mkg_v1": (50, 1, 0.1, None),
                "wikiart_mkg_v2": (20, 1, 0.1, None)}[args.dataset]
    for name, value in zip(("epochs", "negatives", "lr", "patience"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ["epochs", "batch_size", "negatives", "score_chunk", "query_batch", "candidate_chunk",
                 "validate_every", "embedding_dim"]:
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.patience is not None and args.patience <= 0:
        p.error("patience must be positive")
    for name in ["lr", "coordinate_scale", "chart_radius", "initial_logit_scale", "grad_clip",
                 "relation_radius", "schatten_p"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    if not math.isfinite(args.relation_init_norm) or args.relation_init_norm < 0:
        p.error("relation_init_norm must be finite and non-negative")
    if args.matrix_dim < 2 or args.log_sqrt_steps < 0 or args.log_db_iterations < 1 or args.log_terms < 1:
        p.error("invalid v2 matrix/logarithm settings")
    if args.entity_mapping != "direct" and args.embedding_dim != 768:
        p.error("linear/fixed_pad adapters require embedding_dim=768; use direct for a different width")
    if args.entity_mapping in ("fixed_pad", "direct"):
        if args.model not in ("slv2", "euclidv2"):
            p.error(f"--entity-mapping {args.entity_mapping} requires a v2 model")
        coordinate_dim = args.matrix_dim * args.matrix_dim - 1
        if args.entity_mapping == "fixed_pad" and coordinate_dim < args.embedding_dim:
            p.error("--entity-mapping fixed_pad requires matrix_dim**2 - 1 >= embedding_dim")
        if args.entity_mapping == "direct" and coordinate_dim != args.embedding_dim:
            p.error("--entity-mapping direct requires embedding_dim == matrix_dim**2 - 1")
    if args.model in ("slv2", "euclidv2") and args.chart_radius <= 0.5 and args.kind != "smoke":
        p.error("v2 models are meant to run with --chart-radius above 0.5; the v1 bound made SL Euclidean")
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


def load_inputs(root, repo, embedding_dim=768, dataset="wn9_img", profile=None):
    """Pinned author CLIP inputs, author entity IDs, author split and filters."""
    import torch
    import numpy as np
    from vlkge import utils
    from vlkge.dataloader import KnowledgeGraphDataLoader
    from geometry.datasets import author_profile, loader_options, verified_manifest
    profile = author_profile(repo, dataset) if profile is None else profile
    manifest = verified_manifest(root, repo, dataset)
    base = repo / "vlkge/data" / dataset
    data = KnowledgeGraphDataLoader(base / f"{dataset}_triples.csv", dataset, **loader_options(profile))
    entity_ids, relation_ids = data.get_entities_and_relations()
    feature_args = SimpleNamespace(use_visual=True, use_textual=True, use_relation_features=False,
                                  normalize_visual=False, normalize_textual=False,
                                  visual_features_path=str(base / "features" / f"{dataset}_vf_clip.pkl"),
                                  textual_features_path=str(base / "features" / f"{dataset}_tf_clip.pkl"))
    visual, textual, _, vmapping, tmapping = utils.load_features(feature_args, entity_ids, relation_ids)
    frames = data.split_data()
    tensors = []
    for frame in frames:
        values = np.column_stack((frame["head"].map(entity_ids).to_numpy(dtype=np.int64),
                                  frame["relation"].map(relation_ids).to_numpy(dtype=np.int64),
                                  frame["tail"].map(entity_ids).to_numpy(dtype=np.int64)))
        tensors.append(torch.from_numpy(values))
    if dataset == "wn9_img" and ([len(x) for x in tensors] != [11741, 1337, 1319] or len(entity_ids) != 6555):
        raise RuntimeError("Unexpected WN9 split or entity counts")
    if any(len(frame) == 0 for frame in frames) or len(entity_ids) == 0:
        raise RuntimeError("Author dataset has an empty split or no entities")
    model_args = dict(num_entities=len(entity_ids), num_relations=len(relation_ids), embedding_dim=embedding_dim,
                      visual_features=visual, textual_features=textual,
                      visual_entity_to_index=vmapping, textual_entity_to_index=tmapping,
                      fusion_mode="average", use_structural=True, use_visual=True, use_textual=True,
                      freeze_visual=True, freeze_textual=True, visual_proj=False, textual_proj=False,
                      inductive=profile.get("inductive", False),
                      modality_asymmetry=profile.get("modality_asymmetry", False), normalize_before_fusion=False)
    return data, frames, tensors, model_args, manifest


def main():
    args = parse_args()
    root, out = args.root.resolve(), args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    is_v2 = args.model in ("slv2", "euclidv2")
    selection = selection_policy(args.selection_split)
    selection_metadata = selection["metadata"]
    state = dict(status="starting", model=args.model, dataset=args.dataset, kind=args.kind, started_at=utc(),
                 pid=os.getpid(), protocol="author_release_protocol_geometry_v2" if is_v2 else "author_release_protocol_geometry_v1",
                 warning="Adapted geometry models; author all-split/mixed-direction filter preserved. Not strict-protocol evidence.",
                 args=vars(args), completed_epochs=0, **selection_metadata)
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
        from geometry.models_v2 import VLGeometryV2
        from geometry.evaluation import evaluate_full
        from geometry.datasets import (author_profile, mark_training_entities, epoch_training_frame,
                                       sample_negatives, check_v1_negative_capacity)
        from torch.utils.data import DataLoader
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing unintended CPU training")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        utils.set_seed(args.seed)
        profile = author_profile(repo, args.dataset)
        data, frames, triples, model_args, manifest = load_inputs(
            root, repo, embedding_dim=args.embedding_dim, dataset=args.dataset, profile=profile)
        save_json(out / "input_manifest.json", manifest)
        save_json(out / "dataset_protocol.json", profile)
        save_json(out / "config.json", vars(args))
        source_files = list((root / "geometry").rglob("*.py")) + [Path(__file__).resolve()]
        source_files += list((core / "sl_manifold").glob("*.py"))
        source_files += list((repo / "vlkge").rglob("*.py"))
        source_files += [Path(profile["config_path"])]
        save_json(out / "source_sha256.json", {str(x): digest(x) for x in sorted(source_files)})
        state.update(upstream_commit=commit, python=platform.python_version(), torch=torch.__version__,
                     gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                     split_sizes=[len(x) for x in triples], num_entities=model_args["num_entities"],
                     protocol_notes=profile["protocol_notes"],
                     bidirectional_eval=profile.get("bidirectional_eval", True),
                     per_relation_candidates=profile.get("use_per_relation_candidates", False))
        if is_v2:
            model = VLGeometryV2(**model_args, device=device,
                                 geometry="sl" if args.model == "slv2" else "euclidean",
                                 matrix_dim=args.matrix_dim, entity_mapping=args.entity_mapping,
                                 coordinate_scale=args.coordinate_scale,
                                 chart_radius=args.chart_radius, relation_radius=args.relation_radius,
                                 relation_init_norm=args.relation_init_norm, relation_mode=args.relation_mode,
                                 score_form=args.score_form, initial_logit_scale=args.initial_logit_scale,
                                 initial_offset=args.initial_offset, schatten_p=args.schatten_p,
                                 log_sqrt_steps=args.log_sqrt_steps, log_db_iterations=args.log_db_iterations,
                                 log_terms=args.log_terms, score_chunk=args.score_chunk,
                                 checkpoint_blocks=True).to(device)
        else:
            model = VLGeometry(**model_args, device=device, geometry=args.model,
                               coordinate_scale=args.coordinate_scale, chart_radius=args.chart_radius,
                               score_chunk=args.score_chunk, checkpoint_blocks=True,
                               initial_logit_scale=args.initial_logit_scale, initial_offset=args.initial_offset).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        # Author shuffling and negative sampling share the same explicit CPU RNG.
        generator = torch.Generator().manual_seed(args.seed)
        entity_ids, relation_ids = data.get_entities_and_relations()
        seen_count = mark_training_entities(model, frames[0], entity_ids)
        loader = (None if profile.get("downsample_relation") is not None else
                  DataLoader(KGDataset(frames[0], entity_ids, relation_ids),
                             batch_size=args.batch_size, shuffle=True, generator=generator))
        filter_map, relation_probs = data.compute_filter_map(), data.compute_relation_probs()
        use_pools = profile.get("use_per_relation_candidates", False)
        training_pool = data.relation_to_valid_tails_train if use_pools else None
        validation_pool = data.relation_to_valid_tails_eval_val if use_pools else None
        test_pool = data.relation_to_valid_tails_eval_test if use_pools else None
        selection_triples = triples[selection["tensor_index"]]
        selection_pool = validation_pool if args.selection_split == "validation" else test_pool
        selection_key, best_metric_key = selection["event_key"], selection["best_metric_key"]
        if args.dataset == "wikiart_mkg_v1":
            check_v1_negative_capacity(frames[0], entity_ids, relation_ids, filter_map, training_pool, args.negatives)
        state.update(parameters_total=sum(p.numel() for p in model.parameters()),
                     parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     seen_training_entities=seen_count,
                     status="running")
        save_json(out / "status.json", state)
        print(json.dumps({"configuration": vars(args), "parameters_trainable": state["parameters_trainable"]}, default=str), flush=True)
        torch.cuda.reset_peak_memory_stats()
        best, best_epoch, stale = -math.inf, 0, 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            if profile.get("downsample_relation") is not None:
                epoch_frame = epoch_training_frame(frames[0], profile, args.seed, epoch - 1)
                loader = DataLoader(KGDataset(epoch_frame, entity_ids, relation_ids),
                                    batch_size=args.batch_size, shuffle=True, generator=generator)
                print(f"epoch={epoch} effective_train_triples={len(epoch_frame)} "
                      f"downsample_seed={args.seed + epoch - 1}", flush=True)
            model.train()
            loss_sum, num_batches = 0.0, 0
            max_grad_norm = 0.0
            for batch in loader:
                heads, relations, tails = [batch[k].to(device) for k in ["head_id", "relation_id", "tail_id"]]
                optimizer.zero_grad(set_to_none=True)
                pos_scores = model(heads, relations, tails)
                nh, nr, nt = sample_negatives(helpers, args.dataset, heads, relations, tails, model,
                                             args.negatives, filter_map, relation_probs, generator, training_pool)
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
            validate_now = (epoch % args.validate_every == 0) or epoch == args.epochs
            if not validate_now:
                event = dict(epoch=epoch, time=utc(), train_loss=loss_sum / num_batches,
                             train_seconds=train_seconds, epoch_seconds=time.monotonic()-epoch_start,
                             **{selection_key: None}, max_gradient_norm=max_grad_norm,
                             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                with (out / "epochs.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                state.update(completed_epochs=epoch, last_epoch=event, elapsed_seconds=time.monotonic()-start)
                save_json(out / "status.json", state)
                print(f"epoch={epoch} loss={event['train_loss']:.6f} (no {args.selection_split} evaluation this epoch) "
                      f"train={train_seconds:.1f}s", flush=True)
                continue
            val = evaluate_full(model, selection_triples, filter_map, device,
                                query_batch=args.query_batch, candidate_chunk=args.candidate_chunk,
                                max_queries=args.eval_limit, bidirectional=profile.get("bidirectional_eval", True),
                                relation_to_valid_tails=selection_pool)
            if args.kind != "smoke" and not val["complete"]:
                raise RuntimeError("Pilot/formal selection evaluation must be complete")
            val.update(selection_metadata)
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
                    if prefix.endswith("log_max_rel_error_vs_eig") and value > 1e-2:
                        raise FloatingPointError(f"v2 principal log inaccurate on sample {prefix}: {value}")
                    if prefix.endswith("log_fallback_fraction") and value > 0.05:
                        raise FloatingPointError(f"v2 principal log fell back on too many sampled pairs {prefix}: {value}")
            check_health(diagnostics)
            event = dict(epoch=epoch, time=utc(), train_loss=loss_sum / num_batches,
                         train_seconds=train_seconds, epoch_seconds=time.monotonic()-epoch_start,
                         **{selection_key: {k: v for k, v in val.items() if not k.startswith("ranks_")}},
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
                stale += args.validate_every
            state.update(completed_epochs=epoch, best_epoch=best_epoch, best_selection_mrr=best,
                         **{best_metric_key: best},
                         last_epoch=event, elapsed_seconds=time.monotonic()-start)
            save_json(out / "status.json", state)
            metric_label = "val_mrr" if args.selection_split == "validation" else "test_selection_mrr"
            print(f"epoch={epoch} loss={event['train_loss']:.6f} {metric_label}={val['mrr']:.8f} "
                  f"best={best:.8f}@{best_epoch} train={train_seconds:.1f}s total={event['epoch_seconds']:.1f}s "
                  f"peak={event['peak_allocated_gib']:.2f}GiB", flush=True)
            if args.patience is not None and stale >= args.patience:
                break
        if args.kind == "formal":
            saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            final_test = evaluate_full(model, triples[2], filter_map, device,
                                       query_batch=args.query_batch, candidate_chunk=args.candidate_chunk,
                                       bidirectional=profile.get("bidirectional_eval", True),
                                       relation_to_valid_tails=test_pool)
            final_test.update(selection_metadata)
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
