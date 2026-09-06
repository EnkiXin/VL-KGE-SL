"""CPU-only author-input/protocol preflight; does not train or use a GPU."""
import argparse
import json
from pathlib import Path
import resource
import time

from run_geometry import load_inputs, setup_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("wn9_img", "wikiart_mkg_v1", "wikiart_mkg_v2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    start = time.monotonic()
    repo, _ = setup_paths(args.root.resolve())
    from geometry.datasets import author_profile, epoch_training_frame, check_v1_negative_capacity
    profile = author_profile(repo, args.dataset)
    data, frames, triples, model_args, _ = load_inputs(args.root, repo, dataset=args.dataset, profile=profile)
    entities, relations = data.get_entities_and_relations()
    filters = data.compute_filter_map()
    if args.dataset == "wikiart_mkg_v1":
        check_v1_negative_capacity(frames[0], entities, relations, filters,
                                   data.relation_to_valid_tails_train, 1)
    effective = epoch_training_frame(frames[0], profile, 42, 0)
    pools = {}
    if profile.get("use_per_relation_candidates"):
        for label, name in (("train", "relation_to_valid_tails_train"),
                            ("val", "relation_to_valid_tails_eval_val"),
                            ("test", "relation_to_valid_tails_eval_test")):
            pools[label] = {str(r): len(ids) for r, ids in getattr(data, name).items()}
    payload = dict(status="ready", dataset=args.dataset, num_entities=len(entities),
                   num_relations=len(relations), split_sizes=[len(x) for x in triples],
                   first_epoch_effective_train_triples=len(effective), relation_candidate_counts=pools,
                   inductive=model_args["inductive"], modality_asymmetry=model_args["modality_asymmetry"],
                   bidirectional=profile.get("bidirectional_eval", True),
                   negatives=profile["num_neg_samples"], protocol_notes=profile["protocol_notes"],
                   elapsed_seconds=time.monotonic() - start,
                   peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
