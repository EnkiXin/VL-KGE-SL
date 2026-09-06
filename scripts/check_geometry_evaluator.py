"""Read-only numerical audit against a completed author checkpoint."""
import argparse
import json
from pathlib import Path
import sys

from run_geometry import load_inputs, setup_paths, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--author-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root.resolve()
    repo, _ = setup_paths(root)
    import torch
    from vlkge import helpers
    from vlkge.models import DistMult
    from vlkge.dataloader import KGDataset
    from torch.utils.data import DataLoader
    from geometry.evaluation import evaluate_full
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    data, frames, triples, model_args, _ = load_inputs(root, repo)
    model = DistMult(**model_args, device=device).to(device)
    checkpoint = torch.load(args.author_run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    expected = json.loads((args.author_run / "result.json").read_text())["final_test"]
    result = evaluate_full(model, triples[2], data.compute_filter_map(), device)
    ids, relations = data.get_entities_and_relations()
    loader = DataLoader(KGDataset(frames[2], ids, relations), batch_size=512, shuffle=False)
    original = helpers.evaluate_kge(model, loader, data.compute_filter_map(), device=device, bidirectional=True)
    actual = {"mrr": result["mrr"], "hits": result["hits"]}
    observed = {"mrr": original[0], "hits": {str(k): v for k, v in original[1].items()}}
    match = abs(actual["mrr"] - expected["mrr"]) < 1e-12
    match &= abs(actual["mrr"] - observed["mrr"]) < 1e-12
    match &= all(actual["hits"][k] == expected["hits"][k] == observed["hits"][k] for k in actual["hits"])
    report = dict(passed=bool(match), source=str(args.author_run),
                  expected={"mrr": expected["mrr"], "hits": expected["hits"]},
                  author_evaluator=observed, geometry_evaluator=actual,
                  full_test_triples=result["evaluated_triples"], protocol=result["protocol"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps(report), flush=True)
    if not match:
        raise RuntimeError("New evaluation failed author-checkpoint equivalence")


if __name__ == "__main__":
    main()
