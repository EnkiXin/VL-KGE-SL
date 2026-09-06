"""Chunked evaluation matching the author's WN9 and WikiArt release protocols.

This module deliberately accepts the *combined-direction* author filter map.
It does not silently replace that map with a corrected directional protocol.
All candidates are scored, including the target, in the same chunked pass;
the target score is never recomputed using a differently shaped operation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Mapping

import torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _summarize(ranks: list[int]) -> dict:
    count = len(ranks)
    return {
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "hits": {str(k): sum(rank <= k for rank in ranks) / count for k in (1, 3, 10)},
        "count": count,
    }


def _relation_tail_ranks(model, batch, batch_cpu, pools, filter_map, candidate_chunk):
    """Batch by relation while preserving the author's per-query candidate set.

    Score the sorted pool plus the batch's targets; each row removes known tails
    and any extra targets belonging only to other rows, then restores its own target.
    This is the same rank set as (pool - correct_tails) | {target} in the release.
    """
    ranks = [None] * len(batch_cpu)
    for relation in batch_cpu[:, 1].unique().tolist():
        row_ids = (batch_cpu[:, 1] == relation).nonzero(as_tuple=False).flatten().tolist()
        subset = batch[row_ids]
        pool = {int(value) for value in pools.get(relation, ())}
        candidates = sorted(pool | set(batch_cpu[row_ids, 2].tolist()))
        if candidates[0] < 0 or candidates[-1] >= model.num_entities:
            raise ValueError("relation candidate pool contains an out-of-range entity ID")
        candidate_ids = torch.tensor(candidates, dtype=torch.long, device=batch.device)
        size = len(row_ids)
        scores = None
        for start in range(0, len(candidates), candidate_chunk):
            chunk = candidate_ids[start:start + candidate_chunk]
            width = len(chunk)
            values = model(subset[:, 0, None].expand(size, width).reshape(-1),
                           subset[:, 1, None].expand(size, width).reshape(-1),
                           chunk[None].expand(size, width).reshape(-1))
            if values.ndim != 1 or values.numel() != size * width:
                raise ValueError("model must return one flat score per input triple")
            if values.device != batch.device or not values.is_floating_point():
                raise ValueError("model scores must be floating tensors on the evaluation device")
            if not torch.isfinite(values).all():
                raise FloatingPointError("nonfinite unfiltered relation-pool scores")
            if scores is None:
                scores = torch.empty((size, len(candidates)), dtype=values.dtype, device=batch.device)
            elif values.dtype != scores.dtype:
                raise ValueError("model score dtype changed between candidate chunks")
            scores[:, start:start + width] = values.reshape(size, width)
        columns = {entity: column for column, entity in enumerate(candidates)}
        target_columns = torch.tensor([columns[int(t)] for t in batch_cpu[row_ids, 2]], device=batch.device)
        target_scores = scores.gather(1, target_columns[:, None]).clone()
        extra_targets = set(candidates) - pool
        for row, source_row in enumerate(row_ids):
            head, _, target = batch_cpu[source_row].tolist()
            known = filter_map.get((head, relation), set())
            excluded = (set(known) | extra_targets) - {target}
            column_ids = [columns[entity] for entity in excluded if entity in columns]
            if column_ids:
                scores[row, column_ids] = float("-inf")
        better = (scores > target_scores).sum(dim=1)
        ties = ((scores == target_scores) & (candidate_ids[None] < subset[:, 2, None])).sum(dim=1)
        for source_row, rank in zip(row_ids, (1 + better + ties).cpu().tolist()):
            ranks[source_row] = rank
    return ranks


def evaluate_full(
    model: torch.nn.Module,
    triples: torch.Tensor,
    filter_map: Mapping[tuple[int, int], set[int]],
    device: torch.device | str,
    query_batch: int = 16,
    candidate_chunk: int = 512,
    max_queries: int | None = None,
    bidirectional: bool = True,
    relation_to_valid_tails: Mapping[int, list[int]] | None = None,
) -> dict:
    """Evaluate the author candidate/direction policy, with stable tie ranks.

    ``triples`` is a CPU integer tensor with rows ``(head, relation, tail)``.
    ``max_queries`` is a diagnostic cap on *input triples*: both directions
    are evaluated only when ``bidirectional=True`` (the legacy WN9 default).
    WikiArt passes its split-specific relation candidate pools and False.
    ``complete`` in the returned record distinguishes capped evaluation.

    The model must expose ``num_entities`` and return one floating score per
    triple from ``model(head, relation, tail)``. Larger scores are better.
    Feature/representation caching is owned by the model, not this evaluator.
    Global CPU and the selected CUDA RNG states and the model's train/eval mode
    are restored on exit, including on errors.
    """
    query_batch = _positive_integer(query_batch, "query_batch")
    candidate_chunk = _positive_integer(candidate_chunk, "candidate_chunk")
    if not isinstance(bidirectional, bool):
        raise ValueError("bidirectional must be boolean")
    if max_queries is not None:
        max_queries = _positive_integer(max_queries, "max_queries")
    if not isinstance(triples, torch.Tensor) or triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must be a CPU integer tensor of shape [N, 3]")
    if triples.device.type != "cpu" or triples.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("triples must be a CPU integer tensor of shape [N, 3]")
    total_triples = int(triples.shape[0])
    if total_triples == 0:
        raise ValueError("cannot evaluate an empty triple set")
    num_entities = _positive_integer(model.num_entities, "model.num_entities")
    triples = triples.to(dtype=torch.long)
    if (triples < 0).any() or (triples[:, (0, 2)] >= num_entities).any():
        raise ValueError("triples contain an invalid entity or negative relation ID")
    if hasattr(model, "num_relations") and (triples[:, 1] >= model.num_relations).any():
        raise ValueError("triples contain an invalid relation ID")
    if max_queries is not None:
        triples = triples[:max_queries]

    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [torch.cuda.current_device() if device.index is None else device.index]
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    was_training = model.training
    ranks_head: list[int] = []
    ranks_tail: list[int] = []
    # Preserve each direction's original order for author-identical summation.
    relation_head: dict[int, list[int]] = defaultdict(list)
    relation_tail: dict[int, list[int]] = defaultdict(list)

    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            all_ids = torch.arange(num_entities, device=device)
            for offset in range(0, len(triples), query_batch):
                batch_cpu = triples[offset:offset + query_batch]
                batch = batch_cpu.to(device=device)
                size = len(batch)
                for direction in (("tail", "head") if bidirectional else ("tail",)):
                    if direction == "tail" and relation_to_valid_tails is not None:
                        ranks = _relation_tail_ranks(model, batch, batch_cpu, relation_to_valid_tails,
                                                     filter_map, candidate_chunk)
                        ranks_tail.extend(ranks)
                        for relation, rank in zip(batch_cpu[:, 1].tolist(), ranks):
                            relation_tail[relation].append(rank)
                        continue
                    scores = None
                    for left in range(0, num_entities, candidate_chunk):
                        candidates = all_ids[left:left + candidate_chunk]
                        width = len(candidates)
                        candidate_ids = candidates.unsqueeze(0).expand(size, width)
                        relation_ids = batch[:, 1, None].expand(size, width)
                        if direction == "tail":
                            head_ids = batch[:, 0, None].expand(size, width)
                            tail_ids = candidate_ids
                        else:
                            head_ids = candidate_ids
                            tail_ids = batch[:, 2, None].expand(size, width)
                        values = model(
                            head_ids.reshape(-1), relation_ids.reshape(-1), tail_ids.reshape(-1)
                        )
                        if values.ndim != 1 or values.numel() != size * width:
                            raise ValueError("model must return one flat score per input triple")
                        if values.device != batch.device or not values.is_floating_point():
                            raise ValueError("model scores must be floating tensors on the evaluation device")
                        if not torch.isfinite(values).all():
                            raise FloatingPointError(
                                f"nonfinite unfiltered {direction} scores at triple offset "
                                f"{offset}, candidate offset {left}"
                            )
                        if scores is None:
                            scores = torch.empty((size, num_entities), dtype=values.dtype, device=device)
                        elif values.dtype != scores.dtype:
                            raise ValueError("model score dtype changed between candidate chunks")
                        scores[:, left:left + width] = values.reshape(size, width)

                    target_column = 2 if direction == "tail" else 0
                    filter_column = 0 if direction == "tail" else 2
                    targets = batch[:, target_column]
                    target_scores = scores.gather(1, targets[:, None]).clone()
                    for row, triple in enumerate(batch_cpu.tolist()):
                        target = triple[target_column]
                        filtered = filter_map.get((triple[filter_column], triple[1]), set())
                        excluded = [int(entity) for entity in filtered if int(entity) != target]
                        if excluded:
                            if min(excluded) < 0 or max(excluded) >= num_entities:
                                raise ValueError("filter map contains an out-of-range entity ID")
                            scores[row, excluded] = float("-inf")
                    # Stable descending argsort on ascending candidate IDs has
                    # exactly this rank, without allocating sorted indices.
                    better = (scores > target_scores).sum(dim=1)
                    earlier_ties = ((scores == target_scores) & (all_ids[None] < targets[:, None])).sum(dim=1)
                    ranks = (1 + better + earlier_ties).cpu().tolist()
                    output = ranks_tail if direction == "tail" else ranks_head
                    per_relation = relation_tail if direction == "tail" else relation_head
                    output.extend(ranks)
                    for relation, rank in zip(batch_cpu[:, 1].tolist(), ranks):
                        per_relation[relation].append(rank)
    finally:
        model.train(was_training)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = _summarize(ranks_tail + ranks_head)
    # Author code sums the two directional reciprocal-rank sums separately.
    summary["mrr"] = (
        sum(1.0 / rank for rank in ranks_tail) + sum(1.0 / rank for rank in ranks_head)
    ) / (len(ranks_tail) + len(ranks_head))
    elapsed = time.perf_counter() - start
    return {
        "mrr": summary["mrr"],
        "hits": summary["hits"],
        "ranks_head": ranks_head,
        "ranks_tail": ranks_tail,
        "per_relation": {
            str(relation): _summarize(relation_tail[relation] + relation_head[relation])
            for relation in sorted(relation_tail)
        },
        "elapsed_seconds": elapsed,
        "elapsed": elapsed,
        "evaluated_triples": len(triples),
        "directional_queries": (2 if bidirectional else 1) * len(triples),
        "total_triples": total_triples,
        "num_entities": num_entities,
        "complete": len(triples) == total_triples,
        "protocol": "author_release_protocol",
        "bidirectional": bidirectional,
        "candidate_policy": "per_relation_tail_pool" if relation_to_valid_tails is not None else "all_entities",
        "tie_policy": "stable_descending_score_then_ascending_entity_id",
    }
