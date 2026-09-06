"""Dataset-specific author-release policies, without editing the author checkout."""

import hashlib
import json
from pathlib import Path


DATASETS = ("wn9_img", "wikiart_mkg_v1", "wikiart_mkg_v2")
AUTHOR_COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"
V2_INVERSE_NOTE = (
    "The released WikiArt-v2 YAML parses add_inverse_relations as "
    "{'isCreatedByArtist:hasCreatedArtwork': None}. The author's loader therefore "
    "adds no inverse triples. This experiment preserves that released behavior, "
    "not the documentation's intended isCreatedByArtist -> hasCreatedArtwork mapping."
)


def author_profile(repo, dataset):
    """Read the locked DistMult/CLIP profile, retaining its literal release semantics."""
    import yaml

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    path = Path(repo) / "vlkge" / "configs" / dataset / "distmult_clip.yaml"
    profile = yaml.safe_load(path.read_text())
    if profile.get("dataset") != dataset or profile.get("embedding_dim") != 768:
        raise ValueError("Unexpected author dataset or embedding width")
    profile = dict(profile)
    profile["config_path"] = str(path)
    profile["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    profile["protocol_notes"] = [
        "Author all-split, combined-direction filter is retained for sampling and evaluation."
    ]
    if dataset == "wikiart_mkg_v2":
        malformed = {"isCreatedByArtist:hasCreatedArtwork": None}
        if profile.get("add_inverse_relations") != malformed:
            raise ValueError("WikiArt-v2 inverse-relation release configuration changed")
        # Omitting this no-op gives exactly the published loader's resulting triples.
        profile["effective_inverse_relations"] = None
        profile["protocol_notes"].append(V2_INVERSE_NOTE)
    else:
        profile["effective_inverse_relations"] = profile.get("add_inverse_relations")
    return profile


def loader_options(profile):
    return dict(
        exclude_relations=profile.get("exclude_relations"),
        exclude_relations_eval=profile.get("exclude_relations_eval"),
        add_inverse_relations=profile.get("effective_inverse_relations"),
        use_per_relation_candidates=profile.get("use_per_relation_candidates", False),
        artist2artist_relations=profile.get("artist2artist_relations"),
        bidirectional_eval=profile.get("bidirectional_eval", True),
    )


def verified_manifest(root, repo, dataset):
    artifacts = Path(root) / "artifacts"
    names = (["wn9-inputs.json", "wn9-inputs-local.json", "wn9_img-inputs.json"]
             if dataset == "wn9_img" else [f"{dataset}-inputs.json"])
    path = next((artifacts / name for name in names if (artifacts / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing verified input manifest for {dataset}: {names}")
    manifest = json.loads(path.read_text())
    if manifest.get("upstream_commit") != AUTHOR_COMMIT:
        raise ValueError("Input manifest upstream_commit does not match the locked author release")
    # The original WN9 manifest predates the explicit dataset field.
    manifest_dataset = manifest.get("dataset", "wn9_img" if dataset == "wn9_img" else None)
    if manifest_dataset != dataset:
        raise ValueError("Input manifest dataset does not match the requested dataset")
    records = {record["path"]: record for record in manifest["files"]}
    prefix = f"vlkge/data/{dataset}/"
    required = [f"{prefix}{dataset}_triples.csv",
                f"{prefix}features/{dataset}_vf_clip.pkl",
                f"{prefix}features/{dataset}_tf_clip.pkl"]
    for relative in required:
        if relative not in records:
            raise ValueError(f"Input manifest does not cover {relative}")
        record = records[relative]
        file_path = Path(repo) / relative
        if file_path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Input size mismatch: {file_path}")
        sha = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != record["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {file_path}")
    return manifest


def mark_training_entities(model, train_frame, entity_to_id):
    """Match the author's seen set, formed before optional per-epoch downsampling."""
    import torch

    if not model.inductive:
        return None
    names = set(train_frame["head"]) | set(train_frame["tail"])
    ids = torch.tensor(sorted(entity_to_id[name] for name in names), dtype=torch.long)
    model.mark_seen_entities(ids)
    return len(ids)


def epoch_training_frame(train_frame, profile, seed, epoch_index):
    """Use the author's zero-based epoch seed and pandas relation downsampling."""
    import pandas as pd

    relation = profile.get("downsample_relation")
    if relation is None:
        return train_frame
    related = train_frame[train_frame["relation"] == relation]
    others = train_frame[train_frame["relation"] != relation]
    sampled = (related.sample(frac=profile.get("downsample_fraction", 1.0),
                              random_state=seed + epoch_index)
               if len(related) else related)
    return pd.concat([others, sampled], ignore_index=True)


def sample_negatives(helpers, dataset, heads, relations, tails, model, count,
                     filter_map, relation_probs, generator, training_pool=None):
    if dataset == "wn9_img":
        return helpers.negative_sampling_uniform(
            heads, relations, tails, model.num_entities, count,
            filter_map, relation_probs, generator, use_bernoulli=False)
    if training_pool is None:
        raise ValueError("WikiArt requires the author's training-only relation candidate pools")
    sampler = (helpers.negative_sampling_per_relation if dataset == "wikiart_mkg_v1"
               else helpers.negative_sampling_per_relation_fast)
    result = sampler(heads, relations, tails, count, filter_map, training_pool, generator)
    if any(value.numel() != heads.numel() * count for value in result):
        raise RuntimeError(
            "The author WikiArt sampler returned fewer negatives than requested. "
            "Refusing to reshape misaligned examples or silently change the sampler/loss."
        )
    return result


def check_v1_negative_capacity(train_frame, entity_to_id, relation_to_id, filter_map,
                               training_pool, count):
    """Fail before training if the author's no-replacement sampler would be ragged."""
    pool_sets = {relation: set(values) for relation, values in training_pool.items()}
    for head, relation in train_frame[["head", "relation"]].drop_duplicates().itertuples(index=False, name=None):
        h, r = entity_to_id[head], relation_to_id[relation]
        pool = pool_sets.get(r, set())
        available_count = len(pool) - len(pool & filter_map.get((h, r), set()))
        if available_count < count:
            raise ValueError(
                f"WikiArt-v1 author sampler has {available_count} valid negatives for "
                f"head={h}, relation={r}, but {count} were requested; no sampling fallback is applied."
            )
