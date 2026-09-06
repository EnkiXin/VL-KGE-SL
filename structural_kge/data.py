"""Pinned structural KGE data and train-only negative sampling.

The released dictionaries define the IDs, including entities absent from train.
Held-out triples are used ONLY for conventional filtered-ranking masks.  They
never influence the training-negative filter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping
from urllib.request import Request, urlopen

import numpy as np


SOURCE_REPOSITORY = "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding"
SOURCE_COMMIT = "2e440e0f9c687314d5ff67ead68ce985dc446e3a"
SOURCE_VERIFICATION = (
    "https://api.github.com/repos/DeepGraphLearning/KnowledgeGraphEmbedding/"
    "git/trees/" + SOURCE_COMMIT + "?recursive=1"
)
# Git blob IDs independently read from the official author's pinned Git tree.
SOURCES = {
    "WN18RR": {
        "directory": "wn18rr",
        "counts": [40943, 11, 86835, 3034, 3134],
        "blobs": {
            "entities.dict": "e0393449fc513397261edc873dcf3d82114dd461",
            "relations.dict": "40deef1026ad3d8bf50a95e2fcce9a53f6c0ce05",
            "train.txt": "7c7d11ee21594d729ef90f20ecef3adf7573109d",
            "valid.txt": "ca0bf43869c6b537b667b4f284c6074078bd0013",
            "test.txt": "fbd87d236e7183fc60c9320895a714d3f4716da0",
        },
    },
    "FB15k-237": {
        "directory": "FB15k-237",
        "counts": [14541, 237, 272115, 17535, 20466],
        "blobs": {
            "entities.dict": "7c64b7eea5f22538d0cf815789d1b22e1a06feee",
            "relations.dict": "3ab5a19639f76f598262bbe93ee370be0bb01759",
            "train.txt": "64905fe4545b7f2f75454d0b6fd52dfdcab90ed4",
            "valid.txt": "1eb92bf5af311b25748db1b5d85db39cb34ba2e8",
            "test.txt": "01f6b3bf2fdc68b4913e816ac859921b740d09ee",
        },
    },
}


def canonical_name(name: str) -> str:
    for candidate in SOURCES:
        if candidate.lower() == name.lower():
            return candidate
    raise ValueError(f"Unsupported dataset {name!r}; choose {tuple(SOURCES)}")


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(directory: Path, dataset: str) -> dict:
    """Check pinned bytes, not merely a mutable local checksum manifest."""
    dataset = canonical_name(dataset)
    manifest = json.loads((directory / "provenance.json").read_text())
    if manifest.get("dataset") != dataset or manifest.get("commit") != SOURCE_COMMIT:
        raise ValueError("Dataset provenance does not match the pinned release")
    for filename, expected_blob in SOURCES[dataset]["blobs"].items():
        content = (directory / filename).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if git_blob_sha1(content) != expected_blob:
            raise ValueError(f"Pinned source mismatch: {filename}")
        if manifest["files"][filename]["sha256"] != digest:
            raise ValueError(f"SHA256 provenance mismatch: {filename}")
    return manifest


def fetch_dataset(data_root: str | Path, dataset: str, timeout: float = 120.0) -> Path:
    """Download one pinned public release atomically; never overwrite a dataset."""
    dataset = canonical_name(dataset)
    data_root = Path(data_root).expanduser().resolve()
    destination = data_root / dataset
    if destination.exists():
        verify_dataset(destination, dataset)
        return destination
    data_root.mkdir(parents=True, exist_ok=True)
    source = SOURCES[dataset]
    manifest = {
        "dataset": dataset,
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "verification_url": SOURCE_VERIFICATION,
        "id_policy": "unaltered released entities.dict and relations.dict",
        "files": {},
    }
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-download-", dir=data_root) as temporary:
        staged = Path(temporary) / dataset
        staged.mkdir()
        for filename, expected_blob in source["blobs"].items():
            url = ("https://raw.githubusercontent.com/DeepGraphLearning/"
                   f"KnowledgeGraphEmbedding/{SOURCE_COMMIT}/data/"
                   f"{source['directory']}/{filename}")
            request = Request(url, headers={"User-Agent": "structural-kge-reproduction/1"})
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
            if git_blob_sha1(content) != expected_blob:
                raise ValueError(f"Downloaded content does not match pinned blob: {filename}")
            (staged / filename).write_bytes(content)
            manifest["files"][filename] = {
                "url": url, "bytes": len(content), "git_blob_sha1": expected_blob,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        (staged / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_dataset(staged.parent, dataset, verify=True)
        if destination.exists():
            raise FileExistsError(f"Dataset appeared during download: {destination}")
        staged.rename(destination)
    return destination


def read_dictionary(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    ids = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"Expected ID-tab-label at {path}:{number}")
        identifier, label = int(fields[0]), fields[1]
        if identifier < 0 or label in labels or identifier in ids:
            raise ValueError(f"Duplicate or negative dictionary entry at {path}:{number}")
        labels[label] = identifier
        ids.add(identifier)
    if ids != set(range(len(ids))):
        raise ValueError(f"Dictionary IDs must be contiguous from zero: {path}")
    return labels


def read_triples(path: Path, entities: Mapping[str, int], relations: Mapping[str, int]) -> np.ndarray:
    triples = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"Expected head-tab-relation-tab-tail at {path}:{number}")
        h, r, t = fields
        try:
            triples.append((entities[h], relations[r], entities[t]))
        except KeyError as error:
            raise ValueError(f"Triple references an undeclared ID at {path}:{number}") from error
    return np.asarray(triples, dtype=np.int64).reshape(-1, 3)


@dataclass
class FilterIndex:
    tails: dict[tuple[int, int], np.ndarray]
    heads: dict[tuple[int, int], np.ndarray]

    @classmethod
    def from_triples(cls, triples: np.ndarray) -> "FilterIndex":
        tails: dict[tuple[int, int], set[int]] = {}
        heads: dict[tuple[int, int], set[int]] = {}
        for h, r, t in triples:
            tails.setdefault((int(h), int(r)), set()).add(int(t))
            heads.setdefault((int(r), int(t)), set()).add(int(h))
        return cls(
            {key: np.asarray(sorted(values), dtype=np.int64) for key, values in tails.items()},
            {key: np.asarray(sorted(values), dtype=np.int64) for key, values in heads.items()},
        )


@dataclass
class KGDataset:
    name: str
    entities: dict[str, int]
    relations: dict[str, int]
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    provenance: dict

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def training_filter(self) -> FilterIndex:
        return FilterIndex.from_triples(self.train)

    def ranking_filter(self) -> FilterIndex:
        return FilterIndex.from_triples(np.concatenate((self.train, self.valid, self.test)))


def load_dataset(data_root: str | Path, dataset: str, verify: bool = True) -> KGDataset:
    name = canonical_name(dataset)
    directory = Path(data_root).expanduser().resolve() / name
    provenance = verify_dataset(directory, name) if verify else {"verified": False}
    entities = read_dictionary(directory / "entities.dict")
    relations = read_dictionary(directory / "relations.dict")
    splits = [read_triples(directory / f"{split}.txt", entities, relations)
              for split in ("train", "valid", "test")]
    observed = [len(entities), len(relations), *(len(split) for split in splits)]
    if verify and observed != SOURCES[name]["counts"]:
        raise ValueError(f"Unexpected release counts: {observed}")
    return KGDataset(name, entities, relations, *splits, provenance)


class NegativeSampler:
    """Exact uniform sampling WITH replacement from train-only complements.

    Mapping a uniform rank in the complement to an entity ID via searchsorted
    avoids unbounded rejection loops and materializing O(queries * entities)
    pools.  A query with no admissible entity raises immediately.
    """

    def __init__(self, num_entities: int, training_filter: FilterIndex, seed: int):
        self.num_entities = num_entities
        self.rng = np.random.default_rng(seed)
        self._tails = {key: (values - np.arange(len(values)), len(values))
                       for key, values in training_filter.tails.items()}
        self._heads = {key: (values - np.arange(len(values)), len(values))
                       for key, values in training_filter.heads.items()}

    def sample(self, positives: np.ndarray, negatives: int, direction: str) -> np.ndarray:
        if negatives <= 0 or direction not in ("head", "tail"):
            raise ValueError("Require positive negatives and head/tail direction")
        result = np.empty((len(positives), negatives), dtype=np.int64)
        lookup = self._heads if direction == "head" else self._tails
        for row, (h, r, t) in enumerate(positives):
            key = (int(r), int(t)) if direction == "head" else (int(h), int(r))
            blocked, count = lookup.get(key, (np.empty(0, dtype=np.int64), 0))
            available = self.num_entities - count
            if available <= 0:
                raise ValueError(f"No admissible train-negative entity for {direction} query {key}")
            draws = self.rng.integers(available, size=negatives, dtype=np.int64)
            result[row] = draws + np.searchsorted(blocked, draws, side="right")
        return result


def validation_subset(valid: np.ndarray, limit: int, seed: int = 260906) -> tuple[np.ndarray, np.ndarray]:
    if limit < 0:
        raise ValueError("Validation limit must be nonnegative; zero means full split")
    indices = (np.arange(len(valid), dtype=np.int64) if limit == 0 or limit >= len(valid)
               else np.sort(np.random.default_rng(seed).choice(len(valid), size=limit, replace=False)))
    return valid[indices].copy(), indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify canonical, commit-pinned KGE datasets")
    parser.add_argument("--dataset", required=True, choices=list(SOURCES))
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    destination = fetch_dataset(args.data_root, args.dataset)
    print(json.dumps({"dataset": args.dataset, "directory": str(destination),
                      "commit": SOURCE_COMMIT, "verified": True}))


if __name__ == "__main__":
    main()
