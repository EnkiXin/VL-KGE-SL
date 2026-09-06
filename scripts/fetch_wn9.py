"""Fetch pinned author triples/CLIP inputs; legacy WN9 invocation is preserved.

Example: --repo upstream/vl-kge --datasets wikiart_mkg_v1 wikiart_mkg_v2
         --manifest-dir artifacts
No raw images, encoder weights, relation features or unrelated LFS files are fetched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"
DATASETS = ("wn9_img", "wikiart_mkg_v1", "wikiart_mkg_v2")
DATASET_FILES = {
    dataset: (
        f"vlkge/data/{dataset}/{dataset}_triples.csv",
        f"vlkge/data/{dataset}/features/{dataset}_vf_clip.pkl",
        f"vlkge/data/{dataset}/features/{dataset}_tf_clip.pkl",
    ) for dataset in DATASETS
}
# Preserve the historical module constant for callers of this WN9 utility.
FILES = list(DATASET_FILES["wn9_img"])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["wn9_img"])
    manifests = parser.add_mutually_exclusive_group(required=True)
    manifests.add_argument("--manifest", type=Path, help="Legacy/custom path for one dataset only")
    manifests.add_argument("--manifest-dir", type=Path, help="Write separate dataset-named manifests")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    if args.manifest is not None and len(args.datasets) != 1:
        parser.error("--manifest requires one dataset; use --manifest-dir for multiple datasets")
    return args


def manifest_name(dataset):
    return "wn9-inputs.json" if dataset == "wn9_img" else f"{dataset}-inputs.json"


def records_from_pointers(repo, dataset):
    records, pointers = [], {}
    for relative in DATASET_FILES[dataset]:
        pointer = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=repo)
        fields = dict(line.split(" ", 1) for line in pointer.decode().strip().splitlines())
        oid = fields.get("oid", "")
        expected_hash = oid[7:]
        if (fields.get("version") != "https://git-lfs.github.com/spec/v1"
                or not oid.startswith("sha256:") or len(expected_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in expected_hash)):
            raise RuntimeError(f"Pinned input is not a valid SHA256 LFS pointer: {relative}")
        expected_size = int(fields["size"])
        if expected_size < 1:
            raise RuntimeError(f"Invalid LFS size: {relative}")
        records.append({"path": relative, "bytes": expected_size, "sha256": expected_hash,
                        "url": f"https://media.githubusercontent.com/media/thefth/vl-kge/{COMMIT}/{relative}"})
        pointers[relative] = pointer
    return records, pointers


def verify_manifest_target(path, dataset, records):
    """Never overwrite a manifest belonging to another input set or experiment."""
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink manifest: {path}")
    if path.exists():
        current = json.loads(path.read_text())
        if (current.get("upstream_commit") != COMMIT or current.get("files") != records
                or current.get("dataset", dataset) != dataset):
            raise RuntimeError(f"Existing manifest describes different inputs; left unchanged: {path}")


def fetch_dataset(repo, dataset, manifest):
    records, pointers = records_from_pointers(repo, dataset)
    verify_manifest_target(manifest, dataset, records)
    for record in records:
        relative, expected_size, expected_hash = record["path"], record["bytes"], record["sha256"]
        path = repo / relative
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink input: {path}")
        if not (path.is_file() and path.stat().st_size == expected_size and digest(path) == expected_hash):
            # Only the exact, small pinned pointer may be replaced, not a file
            # whose first bytes happen to resemble an LFS pointer.
            pointer = pointers[relative]
            if path.exists() and (not path.is_file() or path.stat().st_size != len(pointer)
                                  or path.read_bytes() != pointer):
                raise RuntimeError(f"Existing file is neither matching data nor its LFS pointer: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".download-", dir=path.parent)
            partial = Path(temporary)
            print(f"Downloading {relative}: {expected_size / 1e6:.1f} MB", flush=True)
            start = last_report = time.monotonic()
            count = 0
            h = hashlib.sha256()
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    with urllib.request.urlopen(record["url"], timeout=120) as response:
                        while chunk := response.read(4 * 1024 * 1024):
                            stream.write(chunk)
                            h.update(chunk)
                            count += len(chunk)
                            if time.monotonic() - last_report >= 10:
                                print(f"  {count / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
                                last_report = time.monotonic()
                if count != expected_size or h.hexdigest() != expected_hash:
                    raise RuntimeError(f"Size or SHA256 mismatch: {relative}")
            except Exception:
                print(f"Download incomplete; original left unchanged, partial retained: {partial}", flush=True)
                raise
            # Recheck before replacement in case another task touched the input.
            if path.is_symlink() or (path.exists() and
                    (not path.is_file() or path.stat().st_size != len(pointer) or path.read_bytes() != pointer)):
                raise RuntimeError(f"Input changed while downloading; verified partial retained: {partial}")
            os.replace(partial, path)
            print(f"Verified {relative} ({time.monotonic() - start:.1f}s)", flush=True)
        else:
            print(f"Already verified: {relative}", flush=True)
    verify_manifest_target(manifest, dataset, records)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        payload = {"upstream_commit": COMMIT, "dataset": dataset, "files": records}
        # Exclusive creation cannot overwrite an older WN9 or other experiment manifest.
        with manifest.open("x") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    print(f"Dataset ready: {dataset}; manifest: {manifest}", flush=True)
    return {"upstream_commit": COMMIT, "dataset": dataset, "files": records}


def main(argv=None):
    args = parse_args(argv)
    repo = args.repo.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {actual}")
    for dataset in args.datasets:
        manifest = args.manifest if args.manifest is not None else args.manifest_dir / manifest_name(dataset)
        fetch_dataset(repo, dataset, manifest)


if __name__ == "__main__":
    main()
