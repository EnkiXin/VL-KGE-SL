"""Fetch only the three author WN9 CLIP inputs and verify Git LFS SHA256s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"
FILES = [
    "vlkge/data/wn9_img/wn9_img_triples.csv",
    "vlkge/data/wn9_img/features/wn9_img_vf_clip.pkl",
    "vlkge/data/wn9_img/features/wn9_img_tf_clip.pkl",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {actual}")
    records = []
    for relative in FILES:
        pointer = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=repo, text=True)
        fields = dict(line.split(" ", 1) for line in pointer.strip().splitlines())
        expected_hash = fields["oid"].removeprefix("sha256:")
        expected_size = int(fields["size"])
        path = repo / relative
        url = f"https://media.githubusercontent.com/media/thefth/vl-kge/{COMMIT}/{relative}"
        if not (path.exists() and path.stat().st_size == expected_size and digest(path) == expected_hash):
            # Never overwrite an unrelated existing file.
            if path.exists() and path.read_bytes()[:100] != pointer.encode()[:100]:
                raise RuntimeError(f"Existing file is neither matching data nor its LFS pointer: {path}")
            partial = path.with_suffix(path.suffix + ".download")
            print(f"Downloading {relative}: {expected_size / 1e6:.1f} MB", flush=True)
            start = last_report = time.monotonic()
            count = 0
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
                    h.update(chunk)
                    count += len(chunk)
                    if time.monotonic() - last_report >= 10:
                        print(f"  {count / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
                        last_report = time.monotonic()
            if count != expected_size or h.hexdigest() != expected_hash:
                raise RuntimeError(f"Size or SHA256 mismatch: {relative}; partial retained for diagnosis")
            os.replace(partial, path)
            print(f"Verified {relative} ({time.monotonic() - start:.1f}s)", flush=True)
        else:
            print(f"Already verified: {relative}", flush=True)
        records.append({"path": relative, "bytes": expected_size, "sha256": expected_hash, "url": url})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"upstream_commit": COMMIT, "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
