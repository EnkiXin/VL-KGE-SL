"""Fetch the pinned public author source and apply only its checkpoint import fix.

No datasets or model weights are downloaded. Existing checkouts are never reset,
cleaned, switched to another commit, or overwritten. --check-only performs only
read-only verification and does not contact the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


URL = "https://github.com/thefth/vl-kge"
COMMIT = "c78994e14cf2dfda251b701c2803215d9d5fe254"
PATCH_SHA256 = "9fa2b64e292ce68e1a78683e593d77f78c8602edc4d95d9f7b96eb129622c96d"
HELPERS = "vlkge/helpers.py"
LFS_FILES = (
    "vlkge/data/wn9_img/wn9_img_triples.csv",
    "vlkge/data/wn9_img/features/wn9_img_vf_clip.pkl",
    "vlkge/data/wn9_img/features/wn9_img_tf_clip.pkl",
)
# Historical local observer documentation/tests may exist but are not changed.
ALLOWED_UNTRACKED = {"REPRODUCTION.md", "requirements-reproduction.txt", "tests/test_reproduction_static.py"}


def git(repo, *arguments):
    environment = os.environ.copy()
    environment.update(GIT_LFS_SKIP_SMUDGE="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(["git", *arguments], cwd=repo, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Git verification failed for {arguments[0]} (exit {result.returncode}); checkout left unchanged")
    return result.stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_patched_helpers(original):
    """Independent exact specification of the reviewed three import edits."""
    expected = original
    edits = (
        (b"from vlkge.dataloader import KGDataset\n", b"from vlkge.dataloader import KGDataset\nfrom vlkge import utils\n"),
        (b"\n        import utils\n", b"\n"),
        (b"\n                    import utils\n", b"\n"),
    )
    for before, after in edits:
        if expected.count(before) != 1:
            raise RuntimeError("Pinned helpers no longer match the reviewed import patch")
        expected = expected.replace(before, after, 1)
    # The reviewed patch also adds the missing final newline, without code edits.
    return expected if expected.endswith(b"\n") else expected + b"\n"


def verify_patch_file(patch):
    if not patch.is_file() or sha256(patch) != PATCH_SHA256:
        raise RuntimeError("Import patch is missing or differs from its reviewed SHA256")


def verify_lfs_file(repo, relative):
    pointer = git(repo, "show", f"{COMMIT}:{relative}")
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing or unexpected LFS input: {relative}")
    if path.stat().st_size == len(pointer) and path.read_bytes() == pointer:
        return "pointer_not_downloaded"
    fields = dict(line.split(" ", 1) for line in pointer.decode("utf-8").strip().splitlines())
    if fields.get("version") != "https://git-lfs.github.com/spec/v1" or not fields.get("oid", "").startswith("sha256:"):
        raise RuntimeError(f"Pinned file is not a recognized LFS pointer: {relative}")
    if path.stat().st_size != int(fields["size"]) or sha256(path) != fields["oid"].split(":", 1)[1]:
        raise RuntimeError(f"Expanded LFS file differs from the author's checksum: {relative}")
    return "expanded_and_verified"


def verify_checkout(repo, patch):
    """Check identity, tracked source/config bytes, and permissible local state."""
    verify_patch_file(patch)
    if not repo.is_dir() or repo.is_symlink():
        raise RuntimeError("Expected a real upstream checkout directory")
    top = Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != repo.resolve():
        raise RuntimeError("upstream/vl-kge is not the root of its own Git worktree")
    if git(repo, "rev-parse", "HEAD").decode().strip() != COMMIT:
        raise RuntimeError("Existing upstream is on a different commit; refusing to switch or reset it")
    if git(repo, "diff", "--cached", "--name-only", "-z"):
        raise RuntimeError("Existing upstream has staged changes; refusing to modify it")
    original = git(repo, "show", f"{COMMIT}:{HELPERS}")
    current = (repo / HELPERS).read_bytes()
    if current == original:
        patch_state = "needs_patch"
    elif current == expected_patched_helpers(original):
        patch_state = "already_patched"
    else:
        raise RuntimeError("helpers.py contains edits beyond the exact import fix")
    paths = git(repo, "ls-files", "-z", "*.py", "*.yaml", "*.yml").decode().split("\0")
    for relative in filter(None, paths):
        if relative == HELPERS:
            continue
        path = repo / relative
        if not path.is_file() or path.is_symlink() or path.read_bytes() != git(repo, "show", f"{COMMIT}:{relative}"):
            raise RuntimeError(f"Tracked author source/config changed: {relative}")
    allowed_modified = {HELPERS, *LFS_FILES}
    allowed_local = []
    for record in filter(None, git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all").decode().split("\0")):
        state, relative = record[:2], record[3:]
        if state == "??" and relative in ALLOWED_UNTRACKED:
            allowed_local.append(relative)
        elif state == " M" and relative in allowed_modified:
            continue
        else:
            raise RuntimeError(f"Unrelated dirty/untracked author-checkout path: {relative}; refusing to modify it")
    lfs = {relative: verify_lfs_file(repo, relative) for relative in LFS_FILES}
    if patch_state == "already_patched":
        git(repo, "apply", "--reverse", "--check", str(patch))
    else:
        git(repo, "apply", "--check", str(patch))
    return {"upstream_commit": COMMIT, "patch_state": patch_state,
            "patch_sha256": PATCH_SHA256, "tracked_source_and_yaml_verified": True,
            "lfs_inputs": lfs, "preserved_local_extras": sorted(allowed_local)}


def setup(root, *, check_only=False):
    root = root.resolve()
    repo = root / "upstream/vl-kge"
    patch = root / "patches/author-utils-import.patch"
    verify_patch_file(patch)
    created = False
    if not repo.exists():
        if check_only:
            raise RuntimeError("Upstream source is absent; --check-only does not clone or download")
        repo.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(GIT_LFS_SKIP_SMUDGE="1", GIT_OPTIONAL_LOCKS="0")
        subprocess.run(["git", "clone", "--no-checkout", URL, str(repo)], env=environment, check=True)
        # This checkout is newly created by us; existing worktrees never reach this branch.
        git(repo, "checkout", "--detach", COMMIT)
        created = True
    result = verify_checkout(repo, patch)
    if result["patch_state"] == "needs_patch":
        if check_only:
            raise RuntimeError("Pinned source is intact but import fix is not installed; --check-only made no changes")
        git(repo, "apply", "--check", str(patch))
        git(repo, "apply", str(patch))
        result = verify_checkout(repo, patch)
    return {**result, "status": "ready", "created_checkout": created, "check_only": check_only,
            "dataset_download_performed": False,
            "next_step": "python scripts/fetch_wn9.py --repo upstream/vl-kge --manifest artifacts/wn9-inputs.json"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(setup(args.root, check_only=args.check_only), indent=2))


if __name__ == "__main__":
    main()
