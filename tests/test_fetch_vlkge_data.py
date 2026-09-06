"""Pinned, dataset-specific download safety checks; no real network access."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_vlkge_tested", ROOT / "scripts/fetch_wn9.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)
SETUP_SPEC = importlib.util.spec_from_file_location("setup_vlkge_fetch_tested", ROOT / "scripts/setup_upstream.py")
setup = importlib.util.module_from_spec(SETUP_SPEC)
SETUP_SPEC.loader.exec_module(setup)


class FetchVLKGEDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "upstream"
        self.repo.mkdir()
        self.payloads = {}
        self.pointers = {}
        for files in fetch.DATASET_FILES.values():
            for relative in files:
                payload = ("trusted test data for " + relative).encode()
                self.payloads[relative] = payload
                pointer = ("version https://git-lfs.github.com/spec/v1\n"
                           f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
                           f"size {len(payload)}\n").encode()
                self.pointers[relative] = pointer
                path = self.repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pointer)
        self.downloaded = []

    def fake_git(self, command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return fetch.COMMIT + "\n"
        self.assertEqual(command[:2], ["git", "show"])
        return self.pointers[command[2].split(":", 1)[1]]

    def fake_open(self, url, **kwargs):
        self.assertTrue(url.startswith("https://media.githubusercontent.com/media/thefth/vl-kge/"))
        relative = url.split(fetch.COMMIT + "/", 1)[1]
        self.downloaded.append(relative)
        return io.BytesIO(self.payloads[relative])

    def test_legacy_argument_defaults_and_allowlist_are_exact(self):
        args = fetch.parse_args(["--repo", str(self.repo), "--manifest", "wn9-inputs.json"])
        self.assertEqual(args.datasets, ["wn9_img"])
        self.assertEqual(fetch.FILES, list(fetch.DATASET_FILES["wn9_img"]))
        self.assertEqual(setup.LFS_FILES, tuple(path for files in fetch.DATASET_FILES.values() for path in files))
        self.assertEqual(len(setup.LFS_FILES), 9)
        for argv in (
            ["--datasets", "wikiart_mkg_v1", "wikiart_mkg_v2", "--manifest", "same.json"],
            ["--datasets", "wn9_img", "wn9_img", "--manifest-dir", "artifacts"],
        ):
            with self.assertRaises(SystemExit):
                fetch.parse_args(["--repo", str(self.repo)] + argv)

    def test_only_wikiart_downloaded_and_wn9_manifest_not_overwritten(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        old_manifest = artifacts / "wn9-inputs.json"
        old_manifest.write_text("legacy WN9 experiment, do not touch\n")
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                patch.object(fetch.urllib.request, "urlopen", side_effect=self.fake_open):
            fetch.main(["--repo", str(self.repo), "--datasets", "wikiart_mkg_v1", "wikiart_mkg_v2",
                        "--manifest-dir", str(artifacts)])
        self.assertEqual(self.downloaded, list(fetch.DATASET_FILES["wikiart_mkg_v1"])
                         + list(fetch.DATASET_FILES["wikiart_mkg_v2"]))
        self.assertEqual(old_manifest.read_text(), "legacy WN9 experiment, do not touch\n")
        for dataset in ("wikiart_mkg_v1", "wikiart_mkg_v2"):
            record = json.loads((artifacts / fetch.manifest_name(dataset)).read_text())
            self.assertEqual(record["dataset"], dataset)
            self.assertEqual(record["upstream_commit"], fetch.COMMIT)
            self.assertEqual([item["path"] for item in record["files"]], list(fetch.DATASET_FILES[dataset]))
        for relative in fetch.FILES:
            self.assertEqual((self.repo / relative).read_bytes(), self.pointers[relative])

    def test_verified_inputs_and_legacy_manifest_are_idempotent(self):
        for relative in fetch.FILES:
            (self.repo / relative).write_bytes(self.payloads[relative])
        manifest = self.root / "wn9-inputs.json"
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git):
            records, _ = fetch.records_from_pointers(self.repo, "wn9_img")
        # The old manifest has no dataset field; retain its bytes exactly.
        manifest.write_text(json.dumps({"upstream_commit": fetch.COMMIT, "files": records}))
        before = manifest.read_bytes()
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                patch.object(fetch.urllib.request, "urlopen") as network:
            fetch.main(["--repo", str(self.repo), "--manifest", str(manifest)])
            network.assert_not_called()
        self.assertEqual(manifest.read_bytes(), before)

    def test_wrong_hash_preserves_original_pointer_and_no_ready_manifest(self):
        dataset = "wikiart_mkg_v2"
        relative = fetch.DATASET_FILES[dataset][0]
        manifest = self.root / "wikiart_mkg_v2-inputs.json"
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                patch.object(fetch.urllib.request, "urlopen", return_value=io.BytesIO(b"bad payload")), \
                self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            fetch.fetch_dataset(self.repo, dataset, manifest)
        self.assertEqual((self.repo / relative).read_bytes(), self.pointers[relative])
        self.assertFalse(manifest.exists())
        self.assertEqual(len(list((self.repo / relative).parent.glob("*.download-*"))), 1)

    def test_unrelated_file_or_symlink_cannot_be_replaced(self):
        relative = fetch.DATASET_FILES["wikiart_mkg_v1"][0]
        path = self.repo / relative
        unrelated = self.pointers[relative] + b"unrelated data, even with matching prefix"
        path.write_bytes(unrelated)
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                patch.object(fetch.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "neither matching data"):
                fetch.fetch_dataset(self.repo, "wikiart_mkg_v1", self.root / "new.json")
            network.assert_not_called()
        self.assertEqual(path.read_bytes(), unrelated)
        path.unlink()
        outside = self.root / "outside-user-file"
        outside.write_bytes(self.pointers[relative])
        path.symlink_to(outside)
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                self.assertRaisesRegex(RuntimeError, "symlink input"):
            fetch.fetch_dataset(self.repo, "wikiart_mkg_v1", self.root / "new.json")
        self.assertEqual(outside.read_bytes(), self.pointers[relative])

    def test_wrong_dataset_manifest_rejected_before_download(self):
        manifest = self.root / "wn9-inputs.json"
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git):
            wn9_records, _ = fetch.records_from_pointers(self.repo, "wn9_img")
        manifest.write_text(json.dumps({"upstream_commit": fetch.COMMIT, "files": wn9_records}))
        before = manifest.read_bytes()
        with patch.object(fetch.subprocess, "check_output", side_effect=self.fake_git), \
                patch.object(fetch.urllib.request, "urlopen") as network, \
                self.assertRaisesRegex(RuntimeError, "different inputs"):
            fetch.fetch_dataset(self.repo, "wikiart_mkg_v1", manifest)
        network.assert_not_called()
        self.assertEqual(manifest.read_bytes(), before)

    def test_wrong_commit_rejected_before_any_download(self):
        with patch.object(fetch.subprocess, "check_output", return_value="wrong\n"), \
                patch.object(fetch.urllib.request, "urlopen") as network, \
                self.assertRaisesRegex(RuntimeError, "Unexpected upstream commit"):
            fetch.main(["--repo", str(self.repo), "--manifest-dir", str(self.root / "artifacts")])
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
