"""Bootstrap safety checks; no network, Git initialization, or real checkout edits."""

import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("setup_upstream_tested", ROOT / "scripts/setup_upstream.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class SetupUpstreamTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "patches").mkdir()
        shutil.copyfile(ROOT / "patches/author-utils-import.patch", self.root / "patches/author-utils-import.patch")
        self.repo = self.root / "upstream/vl-kge"
        self.repo.mkdir(parents=True)
        self.original = b"from vlkge.dataloader import KGDataset\n\ndef f():\n        import utils\n        if True:\n                    import utils\n        return 1"
        helpers = self.repo / setup.HELPERS
        helpers.parent.mkdir(parents=True)
        helpers.write_bytes(self.original)
        self.tracked = {setup.HELPERS: self.original, "vlkge/configs/tiny.yaml": b"epochs: 200\n"}
        for relative in setup.LFS_FILES:
            payload = b"tiny test data"
            pointer = ("version https://git-lfs.github.com/spec/v1\n"
                       f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n").encode()
            self.tracked[relative] = pointer
        for relative, data in self.tracked.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.head = setup.COMMIT
        self.staged = b""
        self.status = b""
        self.git_calls = []

    def fake_git(self, repo, *args):
        self.git_calls.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            return (str(self.repo) + "\n").encode()
        if args == ("rev-parse", "HEAD"):
            return (self.head + "\n").encode()
        if args == ("diff", "--cached", "--name-only", "-z"):
            return self.staged
        if args[0] == "show":
            return self.tracked[args[1].split(":", 1)[1]]
        if args[0] == "ls-files":
            return (setup.HELPERS + "\0vlkge/configs/tiny.yaml\0").encode()
        if args[0] == "status":
            return self.status
        if args[0] == "apply":
            if "--check" not in args:
                (self.repo / setup.HELPERS).write_bytes(setup.expected_patched_helpers(self.original))
            return b""
        raise AssertionError(f"Unexpected test Git call: {args}")

    def test_existing_original_applies_only_reviewed_import_patch(self):
        with patch.object(setup, "git", side_effect=self.fake_git):
            result = setup.setup(self.root)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["created_checkout"])
        self.assertFalse(result["dataset_download_performed"])
        self.assertEqual((self.repo / setup.HELPERS).read_bytes(), setup.expected_patched_helpers(self.original))
        self.assertEqual(sum(args[0] == "apply" and "--check" not in args for args in self.git_calls), 1)
        self.assertFalse(any(args[0] in ("checkout", "reset", "clean", "clone") for args in self.git_calls))

    def test_check_only_missing_or_unpatched_never_clones_or_applies(self):
        with patch.object(setup, "git", side_effect=self.fake_git), patch.object(setup.subprocess, "run") as process:
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                setup.setup(self.root, check_only=True)
            process.assert_not_called()
        self.assertFalse(any(args[0] == "apply" and "--check" not in args for args in self.git_calls))
        missing = self.root / "missing-project"
        (missing / "patches").mkdir(parents=True)
        shutil.copyfile(self.root / "patches/author-utils-import.patch", missing / "patches/author-utils-import.patch")
        with patch.object(setup.subprocess, "run") as process:
            with self.assertRaisesRegex(RuntimeError, "does not clone"):
                setup.setup(missing, check_only=True)
            process.assert_not_called()

    def test_already_patched_is_verified_without_second_apply(self):
        patched = setup.expected_patched_helpers(self.original)
        (self.repo / setup.HELPERS).write_bytes(patched)
        self.status = b" M vlkge/helpers.py\0?? REPRODUCTION.md\0"
        with patch.object(setup, "git", side_effect=self.fake_git):
            result = setup.setup(self.root, check_only=True)
        self.assertEqual(result["patch_state"], "already_patched")
        self.assertIn(("apply", "--reverse", "--check", str(self.root / "patches/author-utils-import.patch")), self.git_calls)
        self.assertFalse(any(args[0] == "apply" and "--check" not in args for args in self.git_calls))
        self.assertEqual((self.repo / setup.HELPERS).read_bytes(), patched)

    def test_wrong_head_staged_unrelated_or_yaml_edits_are_rejected(self):
        scenarios = ("head", "staged", "dirty", "unknown", "yaml", "helpers")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.head, self.staged, self.status = setup.COMMIT, b"", b""
                (self.repo / setup.HELPERS).write_bytes(self.original)
                (self.repo / "vlkge/configs/tiny.yaml").write_bytes(self.tracked["vlkge/configs/tiny.yaml"])
                if scenario == "head":
                    self.head = "wrong"
                elif scenario == "staged":
                    self.staged = b"README.md\0"
                elif scenario == "dirty":
                    self.status = b" M README.md\0"
                elif scenario == "unknown":
                    self.status = b"?? vlkge/new_algorithm.py\0"
                elif scenario == "yaml":
                    (self.repo / "vlkge/configs/tiny.yaml").write_text("epochs: 20\n")
                else:
                    (self.repo / setup.HELPERS).write_bytes(self.original + b"\n# unrelated edit\n")
                before = (self.repo / setup.HELPERS).read_bytes()
                with patch.object(setup, "git", side_effect=self.fake_git), self.assertRaises(RuntimeError):
                    setup.setup(self.root)
                self.assertEqual((self.repo / setup.HELPERS).read_bytes(), before)

    def test_only_checksum_matching_expanded_lfs_data_are_accepted(self):
        data = self.repo / setup.LFS_FILES[0]
        data.write_bytes(b"tiny test data")
        with patch.object(setup, "git", side_effect=self.fake_git):
            self.assertEqual(setup.verify_lfs_file(self.repo, setup.LFS_FILES[0]), "expanded_and_verified")
        data.write_bytes(b"corrupted data")
        with patch.object(setup, "git", side_effect=self.fake_git), self.assertRaisesRegex(RuntimeError, "checksum"):
            setup.verify_lfs_file(self.repo, setup.LFS_FILES[0])
        self.assertEqual(data.read_bytes(), b"corrupted data")

    def test_changed_patch_is_rejected_before_any_git_operation(self):
        (self.root / "patches/author-utils-import.patch").write_text("unreviewed patch\n")
        with patch.object(setup, "git") as git, self.assertRaisesRegex(RuntimeError, "reviewed SHA256"):
            setup.setup(self.root)
        git.assert_not_called()

    def test_all_three_dataset_expansions_allowed_only_with_matching_hashes(self):
        (self.repo / setup.HELPERS).write_bytes(setup.expected_patched_helpers(self.original))
        for relative in setup.LFS_FILES:
            (self.repo / relative).write_bytes(b"tiny test data")
        self.status = "".join(f" M {relative}\0" for relative in setup.LFS_FILES).encode()
        with patch.object(setup, "git", side_effect=self.fake_git):
            result = setup.setup(self.root, check_only=True)
        self.assertEqual(len(result["lfs_inputs"]), 9)
        self.assertTrue(all(value == "expanded_and_verified" for value in result["lfs_inputs"].values()))
        wikiart_path = next(relative for relative in setup.LFS_FILES if "wikiart_mkg_v2" in relative)
        (self.repo / wikiart_path).write_bytes(b"corrupted data")
        with patch.object(setup, "git", side_effect=self.fake_git), self.assertRaisesRegex(RuntimeError, "checksum"):
            setup.setup(self.root, check_only=True)

    def test_non_clip_data_expansion_is_not_whitelisted(self):
        self.status = b" M vlkge/data/wikiart_mkg_v1/features/wikiart_mkg_v1_vf_blip.pkl\0"
        with patch.object(setup, "git", side_effect=self.fake_git), self.assertRaisesRegex(RuntimeError, "Unrelated dirty"):
            setup.setup(self.root, check_only=True)

    def test_published_license_is_verbatim_author_file(self):
        self.assertEqual((ROOT / "third_party/vl-kge-LICENSE").read_bytes(),
                         (ROOT / "upstream/vl-kge/LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
