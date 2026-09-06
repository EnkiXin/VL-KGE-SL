"""Dependency-free regression checks against the pinned author Git objects."""

import ast
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1] / "upstream/vl-kge"
UPSTREAM = "c78994e14cf2dfda251b701c2803215d9d5fe254"


def upstream_bytes(path):
    return subprocess.check_output(["git", "show", f"{UPSTREAM}:{path}"], cwd=ROOT)


class RemoveUtilsImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None if len(node.names) == 1 and node.names[0].name == "utils" else node

    def visit_ImportFrom(self, node):
        if node.module == "vlkge" and len(node.names) == 1 and node.names[0].name == "utils":
            return None
        return node


class AuthorReleaseIntegrity(unittest.TestCase):
    def test_all_python_sources_compile(self):
        self.assertTrue((ROOT / "vlkge/helpers.py").is_file(), "Run scripts/setup_upstream.py first")
        for path in (ROOT / "vlkge").rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_bytes(), str(path), "exec")

    def test_helpers_only_changes_utils_import(self):
        before = ast.parse(upstream_bytes("vlkge/helpers.py"))
        after = ast.parse((ROOT / "vlkge/helpers.py").read_bytes())
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "utils" for alias in node.names)
                             for node in ast.walk(after)))
        normalize = RemoveUtilsImports()
        self.assertEqual(ast.dump(normalize.visit(before)), ast.dump(normalize.visit(after)))

    def test_other_author_sources_and_configs_unchanged(self):
        paths = list((ROOT / "vlkge").rglob("*.py")) + list((ROOT / "vlkge/configs").rglob("*.yaml"))
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if relative != "vlkge/helpers.py":
                with self.subTest(path=relative):
                    self.assertEqual(path.read_bytes(), upstream_bytes(relative))


if __name__ == "__main__":
    unittest.main()
