import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NotesTests(unittest.TestCase):
    def test_exact_section_and_historical_text_preserved(self):
        section = "## [0.8.0] - 2026-10-23\n\n### Fixes\n\n- Preserve existing configuration while migrating to the new home.\n"
        history = "\n## [0.7.0]\n\n- Historical description with an internal milestone 1 label.\n"
        self.assertEqual(module.extract("# Changelog\n\n" + section + history, "0.8.0"), section)

    def test_missing_empty_placeholder_metadata_duplicate_and_mismatch_fail(self):
        for text in ["# Changelog\n", "## [0.8.0]\n\n### Fixes\n", "## [0.8.0]\n\n- TODO fill in changes before release.\n",
                     "## [0.8.0]\n\n- Version bump to the next release.\n", "## [0.8.0]\n- Improve the user installation paths.\n## [0.8.0]\n- Duplicate release section is invalid.\n",
                     "## [0.9.0]\n- Newer release section first.\n## [0.8.0]\n- Improve the user installation paths.\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.extract(text, "0.8.0")

    def test_cli_tag_mismatch_fails_without_output(self):
        with tempfile.TemporaryDirectory(prefix="infer notes spaces ") as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n')
            (root / "CHANGELOG.md").write_text('## [0.8.0]\n\n- Preserve existing configuration on upgrade.\n')
            output = root / "notes.md"
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--version", "v0.9.0", "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
