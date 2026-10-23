"""Offline installer acceptance; never change the real user's home or PATH."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
GIT_BASH = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.exists() else shutil.which("bash")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


class Installers(unittest.TestCase):
    def exercise(self, windows=False, override=False, bad_checksum=False, conflict=False, quote=False, rollback=False):
        with tempfile.TemporaryDirectory(prefix="infer installation spaces ") as directory:
            root = Path(directory)
            home = root / ("owner's home" if quote else "user home")
            shared = home / "rune"
            target = (root / "custom installation") if override else shared
            shared.mkdir(parents=True)
            (shared / "tools/plugins").mkdir(parents=True)
            kit = shared / ("rune-kit.exe" if windows else "rune-kit")
            kit.write_bytes(b"kit-owned executable")
            plugin = shared / "tools/plugins/keep.txt"
            plugin.write_bytes(b"kit-owned plugin")
            legacy = home / ".rune-infer"
            legacy.mkdir()
            (legacy / ".env").write_bytes(b"KEEP_ORIGINAL=private-fixture\n")
            assets = root / "assets"
            assets.mkdir()
            binary_name = "rune-infer.exe" if windows else "rune-infer"
            if windows:
                binary = ROOT / "target/debug/rune-infer.exe"
                self.assertTrue(binary.exists(), "Run cargo build --locked before Windows installer tests")
                name = "rune-infer-windows-x64.zip"
                with zipfile.ZipFile(assets / name, "w") as archive:
                    archive.write(binary, binary_name)
            else:
                name = "rune-infer-linux-x64.tar.gz"
                payload = b'#!/bin/sh\nprintf "rune-infer 0.8.0\\n"\n'
                if rollback:
                    payload = b'#!/bin/sh\ncase "$0" in */rune/rune-infer) exit 1 ;; esac\nprintf "rune-infer 0.8.0\\n"\n'
                with tarfile.open(assets / name, "w:gz") as archive:
                    info = tarfile.TarInfo(binary_name)
                    info.size, info.mode = len(payload), 0o755
                    archive.addfile(info, io.BytesIO(payload))
            digest = "0" * 64 if bad_checksum else hashlib.sha256((assets / name).read_bytes()).hexdigest()
            (assets / (name + ".sha256")).write_text(f"{digest}  {name}\n")
            if conflict:
                target.mkdir(exist_ok=True)
                (target / binary_name).write_bytes(b"unrelated destination data")
            previous = b'#!/bin/sh\nprintf "rune-infer 0.7.0\\n"\n'
            if rollback:
                (target / binary_name).write_bytes(previous)
                (target / binary_name).chmod(0o755)
            env = os.environ.copy()
            # A parent PowerShell 7 module path can hide Windows PowerShell 5's
            # built-in modules; let the child choose its native module paths.
            for key in list(env):
                if key.lower() == "psmodulepath":
                    env.pop(key)
            for key in list(env):
                if key.startswith("RUNE_"):
                    env.pop(key)
            env.update(USERPROFILE=str(home), LOCALAPPDATA=str(root / "old local app data"),
                       HOME=str(home) if windows else shell_path(home), RUNE_VERSION="v0.8.0",
                       RUNE_PLATFORM="windows-x64" if windows else "linux-x64",
                       RUNE_BASE_URL=str(assets) if windows else shell_path(assets),
                       RUNE_NO_MODIFY_PATH="1" if windows else "0", SHELL="/bin/bash")
            if override:
                env["RUNE_INSTALL_DIR"] = str(target) if windows else shell_path(target)
            old_bin = (root / "old local app data/Programs/rune-infer" if windows else home / ".local/bin")
            old_bin.mkdir(parents=True)
            (old_bin / binary_name).write_bytes(b"retained old executable")
            command = (["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "install.ps1")]
                       if windows else [BASH, "--noprofile", "--norc", shell_path(ROOT / "install.sh")])
            for _ in range(1 if bad_checksum or conflict or rollback else 2):
                result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=60)
                if bad_checksum or conflict or rollback:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue((target / binary_name).exists())
                self.assertEqual(kit.read_bytes(), b"kit-owned executable")
                self.assertEqual(plugin.read_bytes(), b"kit-owned plugin")
                self.assertEqual((old_bin / binary_name).read_bytes(), b"retained old executable")
                self.assertEqual((legacy / ".env").read_bytes(), b"KEEP_ORIGINAL=private-fixture\n")
            if rollback:
                self.assertEqual((target / binary_name).read_bytes(), previous)
            elif conflict:
                self.assertEqual((target / binary_name).read_bytes(), b"unrelated destination data")
            elif bad_checksum:
                self.assertFalse((target / binary_name).exists())
            elif not windows:
                profile = home / ".profile"
                self.assertEqual(profile.read_text().count("# Rune shared PATH"), 1)
                check = subprocess.run([BASH, "--noprofile", "--norc", "-c", '. "$HOME/.profile"; . "$HOME/.profile"; printf "%s" "$PATH"'], env=env, capture_output=True, text=True)
                self.assertEqual(check.returncode, 0, check.stderr)
                self.assertEqual(check.stdout.split(":").count(shell_path(target)), 1)

    @unittest.skipUnless(os.name == "nt", "native Windows installer")
    def test_windows_default_repeat_and_kit_preservation(self):
        self.exercise(windows=True)

    @unittest.skipUnless(os.name == "nt", "native Windows installer")
    def test_windows_override_and_checksum_and_conflict(self):
        self.exercise(windows=True, override=True)
        self.exercise(windows=True, bad_checksum=True)
        self.exercise(windows=True, conflict=True)

    @unittest.skipUnless(BASH, "requires a POSIX shell")
    def test_posix_default_repeat_and_path_dedup(self):
        self.exercise()

    @unittest.skipUnless(BASH, "requires a POSIX shell")
    def test_posix_override_quotes_checksum_and_conflict(self):
        self.exercise(override=True, quote=True)
        self.exercise(bad_checksum=True)
        self.exercise(conflict=True)

    @unittest.skipUnless(BASH, "requires a POSIX shell")
    def test_posix_failed_installed_validation_restores_previous_executable(self):
        self.exercise(rollback=True)


if __name__ == "__main__":
    unittest.main()
