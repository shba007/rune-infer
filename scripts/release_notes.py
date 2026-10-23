"""Preview exact changelog release notes; never publish or write Git."""
import argparse
from pathlib import Path
import re
import sys
import tomllib


def extract(changelog: str, version: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("Release version must be a complete semantic version")
    headings = list(re.finditer(r"^## \[([^\]]+)\][^\n]*$", changelog, re.M))
    matches = [h for h in headings if h.group(1) == version]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CHANGELOG section for {version}")
    if headings[0] != matches[0]:
        raise ValueError("Cargo version does not match the first CHANGELOG version")
    start = matches[0]
    end = next((h.start() for h in headings if h.start() > start.start()), len(changelog))
    section = changelog[start.start():end].strip() + "\n"
    body = section.split("\n", 1)[1]
    substantive = [line for line in body.splitlines() if re.match(r"^[-*] \S.{14,}$", line)
                   and not re.match(r"^[-*] (?:bump\b|version bump\b|prepare (?:the )?release\b|release v?\d)", line, re.I)]
    if not substantive:
        raise ValueError("Release section is empty or has no substantive changes")
    if re.search(r"\b(TODO|TBD|placeholder|coming soon)\b", body, re.I):
        raise ValueError("Release section contains a placeholder")
    if re.search(r"\b(cycle\s+\d+|milestone\s+\d+|INF-\d+|IF-\d+)\b", body, re.I):
        raise ValueError("Release section contains an internal planning identifier")
    return section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", help="Expected version/tag; must match Cargo.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = tomllib.loads((args.root / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    if args.version and args.version.removeprefix("v") != version:
        raise ValueError("Requested release tag/version does not match Cargo.toml")
    notes = extract((args.root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if args.output:
        args.output.write_text(notes, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(notes)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f"Release notes error: {error}")
