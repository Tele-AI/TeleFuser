#!/usr/bin/env python3
"""Update tf-kernel's independently released package version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?$")
PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="PEP 440 release version, for example 0.1.1 or 0.2.0rc1")
    return parser.parse_args()


def main() -> int:
    """Replace the single project version declaration in pyproject.toml."""
    args = parse_args()
    if VERSION_PATTERN.fullmatch(args.version) is None:
        raise SystemExit(f"Invalid tf-kernel version: {args.version}")

    contents = PYPROJECT_PATH.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r'(?m)^version = "[^"]+"$',
        f'version = "{args.version}"',
        contents,
        count=1,
    )
    if replacements != 1:
        raise SystemExit(f"Expected exactly one project version in {PYPROJECT_PATH}")

    PYPROJECT_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated tf-kernel version to {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
