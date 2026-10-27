# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Check that each file has an Apache-2.0 SPDX identifier and a copyright notice.

A file changed against COPYRIGHT_BASE_REF (or HEAD, or untracked) must name the
current year in at least one of its notices.
"""

import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LICENSE = "SPDX-License-Identifier: Apache-2.0"
# "Copyright (c) 2026 Name" or "Copyright (c) 2024-2026, Name", optionally after an
# SPDX-FileCopyrightText tag and a comment marker.
NOTICE = re.compile(
    r"^[ \t]*(?:#|//|/\*|\*|<!--)?[ \t]*(?:SPDX-FileCopyrightText:[ \t]*)?"
    r"Copyright[ \t]+(?:\(c\)[ \t]*)?(\d{4})(?:-(\d{4}))?,?[ \t]+\S",
    re.MULTILINE | re.IGNORECASE,
)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, check=False)


def changed_files() -> set[Path] | None:
    """Paths changed since the base, or None when every file counts as changed."""
    base = os.environ.get("COPYRIGHT_BASE_REF", "")
    top = git("rev-parse", "--show-toplevel")
    # Outside a repository, on a branch's first push (an all-zero base) or before the
    # first commit, there is nothing to compare against.
    if top.returncode or (base and set(base) == {"0"}):
        return None
    root = Path(os.fsdecode(top.stdout).strip())
    revision = git("rev-parse", "--verify", "--end-of-options", f"{base or 'HEAD'}^{{commit}}")
    if revision.returncode:
        if base:
            raise ValueError(f"cannot resolve {base!r}; fetch the history it needs")
        return None
    diff = git(
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACMRT",
        "-z",
        os.fsdecode(revision.stdout).strip(),
        "--",
    )
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    names = (diff.stdout + untracked.stdout).split(b"\0")
    return {(root / os.fsdecode(name)).resolve() for name in names if name}


def check(path: Path, changed: set[Path] | None, year: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return [f"{path}: {error}"]
    errors = []
    if LICENSE not in text:
        errors.append(f"{path}: missing '{LICENSE}'")
    notices = [(int(first), int(last or first)) for first, last in NOTICE.findall(text)]
    if not notices:
        errors.append(f"{path}: missing copyright notice")
    elif any(not 1970 <= first <= last <= year for first, last in notices):
        errors.append(f"{path}: copyright year is invalid or in the future")
    elif (changed is None or path.resolve() in changed) and all(
        last != year for _, last in notices
    ):
        errors.append(f"{path}: changed, so its copyright notice must include {year}")
    return errors


def main(filenames: list[str]) -> int:
    try:
        changed = changed_files()
    except ValueError as error:
        print(f"copyright check: {error}", file=sys.stderr)
        return 1
    year = datetime.now(UTC).year
    errors = [error for name in filenames for error in check(Path(name), changed, year)]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
