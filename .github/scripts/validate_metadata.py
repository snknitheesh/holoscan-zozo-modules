# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Validate metadata.json files against the schemas of the installed Holoscan CLI."""

import json
import sys
from pathlib import Path

from holoscan_cli.metadata.metadata_validator import validate_json


def main(filenames: list[str]) -> int:
    failed = False
    for name in filenames:
        path = Path(name)
        try:
            valid, message = validate_json(
                json.loads(path.read_text(encoding="utf-8")), path.parent
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            valid, message = False, error
        if not valid:
            print(f"{path}: {message}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
