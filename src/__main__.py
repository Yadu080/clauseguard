"""Command-line interface.

    python -m src "Candidates must submit the form." "Candidates may submit the form."
    python -m src --passage original.txt rewritten.txt --json
    python -m src ... --markdown > report.md

Exit code: 0 = no configured checks triggered, 1 = potential drift (HIGH/MEDIUM),
2 = manual review recommended. Useful in scripts or CI checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import analyze
from .export import passage_to_markdown, result_to_markdown
from .passage import analyze_passage
from .schemas import Status

EXIT_CODES = {Status.NONE: 0, Status.DRIFT: 1, Status.REVIEW: 2}


def _read(value: str, is_file: bool) -> str:
    return Path(value).read_text(encoding="utf-8") if is_file else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src", description="Detect meaning drift between an "
                                     "original instruction and its rewrite.")
    parser.add_argument("original", help="original sentence (or a file path with --files/--passage)")
    parser.add_argument("rewritten", help="rewritten sentence (or a file path with --files/--passage)")
    parser.add_argument("--files", action="store_true", help="treat both arguments as text files")
    parser.add_argument("--passage", action="store_true", help="passage mode: arguments are text files "
                        "containing several sentences")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print the full structured result as JSON")
    out.add_argument("--markdown", action="store_true", help="print a Markdown report")
    args = parser.parse_args(argv)

    from_files = args.files or args.passage
    original, rewritten = _read(args.original, from_files), _read(args.rewritten, from_files)

    if args.passage:
        result = analyze_passage(original, rewritten)
        if args.json:
            print(result.to_json())
        else:
            print(passage_to_markdown(result))
        return EXIT_CODES[result.status]

    result = analyze(original, rewritten)
    if args.json:
        print(result.to_json())
    elif args.markdown:
        print(result_to_markdown(result))
    else:
        print(result.status.value)
        for f in result.findings:
            print(f"  {f.severity.value:<6} {f.category:<10} {f.rule_id:<28} {f.explanation}")
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
