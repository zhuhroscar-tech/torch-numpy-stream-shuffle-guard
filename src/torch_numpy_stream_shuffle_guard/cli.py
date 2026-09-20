"""Command-line interface: run the from-scratch diagnosis of the
torch._numpy.random.shuffle row-multiset corruption bug against the
currently installed torch build, using the shared semantic-color design
system."""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-numpy-stream-shuffle-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch._numpy.random.shuffle (under "
            "use_numpy_random_stream=True) corrupts a tensor's row "
            "multiset -- duplicating some rows and dropping others "
            "instead of performing a true permutation "
            "(pytorch/pytorch#197795) -- and verify the safe_row_shuffle "
            "guard always preserves the row multiset, on THIS host's "
            "actual installed torch version -- never trusts the upstream "
            "issue's reported version alone."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-numpy-stream-shuffle-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_effective"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("tracking issue", ", ".join(report["issue_urls"])),
        ]
    )

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "row-multiset corruption reproduced on this host's installed torch build"))
    else:
        print(status_headline(style, "info", "bug NOT reproduced on this host's installed torch build (fixed upstream)"))

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "safe_row_shuffle always preserves the row multiset at every tested seed/shape"))
    else:
        print(status_headline(style, "fail", "the guard itself failed to preserve the row multiset in at least one case"))

    section("per-case results (seed x shape)")
    for c in report["cases"]:
        bug_flag = "MULTISET-CORRUPTED" if not c["unsafe_preserved_multiset"] else "preserved"
        guard_flag = "guard-ok" if c["guard_preserved_multiset"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"seed={c['seed']} shape=({c['n_rows']},{c['n_cols']})",
                    f"unsafe={bug_flag:20s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_effective"] else 1


if __name__ == "__main__":
    sys.exit(main())
