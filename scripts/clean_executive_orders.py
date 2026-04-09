#!/usr/bin/env python3
"""
Prepare executive-order text files for NLP by removing front/back metadata.

This script removes:
- Header/title lines (e.g., "Title 3", "The President", EO title/date block)
- Footer/signature/metadata lines (e.g., "THE WHITE HOUSE", FR Doc, billing code)

It intentionally does not rewrite or normalize the body text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HEADER_AUTHORITY_RE = re.compile(r"^\s*By the authority vested in me\b", re.IGNORECASE)
HEADER_SECTION_RE = re.compile(
    r"^\s*(Section|Sec\.?|Part)\s+[0-9IVXLC]+\b", re.IGNORECASE
)
HEADER_CORRECTION_RE = re.compile(r"^\s*Correction\s*$", re.IGNORECASE)
HEADER_PRES_DOC_RE = re.compile(r"^\s*In Presidential document\b", re.IGNORECASE)

HEADER_TITLE3_RE = re.compile(r"^\s*Title\s+3\b", re.IGNORECASE)
HEADER_PRESIDENT_RE = re.compile(r"^\s*The President\s*$", re.IGNORECASE)
HEADER_EXEC_ORDER_RE = re.compile(r"^\s*Executive Order\s+\d+", re.IGNORECASE)

FOOTER_WHITE_HOUSE_RE = re.compile(r"^\s*THE WHITE HOUSE,?\s*$", re.IGNORECASE)
FOOTER_FR_DOC_RE = re.compile(r"^\s*\[?\s*FR Doc\.", re.IGNORECASE)
FOOTER_FILED_RE = re.compile(r"^\s*Filed\s+\d", re.IGNORECASE)
FOOTER_BILLING_RE = re.compile(r"^\s*Billing\s+code\b", re.IGNORECASE)
FOOTER_EPS_RE = re.compile(r"^\s*[A-Za-z0-9#._-]+\.EPS\s*$")
FOOTER_SIGNATURE_RE = re.compile(r"^\s*[A-Za-z]{1,4}[.#]?\s*$")


def _is_noise_separator_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 12 and all(ord(ch) > 127 for ch in stripped):
        return True
    if any(ch.isalnum() for ch in stripped):
        return False
    return len(stripped) <= 12


def _find_body_start(lines: list[str]) -> int:
    if not lines:
        return 0

    search_limit = min(len(lines), 160)

    body_anchors: list[int] = []
    for i in range(search_limit):
        line = lines[i]
        if (
            HEADER_AUTHORITY_RE.match(line)
            or HEADER_SECTION_RE.match(line)
            or HEADER_CORRECTION_RE.match(line)
            or HEADER_PRES_DOC_RE.match(line)
        ):
            body_anchors.append(i)

    if body_anchors:
        return min(body_anchors)

    # Conservative fallback for files that only contain EO header metadata.
    has_header_shape = any(
        HEADER_TITLE3_RE.match(lines[i])
        or HEADER_PRESIDENT_RE.match(lines[i])
        or HEADER_EXEC_ORDER_RE.match(lines[i])
        for i in range(min(search_limit, 12))
    )
    if not has_header_shape:
        return 0

    i = 0
    while i < search_limit and not lines[i].strip():
        i += 1

    while i < search_limit and (
        HEADER_TITLE3_RE.match(lines[i])
        or HEADER_PRESIDENT_RE.match(lines[i])
        or HEADER_EXEC_ORDER_RE.match(lines[i])
    ):
        i += 1

    while i < search_limit and not lines[i].strip():
        i += 1

    # Skip one likely subject/title line (e.g., EO name) when no stronger anchor exists.
    if i < search_limit:
        i += 1

    while i < len(lines) and not lines[i].strip():
        i += 1

    return min(i, len(lines))


def _is_footer_marker(line: str) -> bool:
    return bool(
        FOOTER_WHITE_HOUSE_RE.match(line)
        or FOOTER_FR_DOC_RE.match(line)
        or FOOTER_BILLING_RE.match(line)
        or FOOTER_EPS_RE.match(line)
    )


def _trim_footer(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1

    if end == 0:
        return []

    tail_start = max(0, end - 100)
    cut_index: int | None = None
    for i in range(tail_start, end):
        if _is_footer_marker(lines[i]):
            cut_index = i
            break

    if cut_index is not None:
        # Remove nearby signature initials directly above the footer marker.
        while cut_index > 0:
            prev = lines[cut_index - 1].strip()
            if not prev:
                cut_index -= 1
                continue
            if FOOTER_SIGNATURE_RE.match(prev):
                cut_index -= 1
                continue
            if _is_noise_separator_line(prev):
                cut_index -= 1
                continue
            break
        end = cut_index
    else:
        while end > 0:
            line = lines[end - 1]
            if not line.strip():
                end -= 1
                continue
            if (
                FOOTER_FILED_RE.match(line)
                or FOOTER_BILLING_RE.match(line)
                or FOOTER_FR_DOC_RE.match(line)
                or FOOTER_EPS_RE.match(line)
                or _is_noise_separator_line(line)
            ):
                end -= 1
                continue
            break

    while end > 0 and not lines[end - 1].strip():
        end -= 1

    while end > 0 and _is_noise_separator_line(lines[end - 1]):
        end -= 1

    return lines[:end]


def _trim_leading_authority_preamble(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines) or not HEADER_AUTHORITY_RE.match(lines[i]):
        return lines

    j = i + 1
    while j < len(lines):
        line = lines[j]
        if (
            HEADER_SECTION_RE.match(line)
            or HEADER_CORRECTION_RE.match(line)
            or HEADER_PRES_DOC_RE.match(line)
        ):
            return lines[j:]
        j += 1

    # Fallback: if no clear body anchor is found, remove only the authority line.
    trimmed = lines[:i] + lines[i + 1 :]
    while trimmed and not trimmed[0].strip():
        trimmed = trimmed[1:]
    return trimmed


def clean_executive_order_text(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""

    start = _find_body_start(lines)
    body_lines = lines[start:]
    body_lines = _trim_leading_authority_preamble(body_lines)
    body_lines = _trim_footer(body_lines)

    if not body_lines:
        return "\n".join(lines).strip()

    return "\n".join(body_lines).strip()


def process_file(src: Path, dst: Path) -> tuple[bool, bool]:
    raw = src.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_executive_order_text(raw)

    if not cleaned:
        cleaned = raw.strip()

    changed = cleaned != raw.strip()
    wrote = False

    if not dst.parent.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)

    dst.write_text(cleaned + "\n", encoding="utf-8")
    wrote = True

    return changed, wrote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove EO header/footer metadata and write NLP-ready text files "
            "while preserving body text."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("executive_orders"),
        help="Root directory containing EO .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("executive_orders_cleaned"),
        help="Root directory for cleaned EO .txt files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files instead of writing to --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Default behavior is to skip them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = input_dir if args.in_place else args.output_dir
    if not args.in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {input_dir}")

    written = 0
    changed = 0
    skipped = 0

    for src in files:
        rel = src.relative_to(input_dir)
        dst = src if args.in_place else output_dir / rel

        if dst.exists() and dst != src and not args.overwrite:
            skipped += 1
            continue

        did_change, did_write = process_file(src, dst)
        if did_write:
            written += 1
        if did_change:
            changed += 1

    print(f"Input files found : {len(files)}")
    print(f"Files written     : {written}")
    print(f"Files changed     : {changed}")
    print(f"Files skipped     : {skipped}")
    print(f"Output directory  : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
