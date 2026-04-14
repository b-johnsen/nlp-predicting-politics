#!/usr/bin/env python3
"""Build a labeled train/test split for executive order text classification.

The script selects a subset of presidents from two in-script lists
(`DEMOCRAT_PRESIDENT_DIRS` and `REPUBLICAN_PRESIDENT_DIRS`), cleans each
document to body text using the same cleaner as `clean_executive_orders.py`,
and writes an NLP-friendly directory layout:

    <output_dir>/train/<label>/<president_dir>/<file>.txt
    <output_dir>/test/<label>/<president_dir>/<file>.txt

It also writes:
    - <output_dir>/manifest.csv
    - <output_dir>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from clean_executive_orders import clean_executive_order_text

INPUT_DIR_DEFAULT = Path("all_executive_orders_txt_clean")
OUTPUT_DIR_DEFAULT = Path("eo_labeled_split")
TRAIN_RATIO_DEFAULT = 0.8
SEED_DEFAULT = 42
BALANCE_LABELS_DEFAULT = True

TRAILING_SIGNATURE_RE = re.compile(r"^[A-Z][A-Z .,'-]{2,}$")
LEADING_EXEC_ORDER_TITLE_RE = re.compile(r"^\s*Executive\s+Order\b", re.IGNORECASE)
LEADING_DATE_RE = re.compile(
    r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},\s+\d{4}\s*$",
    re.IGNORECASE,
)
LEADING_DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Z\-'. ]+,\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\.?\s*$"
)
TRAILING_SOURCE_RE = re.compile(r"^\s*SOURCE\s*:\s*", re.IGNORECASE)

# Edit these lists to control which president directories are included.
DEMOCRAT_PRESIDENT_DIRS = [
    "Andrew_Jackson",
    "Martin_van_Buren",
    "James_K_Polk",
    "Franklin_Pierce",
    "James_Buchanan",
    "Andrew_Johnson",
    "Grover_Cleveland",
    "Woodrow_Wilson",
    "Franklin_D_Roosevelt",
    "Harry_S_Truman",
    "John_F_Kennedy",
    "Lyndon_B_Johnson",
    "Jimmy_Carter",
    "William_J_Clinton",
    "Barack_Obama",
    "Joseph_R_Biden_Jr",
]

REPUBLICAN_PRESIDENT_DIRS = [
    "Abraham_Lincoln",
    "Ulysses_S_Grant",
    "Rutherford_B_Hayes",
    "James_A_Garfield",
    "Chester_A_Arthur",
    "Benjamin_Harrison",
    "William_McKinley",
    "Theodore_Roosevelt",
    "William_Howard_Taft",
    "Warren_G_Harding",
    "Calvin_Coolidge",
    "Herbert_Hoover",
    "Dwight_D_Eisenhower",
    "Richard_Nixon",
    "Gerald_R_Ford",
    "Ronald_Reagan",
    "George_Bush",
    "George_W_Bush",
    "Donald_J_Trump_(1st_Term)",
    "Donald_J_Trump_(2nd_Term)",
]


@dataclass
class DocumentRecord:
    source_path: Path
    president_dir: str
    label: str
    cleaned_text: str
    char_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a labeled train/test split from executive order text files "
            "using democrat/republican president directory lists."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR_DEFAULT,
        help="Root directory containing president subdirectories with .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output root for train/test labeled split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO_DEFAULT,
        help="Train split ratio in (0, 1). Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--balance-labels",
        dest="balance_labels",
        action="store_true",
        default=BALANCE_LABELS_DEFAULT,
        help=(
            "If enabled, downsample to a 50/50 democrat/republican dataset "
            "before train/test split (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-balance-labels",
        dest="balance_labels",
        action="store_false",
        help="Disable 50/50 label balancing and keep all selected records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    return parser.parse_args()


def build_label_map() -> dict[str, str]:
    overlap = set(DEMOCRAT_PRESIDENT_DIRS) & set(REPUBLICAN_PRESIDENT_DIRS)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise SystemExit(f"President directories overlap party lists: {joined}")

    label_map: dict[str, str] = {}
    for president_dir in DEMOCRAT_PRESIDENT_DIRS:
        label_map[president_dir] = "democrat"
    for president_dir in REPUBLICAN_PRESIDENT_DIRS:
        label_map[president_dir] = "republican"
    return label_map


def validate_inputs(
    input_dir: Path, output_dir: Path, overwrite: bool, label_map: dict[str, str]
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    missing_dirs = [
        president_dir
        for president_dir in sorted(label_map)
        if not (input_dir / president_dir).is_dir()
    ]
    if missing_dirs:
        joined = "\n".join(f"  - {name}" for name in missing_dirs)
        raise SystemExit(
            "Listed president directories are missing from input dir:\n" f"{joined}"
        )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Use --overwrite to replace it."
        )


def collect_records(
    input_dir: Path, label_map: dict[str, str]
) -> tuple[list[DocumentRecord], int]:
    records: list[DocumentRecord] = []
    empty_count = 0

    for president_dir, label in sorted(label_map.items()):
        president_root = input_dir / president_dir
        for source_path in sorted(president_root.rglob("*.txt")):
            raw_text = source_path.read_text(encoding="utf-8", errors="replace")
            cleaned_text = clean_executive_order_text(raw_text).strip()
            cleaned_text = trim_leading_metadata(cleaned_text, president_dir)
            cleaned_text = trim_trailing_signature(cleaned_text, president_dir)
            if not cleaned_text:
                # Secondary pass: if the shared cleaner collapses to empty, run
                # header/footer trimming directly against raw text before any
                # final raw fallback.
                cleaned_text = trim_leading_metadata(raw_text.strip(), president_dir)
                cleaned_text = trim_trailing_signature(cleaned_text, president_dir)
            if not cleaned_text:
                cleaned_text = raw_text.strip()
            if not cleaned_text:
                empty_count += 1
                continue

            records.append(
                DocumentRecord(
                    source_path=source_path,
                    president_dir=president_dir,
                    label=label,
                    cleaned_text=cleaned_text,
                    char_count=len(cleaned_text),
                )
            )

    return records, empty_count


def trim_trailing_signature(text: str, president_dir: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    president_variants = _president_name_variants(president_dir)

    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    # Removes trailing metadata/signature blocks, including:
    # - SOURCE notes
    # - Signature name lines (e.g., "GROVER CLEVELAND.")
    # - Optional short title lines that often follow signatures
    while lines:
        candidate = lines[-1].strip()
        if not candidate:
            lines.pop()
            continue

        is_signature_name = (
            TRAILING_SIGNATURE_RE.fullmatch(candidate.rstrip("."))
            and len(candidate) <= 80
            and len(candidate.split()) >= 2
        )
        is_president_name = (
            _normalize_header_text(candidate.rstrip(".")) in president_variants
        )
        is_source_line = bool(TRAILING_SOURCE_RE.match(candidate))
        is_role_line = bool(
            re.fullmatch(
                r"(?i)^(Secretary(\s+of\s+[A-Za-z .'-]+)?|Acting\s+Secretary(\s+of\s+[A-Za-z .'-]+)?|Attorney\s+General|Director|Chairman)\.?$",
                candidate,
            )
        )

        if is_source_line or is_signature_name or is_president_name or is_role_line:
            lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            continue

        break

    return "\n".join(lines).strip()


def _normalize_header_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value.lower()).split())


def _president_name_variants(president_dir: str) -> set[str]:
    display = president_dir.replace("_", " ")
    display = re.sub(r"\s*\(.*?\)\s*", "", display).strip()
    tokens = display.split()

    variants: set[str] = {display}
    if tokens:
        title_tokens = []
        for token in tokens:
            if len(token) == 1 and token.isalpha():
                title_tokens.append(f"{token.upper()}.")
            else:
                title_tokens.append(token[0].upper() + token[1:] if token else token)
        variants.add(" ".join(title_tokens))

    normalized = {_normalize_header_text(item) for item in variants if item.strip()}
    return {item for item in normalized if item}


def trim_leading_metadata(text: str, president_dir: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    president_variants = _president_name_variants(president_dir)
    idx = 0

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        normalized_line = _normalize_header_text(line)
        is_president_name = normalized_line in president_variants

        if (
            LEADING_EXEC_ORDER_TITLE_RE.match(line)
            or LEADING_DATE_RE.match(line)
            or LEADING_DATELINE_RE.match(line)
            or is_president_name
        ):
            idx += 1
            continue

        break

    trimmed = lines[idx:]
    while trimmed and not trimmed[0].strip():
        trimmed = trimmed[1:]

    return "\n".join(trimmed).strip()


def stratified_split(
    records: list[DocumentRecord],
    train_ratio: float,
    seed: int,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    rng = random.Random(seed)
    train_records: list[DocumentRecord] = []
    test_records: list[DocumentRecord] = []

    for label, items in sorted(grouped.items()):
        rng.shuffle(items)
        n_items = len(items)
        if n_items == 0:
            continue

        if n_items == 1:
            train_size = 1
        else:
            train_size = int(n_items * train_ratio)
            train_size = max(1, min(train_size, n_items - 1))

        train_records.extend(items[:train_size])
        test_records.extend(items[train_size:])

    return train_records, test_records


def balance_records_by_label(
    records: list[DocumentRecord],
    seed: int,
) -> tuple[list[DocumentRecord], dict[str, int]]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    expected_labels = {"democrat", "republican"}
    missing = sorted(label for label in expected_labels if not grouped.get(label))
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Cannot create 50/50 label balance because these labels are missing: "
            f"{joined}"
        )

    target_count = min(len(grouped["democrat"]), len(grouped["republican"]))
    if target_count == 0:
        raise SystemExit("Cannot create 50/50 label balance with zero-count labels.")

    rng = random.Random(seed)
    balanced_records: list[DocumentRecord] = []
    dropped_counts: dict[str, int] = {}

    for label in sorted(expected_labels):
        items = list(grouped[label])
        rng.shuffle(items)
        balanced_records.extend(items[:target_count])
        dropped_counts[label] = len(items) - target_count

    return balanced_records, dropped_counts


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_split_files(
    records: list[DocumentRecord],
    split_name: str,
    input_dir: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []

    for record in records:
        relative_source = record.source_path.relative_to(input_dir)
        destination = (
            output_dir
            / split_name
            / record.label
            / record.president_dir
            / record.source_path.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(record.cleaned_text + "\n", encoding="utf-8")

        manifest_rows.append(
            {
                "split": split_name,
                "label": record.label,
                "president_dir": record.president_dir,
                "filename": record.source_path.name,
                "source_path": relative_source.as_posix(),
                "output_path": destination.relative_to(output_dir).as_posix(),
                "char_count": str(record.char_count),
            }
        )

    return manifest_rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "split",
        "label",
        "president_dir",
        "filename",
        "source_path",
        "output_path",
        "char_count",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    train_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    selected_count: int,
    selected_count_before_balance: int,
    skipped_empty: int,
    train_ratio: float,
    seed: int,
    balance_labels: bool,
    dropped_by_balance: dict[str, int],
) -> None:
    split_counts = {
        "train": len(train_records),
        "test": len(test_records),
    }

    by_label = {
        "train": dict(Counter(record.label for record in train_records)),
        "test": dict(Counter(record.label for record in test_records)),
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_ratio": train_ratio,
        "seed": seed,
        "balance_labels": balance_labels,
        "selected_files_before_balance": selected_count_before_balance,
        "selected_files": selected_count,
        "dropped_by_balance": dropped_by_balance,
        "skipped_empty_files": skipped_empty,
        "split_counts": split_counts,
        "split_counts_by_label": by_label,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(
    input_dir: Path,
    output_dir: Path,
    selected_count: int,
    selected_count_before_balance: int,
    skipped_empty: int,
    train_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    balance_labels: bool,
    dropped_by_balance: dict[str, int],
) -> None:
    label_counts_total = Counter(
        record.label for record in train_records + test_records
    )
    train_counts = Counter(record.label for record in train_records)
    test_counts = Counter(record.label for record in test_records)

    print(f"Input directory         : {input_dir.resolve()}")
    print(f"Output directory        : {output_dir.resolve()}")
    print(f"Balance labels (50/50)  : {balance_labels}")
    if balance_labels:
        print(f"Selected before balance : {selected_count_before_balance}")
    print(f"Selected files          : {selected_count}")
    if balance_labels:
        dropped_total = sum(dropped_by_balance.values())
        print(f"Dropped by balancing    : {dropped_total}")
    print(f"Skipped empty files     : {skipped_empty}")
    print(f"Train files             : {len(train_records)}")
    print(f"Test files              : {len(test_records)}")
    print("Label totals            :")
    for label in sorted(label_counts_total):
        print(f"  - {label}: {label_counts_total[label]}")
    print("Train/Test by label     :")
    for label in sorted(label_counts_total):
        print(f"  - {label}: train={train_counts[label]} test={test_counts[label]}")


def main() -> None:
    args = parse_args()

    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be strictly between 0 and 1.")

    label_map = build_label_map()
    validate_inputs(args.input_dir, args.output_dir, args.overwrite, label_map)

    records, skipped_empty = collect_records(args.input_dir, label_map)
    if not records:
        raise SystemExit(
            "No eligible non-empty .txt files were found in listed directories."
        )

    selected_count_before_balance = len(records)
    dropped_by_balance = {"democrat": 0, "republican": 0}
    if args.balance_labels:
        records, dropped_by_balance = balance_records_by_label(records, args.seed)

    train_records, test_records = stratified_split(records, args.train_ratio, args.seed)

    prepare_output_dir(args.output_dir, args.overwrite)
    manifest_rows: list[dict[str, str]] = []
    manifest_rows.extend(
        write_split_files(train_records, "train", args.input_dir, args.output_dir)
    )
    manifest_rows.extend(
        write_split_files(test_records, "test", args.input_dir, args.output_dir)
    )

    write_manifest(args.output_dir, manifest_rows)
    write_summary(
        args.output_dir,
        train_records,
        test_records,
        selected_count=len(records),
        selected_count_before_balance=selected_count_before_balance,
        skipped_empty=skipped_empty,
        train_ratio=args.train_ratio,
        seed=args.seed,
        balance_labels=args.balance_labels,
        dropped_by_balance=dropped_by_balance,
    )
    print_summary(
        args.input_dir,
        args.output_dir,
        selected_count=len(records),
        selected_count_before_balance=selected_count_before_balance,
        skipped_empty=skipped_empty,
        train_records=train_records,
        test_records=test_records,
        balance_labels=args.balance_labels,
        dropped_by_balance=dropped_by_balance,
    )


if __name__ == "__main__":
    main()
