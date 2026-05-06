#!/usr/bin/env python3
"""Build a train/test split for bill .txt files produced by build_congress_bills_txt.py."""

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

INPUT_DIR_DEFAULT = Path("congress_bills_txt_113_119")
OUTPUT_DIR_DEFAULT = Path("bills_labeled_split_113_119")
TRAIN_RATIO_DEFAULT = 0.8
SEED_DEFAULT = 42
BALANCE_LABELS_DEFAULT = True
BALANCE_STRATEGY_DEFAULT = "upsample"

BODY_START_PATTERNS = [
    re.compile(r"^Be it enacted\b", re.IGNORECASE),
    re.compile(r"^Resolved[, ]", re.IGNORECASE),
    re.compile(r"^Whereas[, ]", re.IGNORECASE),
    re.compile(r"^SECTION\s+1\.", re.IGNORECASE),
    re.compile(r"^SEC\.\s*1\.", re.IGNORECASE),
    re.compile(r"^\d+\.$"),
]

LEADING_METADATA_PATTERNS = [
    re.compile(r"^\d{2,3}\s+[A-Z]{1,10}\s+\d+\s+[A-Z0-9]{1,8}:"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2,3}(st|nd|rd|th)\s+CONGRESS$", re.IGNORECASE),
    re.compile(r"^\d+(st|nd|rd|th)\s+Session$", re.IGNORECASE),
    re.compile(r"^\d+[a-z]{2}\s+Session$", re.IGNORECASE),
    re.compile(r"^[IVXLCM]+$", re.IGNORECASE),
    re.compile(r"^IN THE (HOUSE|SENATE) OF (THE )?REPRESENTATIVES$", re.IGNORECASE),
    re.compile(r"^IN THE SENATE OF THE UNITED STATES$", re.IGNORECASE),
    re.compile(
        r"^(H|S)\.\s*(R\.|J\.\s*RES\.|CON\.\s*RES\.|RES\.)\s*\d+", re.IGNORECASE
    ),
    re.compile(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$",
        re.IGNORECASE,
    ),
    re.compile(r"^(U\.S\.|United States) (House|Senate)", re.IGNORECASE),
    re.compile(r"^text/xml$", re.IGNORECASE),
    re.compile(r"^EN$", re.IGNORECASE),
    re.compile(r"^Pursuant to Title 17 Section 105", re.IGNORECASE),
    re.compile(
        r"^(A BILL|AN ACT|JOINT RESOLUTION|CONCURRENT RESOLUTION|RESOLUTION)$",
        re.IGNORECASE,
    ),
    re.compile(r"^for (himself|herself|themselves|herself and).*", re.IGNORECASE),
    re.compile(r"^\(?for herself,?\)?$", re.IGNORECASE),
    re.compile(r"^\(?for himself,?\)?$", re.IGNORECASE),
    re.compile(r"^\(?for themselves,?\)?$", re.IGNORECASE),
    re.compile(r"^(Mr|Mrs|Ms|Miss|Sen)\.\s+[A-Z].*"),
    re.compile(r"^Committee on ", re.IGNORECASE),
    re.compile(r"^submitted the following", re.IGNORECASE),
    re.compile(r"^referred to the", re.IGNORECASE),
    re.compile(r"^[,;:.()\-]+$"),
]


@dataclass
class BillRecord:
    source_path: Path
    bill_id: str
    congress: str
    bill_type: str
    label: str
    cleaned_text: str
    char_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/test split from congress bill txt dataset, "
            "stratified by party label."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR_DEFAULT,
        help="Input dataset root containing manifest.csv and label subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for train/test split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO_DEFAULT,
        help="Train split ratio in (0, 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--include-independent",
        action="store_true",
        help="Include independent_or_unknown label in split. Default excludes it.",
    )
    parser.add_argument(
        "--balance-labels",
        dest="balance_labels",
        action="store_true",
        default=BALANCE_LABELS_DEFAULT,
        help=(
            "Balance labels before split. Default: enabled. "
            "For bigger balanced sets, default strategy is upsample."
        ),
    )
    parser.add_argument(
        "--no-balance-labels",
        dest="balance_labels",
        action="store_false",
        help="Disable label balancing and keep natural label counts.",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=["upsample", "downsample"],
        default=BALANCE_STRATEGY_DEFAULT,
        help=(
            "Balancing strategy: upsample duplicates minority examples to grow dataset, "
            "downsample trims majority labels."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output directory if it exists.",
    )
    return parser.parse_args()


def normalize_line(line: str) -> str:
    return " ".join(line.replace("\u2019", "'").replace("\u2018", "'").split())


def find_body_start(lines: list[str]) -> int:
    search_limit = min(len(lines), 400)

    for idx in range(search_limit):
        line = lines[idx]
        if any(pattern.match(line) for pattern in BODY_START_PATTERNS):
            return idx

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue
        if any(pattern.match(line) for pattern in LEADING_METADATA_PATTERNS):
            idx += 1
            continue
        break
    return min(idx, len(lines))


def clean_bill_text(text: str) -> str:
    normalized_lines = [normalize_line(line) for line in text.splitlines()]
    if not normalized_lines:
        return ""

    start = find_body_start(normalized_lines)
    body_lines = normalized_lines[start:]

    while body_lines and not body_lines[0]:
        body_lines.pop(0)
    while body_lines and not body_lines[-1]:
        body_lines.pop()

    cleaned_lines: list[str] = []
    prev_blank = False
    for line in body_lines:
        if not line:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
            continue
        cleaned_lines.append(line)
        prev_blank = False

    return "\n".join(cleaned_lines).strip()


def validate_inputs(input_dir: Path, output_dir: Path, overwrite: bool) -> Path:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    manifest = input_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Use --overwrite to replace it."
        )
    return manifest


def load_records(
    manifest_path: Path, input_dir: Path, include_independent: bool
) -> list[BillRecord]:
    allowed_labels = {"democrat", "republican"}
    if include_independent:
        allowed_labels.add("independent_or_unknown")

    records: list[BillRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "written":
                continue

            label = (row.get("label") or "").strip().lower()
            if label not in allowed_labels:
                continue

            output_path = (row.get("output_path") or "").strip()
            if not output_path:
                continue

            source_path = input_dir / output_path
            if not source_path.exists():
                continue

            raw_text = source_path.read_text(encoding="utf-8", errors="replace")
            cleaned_text = clean_bill_text(raw_text)
            if not cleaned_text:
                continue

            records.append(
                BillRecord(
                    source_path=source_path,
                    bill_id=(row.get("bill_id") or "").strip(),
                    congress=(row.get("congress") or "").strip(),
                    bill_type=(row.get("bill_type") or "").strip(),
                    label=label,
                    cleaned_text=cleaned_text,
                    char_count=len(cleaned_text),
                )
            )

    return records


def balance_records_by_label(
    records: list[BillRecord],
    seed: int,
    strategy: str,
) -> tuple[list[BillRecord], dict[str, int]]:
    grouped: dict[str, list[BillRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    if len(grouped) < 2:
        raise SystemExit(
            "Balancing requires at least two labels with records. "
            "Use --no-balance-labels to disable balancing."
        )

    counts = {label: len(items) for label, items in grouped.items()}
    target = max(counts.values()) if strategy == "upsample" else min(counts.values())
    if target <= 0:
        raise SystemExit("Cannot balance labels with zero-count classes.")

    rng = random.Random(seed)
    balanced: list[BillRecord] = []
    adjustments: dict[str, int] = {}

    for label, items in sorted(grouped.items()):
        bucket = list(items)
        rng.shuffle(bucket)

        if len(bucket) == target:
            chosen = bucket
            adjustments[label] = 0
        elif len(bucket) > target:
            chosen = bucket[:target]
            adjustments[label] = target - len(bucket)
        else:
            chosen = list(bucket)
            needed = target - len(bucket)
            for _ in range(needed):
                chosen.append(bucket[rng.randrange(len(bucket))])
            adjustments[label] = needed

        balanced.extend(chosen)

    rng.shuffle(balanced)
    return balanced, adjustments


def stratified_split(
    records: list[BillRecord],
    train_ratio: float,
    seed: int,
) -> tuple[list[BillRecord], list[BillRecord]]:
    grouped: dict[str, list[BillRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    rng = random.Random(seed)
    train_records: list[BillRecord] = []
    test_records: list[BillRecord] = []

    for label, items in sorted(grouped.items()):
        rng.shuffle(items)
        n_items = len(items)

        if n_items == 1:
            train_size = 1
        else:
            train_size = int(n_items * train_ratio)
            train_size = max(1, min(train_size, n_items - 1))

        train_records.extend(items[:train_size])
        test_records.extend(items[train_size:])

    return train_records, test_records


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_split_files(
    records: list[BillRecord],
    split_name: str,
    input_dir: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    file_name_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    for record in records:
        relative_source = record.source_path.relative_to(input_dir)
        original_file_name = record.source_path.name

        key = (record.label, record.congress, record.bill_type, original_file_name)
        duplicate_index = file_name_counts[key]
        file_name_counts[key] += 1

        if duplicate_index == 0:
            file_name = original_file_name
        else:
            source_name = Path(original_file_name)
            file_name = f"{source_name.stem}__dup{duplicate_index}{source_name.suffix}"

        destination = (
            output_dir
            / split_name
            / record.label
            / record.congress
            / record.bill_type
            / file_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(record.cleaned_text + "\n", encoding="utf-8")

        rows.append(
            {
                "split": split_name,
                "label": record.label,
                "congress": record.congress,
                "bill_type": record.bill_type,
                "bill_id": record.bill_id,
                "filename": file_name,
                "source_path": relative_source.as_posix(),
                "output_path": destination.relative_to(output_dir).as_posix(),
                "char_count": str(record.char_count),
            }
        )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    path = output_dir / "manifest.csv"
    fieldnames = [
        "split",
        "label",
        "congress",
        "bill_type",
        "bill_id",
        "filename",
        "source_path",
        "output_path",
        "char_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    train_records: list[BillRecord],
    test_records: list[BillRecord],
    train_ratio: float,
    seed: int,
    include_independent: bool,
    balance_labels: bool,
    balance_strategy: str,
    balance_adjustments: dict[str, int],
    selected_before_balance: int,
    selected_total: int,
) -> None:
    by_label = {
        "train": dict(Counter(record.label for record in train_records)),
        "test": dict(Counter(record.label for record in test_records)),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_ratio": train_ratio,
        "seed": seed,
        "include_independent": include_independent,
        "balance_labels": balance_labels,
        "balance_strategy": balance_strategy,
        "selected_files_before_balance": selected_before_balance,
        "selected_files": selected_total,
        "balance_adjustments": balance_adjustments,
        "split_counts": {
            "train": len(train_records),
            "test": len(test_records),
        },
        "split_counts_by_label": by_label,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(
    input_dir: Path,
    output_dir: Path,
    train_records: list[BillRecord],
    test_records: list[BillRecord],
    balance_labels: bool,
    balance_strategy: str,
    selected_before_balance: int,
    selected_total: int,
    balance_adjustments: dict[str, int],
) -> None:
    total_counts = Counter(record.label for record in train_records + test_records)
    train_counts = Counter(record.label for record in train_records)
    test_counts = Counter(record.label for record in test_records)

    print(f"Input directory  : {input_dir.resolve()}")
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Balance labels   : {balance_labels} ({balance_strategy})")
    if balance_labels:
        print(f"Before balance   : {selected_before_balance}")
    print(f"Selected files   : {selected_total}")
    if balance_labels:
        print("Balance changes  :")
        for label in sorted(balance_adjustments):
            delta = balance_adjustments[label]
            action = "added" if delta >= 0 else "dropped"
            print(f"  - {label}: {action} {abs(delta)}")
    print(f"Train files      : {len(train_records)}")
    print(f"Test files       : {len(test_records)}")
    print("Label totals     :")
    for label in sorted(total_counts):
        print(f"  - {label}: {total_counts[label]}")
    print("Train/Test labels:")
    for label in sorted(total_counts):
        print(f"  - {label}: train={train_counts[label]} test={test_counts[label]}")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be strictly between 0 and 1.")

    manifest = validate_inputs(args.input_dir, args.output_dir, args.overwrite)
    records = load_records(manifest, args.input_dir, args.include_independent)
    if not records:
        raise SystemExit("No eligible records found for requested labels.")

    selected_before_balance = len(records)
    balance_adjustments: dict[str, int] = {}
    if args.balance_labels:
        records, balance_adjustments = balance_records_by_label(
            records, args.seed, args.balance_strategy
        )

    train_records, test_records = stratified_split(records, args.train_ratio, args.seed)

    prepare_output_dir(args.output_dir, args.overwrite)
    rows: list[dict[str, str]] = []
    rows.extend(
        write_split_files(train_records, "train", args.input_dir, args.output_dir)
    )
    rows.extend(
        write_split_files(test_records, "test", args.input_dir, args.output_dir)
    )

    write_manifest(args.output_dir, rows)
    write_summary(
        args.output_dir,
        train_records,
        test_records,
        train_ratio=args.train_ratio,
        seed=args.seed,
        include_independent=args.include_independent,
        balance_labels=args.balance_labels,
        balance_strategy=args.balance_strategy,
        balance_adjustments=balance_adjustments,
        selected_before_balance=selected_before_balance,
        selected_total=len(records),
    )
    print_summary(
        args.input_dir,
        args.output_dir,
        train_records,
        test_records,
        balance_labels=args.balance_labels,
        balance_strategy=args.balance_strategy,
        selected_before_balance=selected_before_balance,
        selected_total=len(records),
        balance_adjustments=balance_adjustments,
    )


if __name__ == "__main__":
    main()
