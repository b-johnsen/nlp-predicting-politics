#!/usr/bin/env python3
"""Plot executive order counts by year and export summary statistics.

This script scans text files under eo_data/all_executive_orders_txt_clean,
extracts years from filenames, and aggregates counts and text statistics
by year.

Outputs:
    - eo_count_by_year.png (line/bar chart)
    - eo_year_summary.csv (per-year count + text statistics)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR_DEFAULT = REPO_ROOT / "eo_data" / "all_executive_orders_txt_clean"
OUTPUT_DIR_DEFAULT = REPO_ROOT / "eo_data" / "analysis"

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

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 38
TICK_FONT_SIZE = 36
YEAR_TICK_FONT_SIZE = 36
COUNT_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 22

TOKEN_RE = re.compile(r"\b[\w']+\b")
YEAR_RE = re.compile(r", (\d{4})__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a year-based EO count graph and per-year text stats table."
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
        help="Directory where output plot and tables are written.",
    )
    parser.add_argument(
        "--plot-name",
        default="eo_count_by_year.png",
        help="Output filename for the bar chart image.",
    )
    parser.add_argument(
        "--table-name",
        default="eo_year_summary.csv",
        help="Output filename for the per-year summary CSV.",
    )
    return parser.parse_args()


def extract_year(filename: str) -> int | None:
    """Extract year from filename using the format ', YYYY__'."""
    match = YEAR_RE.search(filename)
    if match:
        return int(match.group(1))
    return None


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    sentences = [part for part in parts if re.search(r"\w", part)]
    if sentences:
        return len(sentences)
    return 1


def build_party_map() -> dict[str, str]:
    overlap = set(DEMOCRAT_PRESIDENT_DIRS) & set(REPUBLICAN_PRESIDENT_DIRS)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise SystemExit(f"Overlapping presidents in party lists: {joined}")

    party_map: dict[str, str] = {}
    for president in DEMOCRAT_PRESIDENT_DIRS:
        party_map[president] = "Democrat"
    for president in REPUBLICAN_PRESIDENT_DIRS:
        party_map[president] = "Republican"
    return party_map


def collect_year_stats(input_dir: Path, party_map: dict[str, str]) -> pd.DataFrame:
    """Collect per-year statistics from all EO files."""
    year_data: dict[int, dict[str, object]] = {}

    for president_dir in input_dir.iterdir():
        if not president_dir.is_dir():
            continue

        president_name = president_dir.name
        party = party_map.get(president_name, "Unknown")

        txt_files = sorted(president_dir.rglob("*.txt"))

        for txt_path in txt_files:
            year = extract_year(txt_path.name)
            if year is None:
                print(f"Warning: could not extract year from {txt_path.name}")
                continue

            if year not in year_data:
                year_data[year] = {
                    "token_counts": [],
                    "sentence_counts": [],
                    "char_counts": [],
                    "eo_count": 0,
                }

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            year_data[year]["token_counts"].append(count_tokens(text))
            year_data[year]["sentence_counts"].append(count_sentences(text))
            year_data[year]["char_counts"].append(len(text))
            year_data[year]["eo_count"] += 1

    rows: list[dict[str, object]] = []
    for year in sorted(year_data.keys()):
        data = year_data[year]
        token_counts = data["token_counts"]
        sentence_counts = data["sentence_counts"]
        char_counts = data["char_counts"]

        eo_count = data["eo_count"]
        total_tokens = int(sum(token_counts))
        total_sentences = int(sum(sentence_counts))
        total_chars = int(sum(char_counts))

        avg_tokens = total_tokens / eo_count if eo_count else 0.0
        avg_sentences = total_sentences / eo_count if eo_count else 0.0
        avg_chars = total_chars / eo_count if eo_count else 0.0
        avg_words_per_sentence = (
            total_tokens / total_sentences if total_sentences else 0.0
        )

        rows.append(
            {
                "year": year,
                "eo_count": eo_count,
                "total_tokens": total_tokens,
                "avg_tokens_per_eo": avg_tokens,
                "median_tokens_per_eo": (
                    float(pd.Series(token_counts).median()) if token_counts else 0.0
                ),
                "total_sentences": total_sentences,
                "avg_sentences_per_eo": avg_sentences,
                "avg_words_per_sentence": avg_words_per_sentence,
                "total_characters": total_chars,
                "avg_characters_per_eo": avg_chars,
            }
        )

    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        raise SystemExit("No data found to plot.")

    fig_width = max(24, 0.25 * len(df))
    plt.figure(figsize=(fig_width, 12))
    plt.plot(
        df["year"],
        df["eo_count"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#2C7FB8",
    )
    plt.bar(df["year"], df["eo_count"], alpha=0.3, color="#2C7FB8", width=0.6)

    plt.xlabel("Year", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Number of Executive Orders (.txt files)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Executive Order Count by Year", fontsize=TITLE_FONT_SIZE)

    # Set year ticks every 5 years for better readability
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    year_ticks = range(min_year, max_year + 1, 5)
    plt.xticks(year_ticks, rotation=45, ha="right", fontsize=YEAR_TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    party_map = build_party_map()
    summary_df = collect_year_stats(args.input_dir, party_map)

    if summary_df.empty:
        raise SystemExit("No year data collected.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    table_path = args.output_dir / f"{Path(args.table_name).stem}_{timestamp}.csv"
    plot_path = args.output_dir / f"{Path(args.plot_name).stem}_{timestamp}.png"

    # Keep table values readable for quick inspection.
    rounded = summary_df.copy()
    for col in [
        "avg_tokens_per_eo",
        "median_tokens_per_eo",
        "avg_sentences_per_eo",
        "avg_words_per_sentence",
        "avg_characters_per_eo",
    ]:
        rounded[col] = rounded[col].round(2)

    rounded.to_csv(table_path, index=False)
    save_plot(summary_df, plot_path)

    print(f"Saved plot: {plot_path}")
    print(f"Saved table: {table_path}")
    print(f"Years included: {len(summary_df)}")
    print(f"Year range: {summary_df['year'].min()} - {summary_df['year'].max()}")
    print(f"Total EO files counted: {int(summary_df['eo_count'].sum())}")


if __name__ == "__main__":
    main()
