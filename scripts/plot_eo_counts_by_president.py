#!/usr/bin/env python3
"""Plot executive order counts by president and export summary statistics.

This script scans text files under eo_data/all_executive_orders_txt_clean,
restricted to the president directories listed in
DEMOCRAT_PRESIDENT_DIRS and REPUBLICAN_PRESIDENT_DIRS.

Outputs:
    - eo_count_by_president.png (party-colored bar chart)
    - eo_president_summary.csv (per-president count + text statistics)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
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

CHRONOLOGICAL_PRESIDENT_DIRS = [
    "Andrew_Jackson",
    "Martin_van_Buren",
    "James_K_Polk",
    "Franklin_Pierce",
    "James_Buchanan",
    "Abraham_Lincoln",
    "Andrew_Johnson",
    "Ulysses_S_Grant",
    "Rutherford_B_Hayes",
    "James_A_Garfield",
    "Chester_A_Arthur",
    "Grover_Cleveland",
    "Benjamin_Harrison",
    "William_McKinley",
    "Theodore_Roosevelt",
    "William_Howard_Taft",
    "Woodrow_Wilson",
    "Warren_G_Harding",
    "Calvin_Coolidge",
    "Herbert_Hoover",
    "Franklin_D_Roosevelt",
    "Harry_S_Truman",
    "Dwight_D_Eisenhower",
    "John_F_Kennedy",
    "Lyndon_B_Johnson",
    "Richard_Nixon",
    "Gerald_R_Ford",
    "Jimmy_Carter",
    "Ronald_Reagan",
    "George_Bush",
    "William_J_Clinton",
    "George_W_Bush",
    "Barack_Obama",
    "Donald_J_Trump_(1st_Term)",
    "Joseph_R_Biden_Jr",
    "Donald_J_Trump_(2nd_Term)",
]

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 13
COUNT_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 13

TOKEN_RE = re.compile(r"\b[\w']+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a party-colored EO count graph and per-president text stats table."
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
        default="eo_count_by_president.png",
        help="Output filename for the bar chart image.",
    )
    parser.add_argument(
        "--table-name",
        default="eo_president_summary.csv",
        help="Output filename for the per-president summary CSV.",
    )
    return parser.parse_args()


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


def president_display_name(president_dir: str) -> str:
    return president_dir.replace("_", " ")


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


def collect_president_stats(input_dir: Path, party_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_order = CHRONOLOGICAL_PRESIDENT_DIRS

    for president_dir in selected_order:
        president_path = input_dir / president_dir
        if not president_path.is_dir():
            print(f"Warning: missing directory, skipping: {president_path}")
            continue

        txt_files = sorted(president_path.rglob("*.txt"))
        token_counts: list[int] = []
        sentence_counts: list[int] = []
        char_counts: list[int] = []

        for txt_path in txt_files:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            token_counts.append(count_tokens(text))
            sentence_counts.append(count_sentences(text))
            char_counts.append(len(text))

        eo_count = len(txt_files)
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
                "president_dir": president_dir,
                "president": president_display_name(president_dir),
                "party": party_map[president_dir],
                "eo_count": eo_count,
                "total_tokens": total_tokens,
                "avg_tokens_per_eo": avg_tokens,
                "median_tokens_per_eo": float(pd.Series(token_counts).median())
                if token_counts
                else 0.0,
                "total_sentences": total_sentences,
                "avg_sentences_per_eo": avg_sentences,
                "avg_words_per_sentence": avg_words_per_sentence,
                "total_characters": total_chars,
                "avg_characters_per_eo": avg_chars,
            }
        )

    return pd.DataFrame(rows)


def sort_presidents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    default_order = CHRONOLOGICAL_PRESIDENT_DIRS
    default_rank = {name: idx for idx, name in enumerate(default_order)}

    df = df.copy()
    df["default_rank"] = df["president_dir"].map(default_rank).fillna(10_000)
    df = df.sort_values("default_rank", kind="stable")
    df = df.drop(columns=["default_rank"])
    return df.reset_index(drop=True)


def save_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        raise SystemExit("No data found to plot.")

    party_colors = {"Democrat": "#2C7FB8", "Republican": "#D7301F"}
    bar_colors = [party_colors.get(party, "#777777") for party in df["party"]]

    fig_height = max(10, 0.45 * len(df))
    plt.figure(figsize=(15, fig_height))
    bars = plt.barh(df["president"], df["eo_count"], color=bar_colors)
    plt.xlabel(
        "Number of Executive Orders (.txt files)", fontsize=AXIS_LABEL_FONT_SIZE
    )
    plt.ylabel("President", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Executive Order Count by President", fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.gca().invert_yaxis()

    for bar, count in zip(bars, df["eo_count"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(x + 1, y, str(int(count)), va="center", fontsize=COUNT_LABEL_FONT_SIZE)

    plt.legend(
        handles=[
            Patch(facecolor=party_colors["Democrat"], label="Democrat"),
            Patch(facecolor=party_colors["Republican"], label="Republican"),
        ],
        title="Party",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    party_map = build_party_map()
    summary_df = collect_president_stats(args.input_dir, party_map)
    summary_df = sort_presidents(summary_df)

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
    print(f"Presidents included: {len(summary_df)}")
    print(f"Total EO files counted: {int(summary_df['eo_count'].sum())}")


if __name__ == "__main__":
    main()
