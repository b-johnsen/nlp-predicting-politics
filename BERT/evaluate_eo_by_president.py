#!/usr/bin/env python3
"""Evaluate EO classification accuracy by president and plot results.

This script runs a saved binary classifier (Democrat=0, Republican=1) over the
entire EO manifest, computes per-president accuracy where labels are known,
and saves a chronological bar chart.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Patch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EO_ROOT = REPO_ROOT / "eo_data"
MANIFEST_DEFAULT = EO_ROOT / "all_executive_orders_txt_clean" / "manifest.csv"
MODELS_ROOT = REPO_ROOT / "BERT" / "models"

# Label mapping follows templates/dataset_template.py:
# folder_0 -> label 0 (democrat), folder_1 -> label 1 (republican)
DEMOCRAT_PRESIDENT_DIRS = {
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
}

REPUBLICAN_PRESIDENT_DIRS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EO dataset with a trained BERT/LegalBERT classifier and "
            "plot per-president accuracy in chronological order."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["bert", "legalbert"],
        default="legalbert",
        help="Choose default checkpoint family if --model-path is not passed.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local Hugging Face model checkpoint directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFEST_DEFAULT,
        help="Path to EO manifest.csv.",
    )
    parser.add_argument(
        "--eo-root",
        type=Path,
        default=EO_ROOT,
        help=(
            "Root containing EO files. Local paths in manifest are resolved "
            "relative to this folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_ROOT / "results_eo_by_president",
        help="Directory where CSVs and plot image are saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    return parser.parse_args()


def infer_default_model_path(model_type: str) -> Path:
    pattern = "distillbert_model_*" if model_type == "bert" else "model_legal_bert_*"
    candidates = sorted(MODELS_ROOT.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No model folders found for pattern '{pattern}' in {MODELS_ROOT}"
        )
    return candidates[-1]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_text_path(local_path: str, eo_root: Path) -> Path:
    rel_path = Path(local_path)
    first_part = rel_path.parts[0] if rel_path.parts else ""
    if first_part == "all_executive_orders_txt_clean":
        return eo_root / rel_path
    return eo_root / "all_executive_orders_txt_clean" / rel_path


def president_dir_from_local_path(local_path: str) -> str:
    rel_path = Path(local_path)
    parts = rel_path.parts
    if not parts:
        return ""
    if parts[0] == "all_executive_orders_txt_clean" and len(parts) > 1:
        return parts[1]
    return parts[0]


def true_label_for_president_dir(president_dir: str) -> int | None:
    if president_dir in DEMOCRAT_PRESIDENT_DIRS:
        return 0
    if president_dir in REPUBLICAN_PRESIDENT_DIRS:
        return 1
    return None


def batch_predict(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
) -> list[int]:
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}
            logits = model(**encodings).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(batch_preds)
    return predictions


def build_accuracy_plot(
    per_president_df: pd.DataFrame, out_path: Path, title: str
) -> None:
    colors = per_president_df["true_label"].map({0: "#2C7FB8", 1: "#D7301F"}).tolist()

    plt.figure(figsize=(max(12, 0.45 * len(per_president_df)), 7))
    plt.bar(per_president_df["president"], per_president_df["accuracy"], color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("President")
    plt.title(title)
    plt.xticks(rotation=75, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(
        handles=[
            Patch(facecolor="#2C7FB8", label="Democrat"),
            Patch(facecolor="#D7301F", label="Republican"),
        ]
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    model_path = args.model_path or infer_default_model_path(args.model_type)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = pick_device()
    model.to(device)
    print(f"Using device: {device}")

    print(f"Reading manifest: {args.manifest_path}")
    df = pd.read_csv(args.manifest_path)
    required_cols = {"president", "date", "local_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    df["president_dir"] = df["local_path"].apply(president_dir_from_local_path)
    df["true_label"] = df["president_dir"].apply(true_label_for_president_dir)
    df["text_path"] = df["local_path"].apply(
        lambda p: resolve_text_path(p, args.eo_root)
    )

    missing_files = (~df["text_path"].apply(Path.exists)).sum()
    if missing_files:
        print(
            f"Warning: {missing_files} files listed in manifest were not found on disk."
        )

    available_df = df[df["text_path"].apply(Path.exists)].copy()
    texts = (
        available_df["text_path"]
        .apply(lambda p: p.read_text(encoding="utf-8", errors="ignore"))
        .tolist()
    )

    print(f"Running inference on {len(texts)} documents...")
    preds = batch_predict(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    available_df["pred_label"] = preds

    all_predictions_path = args.output_dir / f"eo_predictions_all_{timestamp}.csv"
    available_df.to_csv(all_predictions_path, index=False)

    known_df = available_df[available_df["true_label"].notna()].copy()
    unknown_count = len(available_df) - len(known_df)
    if unknown_count:
        print(
            "Note: "
            f"{unknown_count} docs have presidents outside Democrat/Republican label lists "
            "and are excluded from accuracy."
        )

    known_df["is_correct"] = known_df["pred_label"].astype(int) == known_df[
        "true_label"
    ].astype(int)
    known_df["date_dt"] = pd.to_datetime(known_df["date"], errors="coerce")

    per_president = (
        known_df.groupby(["president", "true_label"], as_index=False)
        .agg(
            n_samples=("is_correct", "size"),
            n_correct=("is_correct", "sum"),
            accuracy=("is_correct", "mean"),
            first_date=("date_dt", "min"),
        )
        .sort_values("first_date")
    )

    summary_path = args.output_dir / f"accuracy_by_president_{timestamp}.csv"
    per_president.to_csv(summary_path, index=False)

    plot_path = args.output_dir / f"accuracy_by_president_{timestamp}.png"
    title = f"EO Accuracy by President (Chronological) - {model_path.name}"
    build_accuracy_plot(per_president, plot_path, title)

    overall_accuracy = known_df["is_correct"].mean() if len(known_df) else float("nan")
    print(f"Evaluated docs (available): {len(available_df)}")
    print(f"Evaluated docs (known labels): {len(known_df)}")
    print(f"Overall accuracy (known labels): {overall_accuracy:.4f}")
    print(f"Saved all predictions: {all_predictions_path}")
    print(f"Saved per-president accuracy: {summary_path}")
    print(f"Saved bar chart: {plot_path}")


if __name__ == "__main__":
    main()
