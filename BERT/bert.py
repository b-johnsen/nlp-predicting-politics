import numpy as np
import evaluate
from datasets import Dataset
from datetime import datetime
from pathlib import Path
import os
import sys
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = REPO_ROOT / "BERT"
DATASETS_DIR = BERT_DIR / "datasets-no-ner"
RESULTS_DIR = BERT_DIR / "results-no-ner"
EO_DATA_DIR = REPO_ROOT / "eo_data" / "eo_labeled_split"

# Ensure imports work when this script is run directly from the BERT directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from templates.dataset_template import load_text_dataset


def create_bert_datasets():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    train_path = EO_DATA_DIR / "train"

    # First path is democrat, second is for republican
    train_df = load_text_dataset(train_path / "democrat", train_path / "republican")

    # saving dataframe
    train_df.to_pickle(DATASETS_DIR / "train_dataset.pkl")

    test_path = EO_DATA_DIR / "test"

    test_df = load_text_dataset(test_path / "democrat", test_path / "republican")

    # saving dataframe as compressed file for storage
    test_df.to_pickle(DATASETS_DIR / "test_dataset.pkl")

    return train_df, test_df


def build_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    return compute_metrics


def _to_hf_dataset(df):
    return Dataset.from_pandas(df, preserve_index=False)


def train_bert_model(train_df, test_df):
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    train_dataset = _to_hf_dataset(train_df).shuffle(seed=42)
    split_train = train_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = _to_hf_dataset(test_df)

    tokenized_train = split_train["train"].map(preprocess_function, batched=True)
    tokenized_val = split_train["test"].map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=tokenized_val)
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test, metric_key_prefix="test"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_output_dir = BERT_DIR / f"model_{timestamp}"
    trainer.save_model(str(model_output_dir))

    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Saved model to: {model_output_dir}")

    return trainer


def main():
    train_df, test_df = create_bert_datasets()
    trainer = train_bert_model(train_df, test_df)


if __name__ == "__main__":
    main()
