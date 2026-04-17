import os
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

SEED = 42
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
EPS = 1e-8

EVAL_MODE = "leave_one_dataset_out"
HELD_OUT_SOURCE = "Enron"

# After training, evaluate the final model on each dataset source (deduped `combined`) one-by-one.
PER_SOURCE_EVAL_AFTER_TRAIN = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_for_dedup(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_text(df: pd.DataFrame) -> pd.Series:
    if "text_combined" in df.columns:
        return df["text_combined"].astype(str)
    if {"sender", "receiver", "date"}.issubset(df.columns):
        return (
            df["sender"].fillna("").astype(str)
            + " "
            + df["receiver"].fillna("").astype(str)
            + " "
            + df["date"].fillna("").astype(str)
            + " "
            + df["subject"].fillna("").astype(str)
            + " "
            + df["body"].fillna("").astype(str)
        )
    return df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)


def load_datasets() -> list[tuple[str, pd.DataFrame]]:
    return [
        ("CEAS_08", pd.read_csv("phishing/datasets/dataset2/CEAS_08.csv")),
        ("Enron", pd.read_csv("phishing/datasets/dataset2/Enron.csv")),
        ("Ling", pd.read_csv("phishing/datasets/dataset2/Ling.csv")),
        ("Nazario", pd.read_csv("phishing/datasets/dataset2/Nazario.csv")),
        ("Nigerian_Fraud", pd.read_csv("phishing/datasets/dataset2/Nigerian_Fraud.csv")),
        ("phishing_email", pd.read_csv("phishing/datasets/dataset2/phishing_email.csv")),
        ("SpamAssasin", pd.read_csv("phishing/datasets/dataset2/SpamAssasin.csv")),
    ]


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, tokenizer):
    train_loader = DataLoader(
        EmailDataset(x_train, y_train, tokenizer, max_length=MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        EmailDataset(x_val, y_val, tokenizer, max_length=MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        EmailDataset(x_test, y_test, tokenizer, max_length=MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def predict_labels(model, loader, device, show_progress: bool = False):
    model.eval()
    preds = []
    labels = []
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Eval", unit="batch", ncols=100, leave=False)
    with torch.no_grad():
        for batch in iterator:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
    return np.asarray(preds), np.asarray(labels)


def evaluate_per_source(
    model,
    tokenizer,
    combined: pd.DataFrame,
    device: torch.device,
) -> list[dict[str, object]]:
    """
    One forward-pass evaluation per dataset source (all rows in deduped `combined`).
    Returns structured results for printing / downstream use.
    """
    sources = sorted(combined["source"].unique().tolist())
    results: list[dict[str, object]] = []
    for src in sources:
        sub = combined[combined["source"] == src]
        n = len(sub)
        if n == 0:
            continue
        x = sub["text"].astype(str).tolist()
        y = sub["label"].to_numpy(dtype=int)
        loader = DataLoader(
            EmailDataset(x, y, tokenizer, max_length=MAX_LENGTH),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        pred, true = predict_labels(model, loader, device=device, show_progress=True)
        # Single-class slices: zero_division=0; fixed label order for 2x2 CM
        report = classification_report(true, pred, digits=4, zero_division=0, labels=[0, 1])
        cm = confusion_matrix(true, pred, labels=[0, 1])
        macro_f1 = float(f1_score(true, pred, average="macro", zero_division=0))
        row = {
            "source": src,
            "n_samples": n,
            "macro_f1": macro_f1,
            "report": report,
            "cm": cm,
        }
        results.append(row)
    return results


def train_one_run(x_train, y_train, x_val, y_val, x_test, y_test, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, tokenizer
    )

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, eps=EPS)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_state_dict = None

    train_pbar = tqdm(
        total=len(train_loader) * EPOCHS,
        desc=f"Training ({run_name})",
        unit="batch",
        ncols=100,
    )
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_pbar.update(1)
            train_pbar.set_postfix(epoch=f"{epoch + 1}/{EPOCHS}", loss=f"{loss.item():.4f}")

        val_pred, val_true = predict_labels(model, val_loader, device=device)
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))
        print(f"\nVal macro-F1 after epoch {epoch + 1}: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_pbar.close()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_dir = os.path.join(os.path.dirname(__file__), "bert_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    val_pred, val_true = predict_labels(model, val_loader, device=device)
    test_pred, test_true = predict_labels(model, test_loader, device=device)

    run = {
        "run_name": run_name,
        "best_val_macro_f1": best_val_f1,
        "val_report": classification_report(val_true, val_pred, digits=4),
        "val_cm": confusion_matrix(val_true, val_pred),
        "test_report": classification_report(test_true, test_pred, digits=4),
        "test_cm": confusion_matrix(test_true, test_pred),
        "saved_dir": save_dir,
    }
    return run, model, tokenizer, device


def audit_dataframe(name: str, df: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
    }

    key_cols = ["sender", "receiver", "date", "subject", "body", "text_combined", "urls", "label"]
    present = [c for c in key_cols if c in df.columns]
    if present:
        report["nulls_key_columns"] = df[present].isna().sum().to_dict()

    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        invalid_label_rows = int((labels.isna() | (~labels.isin([0, 1]))).sum())
        report["label_value_counts"] = df["label"].value_counts(dropna=False).to_dict()
        report["invalid_label_rows"] = invalid_label_rows

    try:
        text = build_text(df).astype(str).str.strip()
        text_norm = text.map(normalize_for_dedup)
        report["empty_text_rows"] = int((text_norm == "").sum())
        report["very_short_text_rows_lt_10"] = int((text_norm.str.len() < 10).sum())

        if "label" in df.columns:
            labels_int = pd.to_numeric(df["label"], errors="coerce")
            key = pd.DataFrame({"text_norm": text_norm, "label": labels_int})
            report["duplicates_within_dataset_text_norm_and_label"] = int(
                key.duplicated(subset=["text_norm", "label"]).sum()
            )
        else:
            report["duplicates_within_dataset_text_norm"] = int(text_norm.duplicated().sum())
    except Exception as e:
        report["text_stats_error"] = repr(e)

    return report


def main():
    set_seed(SEED)
    datasets = load_datasets()

    # Collect preprocessing audit; print it at the end as a consolidated report.
    preprocessing_report: dict[str, object] = {"per_dataset": []}
    for name, df in datasets:
        preprocessing_report["per_dataset"].append(audit_dataframe(name, df))

    # Build combined set with source info
    parts = []
    for name, df in datasets:
        text = build_text(df)
        parts.append(pd.DataFrame({"text": text.astype(str).str.strip(), "label": df["label"], "source": name}))

    combined = pd.concat(parts, ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    combined["text_norm"] = combined["text"].map(normalize_for_dedup)

    # Cross-dataset duplicates (before dedup)
    cross_dups = int(combined.duplicated(subset=["text_norm", "label"]).sum())
    preprocessing_report["cross_dataset_duplicates_before_global_dedup_text_norm_and_label"] = cross_dups

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_norm", "label"]).reset_index(drop=True)
    after = len(combined)
    preprocessing_report["global_dedup_before_rows"] = before
    preprocessing_report["global_dedup_after_rows"] = after
    preprocessing_report["global_dedup_removed_rows"] = before - after

    # -----------------------------
    # Train/eval (training progress prints live via tqdm)
    # -----------------------------
    runs_summary: list[dict[str, object]] = []
    last_model = None
    last_tokenizer = None
    last_device = None

    if EVAL_MODE == "random_split":
        texts = combined["text"].astype(str).tolist()
        y = combined["label"].to_numpy(dtype=int)

        x_train, x_temp, y_train, y_temp = train_test_split(
            texts, y, test_size=0.30, random_state=SEED, stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
        )
        run, last_model, last_tokenizer, last_device = train_one_run(
            x_train, y_train, x_val, y_val, x_test, y_test, run_name="random_split"
        )
        runs_summary.append(run)

    elif EVAL_MODE == "leave_one_dataset_out":
        sources = sorted(combined["source"].unique().tolist())

        def run_for_source(src: str):
            train_df = combined[combined["source"] != src]
            test_df = combined[combined["source"] == src]
            if len(train_df) == 0 or len(test_df) == 0:
                return

            x_all = train_df["text"].astype(str).tolist()
            y_all = train_df["label"].to_numpy(dtype=int)
            x_train, x_val, y_train, y_val = train_test_split(
                x_all, y_all, test_size=0.10, random_state=SEED, stratify=y_all
            )

            x_test = test_df["text"].astype(str).tolist()
            y_test = test_df["label"].to_numpy(dtype=int)

            print("\n", "=" * 80)
            print(f"Leave-one-dataset-out: held out source = {src} (test size={len(x_test)})")
            run, last_model, last_tokenizer, last_device = train_one_run(
                x_train, y_train, x_val, y_val, x_test, y_test, run_name=f"holdout_{src}"
            )
            run["held_out_source"] = src
            run["test_size"] = len(x_test)
            runs_summary.append(run)

        if HELD_OUT_SOURCE is not None:
            run_for_source(HELD_OUT_SOURCE)
        else:
            for src in sources:
                run_for_source(src)

    else:
        raise ValueError(f"Unknown EVAL_MODE={EVAL_MODE!r}")

    per_source_results: list[dict[str, object]] = []
    if PER_SOURCE_EVAL_AFTER_TRAIN and last_model is not None and last_tokenizer is not None and last_device is not None:
        print("\n" + "=" * 80)
        print("PER-SOURCE EVALUATION (same trained model, all deduped rows per source)")
        print("=" * 80)
        per_source_results = evaluate_per_source(last_model, last_tokenizer, combined, last_device)
        for row in per_source_results:
            print("\n", "-" * 40)
            print(f"Source: {row['source']} | n={row['n_samples']} | macro-F1={row['macro_f1']:.4f}")
            print("Classification report:\n", row["report"])
            print("Confusion matrix:\n", row["cm"])

    # -----------------------------
    # Consolidated output at the end
    # -----------------------------
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (preprocessing + metrics)")
    print("=" * 80)

    print("\nPreprocessing summary (combined):")
    print("Cross-dataset duplicates before global dedup (text_norm+label):",
          preprocessing_report["cross_dataset_duplicates_before_global_dedup_text_norm_and_label"])
    print("Global dedup rows:", preprocessing_report["global_dedup_before_rows"], "->",
          preprocessing_report["global_dedup_after_rows"],
          f"(removed {preprocessing_report['global_dedup_removed_rows']})")

    print("\nPreprocessing summary (per dataset):")
    for d in preprocessing_report["per_dataset"]:
        print("\n", "-" * 40)
        print("Dataset:", d.get("name"))
        print("Shape:", d.get("shape"))
        if "nulls_key_columns" in d:
            print("Null values (key columns):", d["nulls_key_columns"])
        if "label_value_counts" in d:
            print("Label value counts:", d["label_value_counts"])
        print("Invalid label rows:", d.get("invalid_label_rows", 0))
        print("Empty text rows:", d.get("empty_text_rows", "n/a"))
        print("Very short text rows (<10):", d.get("very_short_text_rows_lt_10", "n/a"))
        if "duplicates_within_dataset_text_norm_and_label" in d:
            print("Duplicates within dataset (text_norm+label):", d["duplicates_within_dataset_text_norm_and_label"])
        elif "duplicates_within_dataset_text_norm" in d:
            print("Duplicates within dataset (text_norm):", d["duplicates_within_dataset_text_norm"])

    print("\nModel runs:")
    for r in runs_summary:
        print("\n", "-" * 40)
        print("Run:", r.get("run_name"))
        if "held_out_source" in r:
            print("Held-out source:", r.get("held_out_source"), "| test size:", r.get("test_size"))
        print("Best val macro-F1:", f"{r.get('best_val_macro_f1', 0.0):.4f}")
        print("\nClassification report (val):\n", r.get("val_report"))
        print("Confusion matrix (val):\n", r.get("val_cm"))
        print("\nClassification report (test):\n", r.get("test_report"))
        print("Confusion matrix (test):\n", r.get("test_cm"))
        print("Saved:", r.get("saved_dir"))

    if per_source_results:
        print("\n" + "=" * 80)
        print("PER-SOURCE SUMMARY (macro-F1, n)")
        print("=" * 80)
        for row in per_source_results:
            print(f"  {row['source']}: n={row['n_samples']}, macro-F1={row['macro_f1']:.4f}")


if __name__ == "__main__":
    main()


'''
1. sender, receiver, date, subject, body => OUTPUT: label [1: 21842, 0: 17312]
2. subject, body ==> OUTPUT: label [0: 15791, 1: 13976]
3. subject, body ==> OUTPUT: label [0: 2401, 1: 458]
4. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 1565]
5. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 3332]
6. text_combined ==> OUTPUT: label [0: 39595, 1: 42891]
7. sender, receiver, date, subject, body ==> OUTPUT: label [0: 4091, 1: 1718]
'''