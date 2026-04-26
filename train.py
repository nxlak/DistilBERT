#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import DistilBertModel, DistilBertTokenizerFast, get_cosine_schedule_with_warmup


# ================== НАСТРОЙКИ ==================
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "./distil_multiclass_model_ctx"

MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 5
FREEZE_EPOCHS = 1
GRAD_ACCUM_STEPS = 1
MIN_SAMPLES_PER_CLASS = 3
NUM_WORKERS = 0

SPECIAL_TOKENS = ["[CTX]", "[UTT]", "[AFT]"]
TRUNCATION_SIDE = "left"
DROP_EXACT_DUPLICATES = False

TRANSLATE_TABLE = str.maketrans(
    {
        "\u00A0": " ",
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
    }
)


# ================== БАЗОВЫЕ УТИЛИТЫ ==================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_text(text: str) -> str:
    text = str(text).translate(TRANSLATE_TABLE).strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_structured_text(phrase: str, context_before: str = "", context_after: str = "") -> str:
    phrase = str(phrase or "").strip()
    context_before = str(context_before or "").strip()
    context_after = str(context_after or "").strip()

    if context_before and context_after:
        return f"[CTX] {context_before} [UTT] {phrase} [AFT] {context_after}"
    if context_before:
        return f"[CTX] {context_before} [UTT] {phrase}"
    if context_after:
        return f"[UTT] {phrase} [AFT] {context_after}"
    return phrase


def row_to_training_text(row: pd.Series) -> str:
    phrase = str(row.get("Phrase", "") or "").strip()
    phrase_with_context = str(row.get("PhraseWithContext", "") or "").strip()
    context_before = str(row.get("ContextBefore", "") or "").strip()
    context_after = str(row.get("ContextAfter", "") or "").strip()

    if context_before or context_after:
        return build_structured_text(phrase, context_before, context_after)
    if phrase_with_context:
        return phrase_with_context
    return phrase


def infer_dataset_path(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    here = Path(__file__).resolve().parent
    candidates = [
        here / "dataset.csv",
        here / "dataset_cleaned.csv",
        here / "dataset_.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Не найден dataset.csv / dataset_cleaned.csv / dataset_.csv рядом со скриптом.")


def context_stats(df: pd.DataFrame) -> Dict[str, int]:
    phrase = df.get("Phrase", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    pwc = df.get("PhraseWithContext", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    cb = df.get("ContextBefore", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    ca = df.get("ContextAfter", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()

    structured = ((cb != "") | (ca != "")).sum()
    pwc_nonempty = (pwc != "").sum()
    real_extra = (((cb != "") | (ca != "")) | ((pwc != "") & (pwc != phrase))).sum()

    return {
        "rows": int(len(df)),
        "phrase_with_context_nonempty": int(pwc_nonempty),
        "structured_context_rows": int(structured),
        "rows_with_real_extra_context": int(real_extra),
    }


# ================== ЗАГРУЗКА ДАННЫХ ==================
def load_data(dataset_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    path = infer_dataset_path(dataset_path)
    print(f"Загрузка датасета из: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    required = {"Phrase", "Intentionality"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ожидались колонки {sorted(required)}, но найдены: {list(df.columns)}")

    df["Intentionality"] = df["Intentionality"].astype(str).str.strip()
    df["Intentionality"] = df["Intentionality"].str.replace(r"\s+", " ", regex=True)

    df["TextForTraining"] = df.apply(row_to_training_text, axis=1)
    df["TextForTraining"] = df["TextForTraining"].astype(str).str.strip()
    df["TextForTraining"] = df["TextForTraining"].apply(normalize_text)

    before_drop_empty = len(df)
    df = df[(df["TextForTraining"] != "") & (df["Intentionality"] != "")].copy()
    dropped_empty = before_drop_empty - len(df)
    if dropped_empty:
        print(f"Удалено пустых строк: {dropped_empty}")

    if DROP_EXACT_DUPLICATES:
        before_dupes = len(df)
        df = df.drop_duplicates(subset=["TextForTraining", "Intentionality"]).reset_index(drop=True)
        dropped_dupes = before_dupes - len(df)
        if dropped_dupes:
            print(f"Удалено точных дубликатов text+label: {dropped_dupes}")

    stats = context_stats(df)
    print("Статистика контекста:")
    print(f"  PhraseWithContext непустой:        {stats['phrase_with_context_nonempty']}/{stats['rows']}")
    print(f"  Есть ContextBefore/After:          {stats['structured_context_rows']}/{stats['rows']}")
    print(f"  Есть реальный дополнительный контекст: {stats['rows_with_real_extra_context']}/{stats['rows']}")

    label_counts = df["Intentionality"].value_counts()
    rare_labels = label_counts[label_counts < MIN_SAMPLES_PER_CLASS]
    if len(rare_labels) > 0:
        print(
            f"Удаляю {len(rare_labels)} классов с < {MIN_SAMPLES_PER_CLASS} примерами "
            f"({int(rare_labels.sum())} строк), чтобы stratify не падал."
        )
        df = df[~df["Intentionality"].isin(rare_labels.index)].copy()

    intents = sorted(df["Intentionality"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(intents)}
    df["label_idx"] = df["Intentionality"].map(label2id).astype(int)
    df = df.reset_index(drop=True)

    print(f"Итого после очистки: {len(df)} примеров")
    print(f"Количество классов: {len(intents)}")
    return df, intents, stats


# ================== DATASET ==================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


# ================== МОДЕЛЬ ==================
class DistilBertClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_state = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_state)
        return self.classifier(x)


def set_bert_frozen(model: DistilBertClassifier, frozen: bool) -> None:
    for param in model.bert.parameters():
        param.requires_grad = not frozen


# ================== МЕТРИКИ / ВИЗУАЛИЗАЦИИ ==================
def compute_metrics(true_labels, pred_labels) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "f1_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
    }


def plot_training_history(history: Dict[str, List[float]], path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["val_f1_macro"], label="Val Macro F1")
    plt.title("Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["val_f1_weighted"], label="Val Weighted F1")
    plt.title("Validation Weighted F1")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted F1")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, classes: List[str], path: str) -> None:
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(classes))))

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if len(classes) <= 50:
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, rotation=0, fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    cm_csv_path = path.replace(".png", ".csv")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_csv_path, encoding="utf-8")
    print(f"Числовая матрица ошибок сохранена в: {cm_csv_path}")


# ================== ОБУЧЕНИЕ ==================
def train_model(dataset_path: Optional[Path] = None) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()
    set_seed(SEED)

    print(f"Используемое устройство: {device}")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    print("Загрузка данных...")
    df, intents, ctx_stats = load_data(dataset_path)
    num_classes = len(intents)
    print(f"Первые классы: {intents[:5]}")

    texts = df["TextForTraining"].values
    labels_int = df["label_idx"].values

    label_counts_df = (
        df["label_idx"]
        .value_counts()
        .sort_index()
        .rename_axis("label_idx")
        .reset_index(name="count")
    )
    label_counts_df["intent"] = label_counts_df["label_idx"].map(lambda i: intents[int(i)])
    label_counts_df.to_csv(os.path.join(MODEL_DIR, "class_distribution.csv"), index=False, encoding="utf-8")

    print("Разделение на train / val / test...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts,
        labels_int,
        test_size=0.15,
        random_state=SEED,
        stratify=labels_int,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.15,
        random_state=SEED,
        stratify=y_trainval,
    )
    print(f"Размеры выборок: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = len(y_train) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.1, 30.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print("Пример весов классов (первые 10):")
    for i in range(min(10, num_classes)):
        print(f"  [{i}] {intents[i]}: count={class_counts[i]}, weight={class_weights[i]:.3f}")

    print("Загрузка токенизатора...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.truncation_side = TRUNCATION_SIDE
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"Добавлено специальных токенов: {added_tokens}")
    print(f"truncation_side = {tokenizer.truncation_side}")

    train_dataset = IntentDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = IntentDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = IntentDataset(X_test, y_test, tokenizer, MAX_LEN)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    print("Инициализация модели...")
    model = DistilBertClassifier(model_name=MODEL_NAME, num_classes=num_classes, dropout=DROPOUT).to(device)
    if added_tokens > 0:
        model.bert.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_update_steps_per_epoch = max(1, (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS)
    total_steps = total_update_steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    best_val_macro = -1.0
    patience_counter = 0

    print("\n" + "=" * 72)
    print("Начало обучения DistilBERT (мультикласс + согласованный контекстный пайплайн)")
    print("=" * 72)
    print(
        f"MAX_LEN={MAX_LEN}, LABEL_SMOOTHING={LABEL_SMOOTHING}, FREEZE_EPOCHS={FREEZE_EPOCHS}, "
        f"AMP={'Да' if use_amp else 'Нет'}, GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}\n"
    )

    for epoch in range(1, EPOCHS + 1):
        if FREEZE_EPOCHS > 0 and epoch <= FREEZE_EPOCHS:
            set_bert_frozen(model, frozen=True)
            frozen_state = "FROZEN"
        else:
            set_bert_frozen(model, frozen=False)
            frozen_state = "UNFROZEN"

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train] ({frozen_state})", leave=False)

        for step, batch in enumerate(train_iter, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)

            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

            train_loss += float(loss.item()) * GRAD_ACCUM_STEPS

            should_step = (step % GRAD_ACCUM_STEPS == 0) or (step == len(train_loader))
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss /= max(1, len(train_loader))
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        all_val_true = []
        all_val_pred = []

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
                attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
                labels = batch["labels"].to(device, non_blocking=pin_memory)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())

                preds = logits.argmax(dim=1).cpu().numpy()
                trues = labels.cpu().numpy()
                all_val_true.extend(trues)
                all_val_pred.extend(preds)

        val_loss /= max(1, len(val_loader))
        metrics = compute_metrics(all_val_true, all_val_pred)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1_macro"].append(metrics["f1_macro"])
        history["val_f1_weighted"].append(metrics["f1_weighted"])

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val Macro F1: {metrics['f1_macro']:.4f} | "
            f"Val Weighted F1: {metrics['f1_weighted']:.4f} | "
            f"{frozen_state}"
        )

        if metrics["f1_macro"] > best_val_macro:
            best_val_macro = float(metrics["f1_macro"])
            patience_counter = 0

            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_val_macro,
                    "best_val_weighted_f1": float(metrics["f1_weighted"]),
                    "intents": intents,
                    "label2id": {label: idx for idx, label in enumerate(intents)},
                    "id2label": {idx: label for idx, label in enumerate(intents)},
                    "model_name": MODEL_NAME,
                    "max_len": MAX_LEN,
                    "dropout": DROPOUT,
                    "label_smoothing": LABEL_SMOOTHING,
                    "special_tokens": SPECIAL_TOKENS,
                    "truncation_side": tokenizer.truncation_side,
                    "context_stats": ctx_stats,
                    "seed": SEED,
                },
                save_path,
            )
            print(f"Модель сохранена: {save_path} (best Val Macro F1 = {best_val_macro:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    print("\nЗагрузка лучшей модели и тестирование...")
    best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
    checkpoint = safe_torch_load(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.eval()
    all_test_true = []
    all_test_pred = []
    all_test_probs = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Тестирование"):
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1).cpu().numpy()
            trues = labels.cpu().numpy()

            all_test_true.extend(trues)
            all_test_pred.extend(preds)
            all_test_probs.extend(probs.max(dim=1).values.cpu().numpy().tolist())

            indices = batch["index"].cpu().numpy()
            batch_texts = [X_test[i] for i in indices]
            all_texts.extend(batch_texts)

    test_metrics = compute_metrics(all_test_true, all_test_pred)
    report_dict = classification_report(
        all_test_true,
        all_test_pred,
        labels=list(range(num_classes)),
        target_names=intents,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        all_test_true,
        all_test_pred,
        labels=list(range(num_classes)),
        target_names=intents,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 72)
    print("Финальные метрики на тесте")
    print("=" * 72)
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1:      {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1:   {test_metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(report_text)

    results_df = pd.DataFrame(
        {
            "text": all_texts,
            "true_label_idx": all_test_true,
            "true_label": [intents[i] for i in all_test_true],
            "predicted_label_idx": all_test_pred,
            "predicted_label": [intents[i] for i in all_test_pred],
            "confidence": all_test_probs,
            "correct": [a == b for a, b in zip(all_test_true, all_test_pred)],
        }
    )
    results_df.to_csv(os.path.join(MODEL_DIR, "test_results.csv"), index=False, encoding="utf-8")

    history_df = pd.DataFrame(history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False, encoding="utf-8")

    with open(os.path.join(MODEL_DIR, "intent_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(intents))

    with open(os.path.join(MODEL_DIR, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODEL_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    plot_training_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    plot_confusion_matrix(all_test_true, all_test_pred, intents, os.path.join(MODEL_DIR, "confusion_matrix.png"))

    tokenizer.save_pretrained(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "model_name": MODEL_NAME,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "dropout": DROPOUT,
                "label_smoothing": LABEL_SMOOTHING,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "freeze_epochs": FREEZE_EPOCHS,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
                "special_tokens": SPECIAL_TOKENS,
                "truncation_side": TRUNCATION_SIDE,
                "context_stats": ctx_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nВсе результаты сохранены в: {MODEL_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение DistilBERT-классификатора интенсий")
    parser.add_argument("dataset_path", nargs="?", default=None, help="Путь до CSV-датасета")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_arg = Path(args.dataset_path) if args.dataset_path else None
    train_model(dataset_arg)
