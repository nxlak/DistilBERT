#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


MODEL_DIR_CANDIDATES = [
    "distil_multiclass_model_ctx",
    "distil_multiclass_model",
]

DEFAULT_TOP_K = 5
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

SAMPLES: List[Dict[str, Any]] = [
    {
        "utterance": "Could you explain again what you meant by that last point about the project timeline?",
        "context": "We discussed the milestones yesterday, but I’m not sure how the dates connect to the resources.",
    },
    {
        "utterance": "I should like your opinion on that.",
        "context": "People of my age don't really know anything about those times. We can only read about them in books.",
    },
    {
        "utterance": "And don’t look at me.",
        "context": "Any signal? No. Don't come up to me until you see me among a lot of people.",
    },
    {
        "utterance": "If you ever need someone to practice with before your presentation, just send me a message.",
        "context": "",
    },
]


class DistilBertClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.0):
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


def build_structured_text(utterance: str, context_before: str = "", context_after: str = "") -> str:
    utterance = str(utterance or "").strip()
    context_before = str(context_before or "").strip()
    context_after = str(context_after or "").strip()

    if context_before and context_after:
        return f"[CTX] {context_before} [UTT] {utterance} [AFT] {context_after}"
    if context_before:
        return f"[CTX] {context_before} [UTT] {utterance}"
    if context_after:
        return f"[UTT] {utterance} [AFT] {context_after}"
    return utterance


def build_input_text(sample: Dict[str, Any]) -> str:
    utterance = sample.get("utterance", sample.get("Phrase", ""))
    context_before = sample.get("context_before")
    if context_before is None:
        context_before = sample.get("context", sample.get("ContextBefore", ""))
    context_after = sample.get("context_after", sample.get("ContextAfter", ""))
    text = build_structured_text(utterance, context_before, context_after)
    return normalize_text(text)


def find_model_dir(explicit_model_dir: Optional[str]) -> Path:
    if explicit_model_dir:
        model_dir = Path(explicit_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Указанная директория модели не существует: {model_dir}")
        return model_dir

    here = Path(__file__).resolve().parent
    for name in MODEL_DIR_CANDIDATES:
        candidate = here / name
        if candidate.exists() and (candidate / "best_model.pt").exists():
            return candidate

    if (here / "best_model.pt").exists():
        return here

    raise FileNotFoundError("Не найден best_model.pt. Укажи --model-dir или положи infer.py рядом с моделью.")


def load_checkpoint(model_dir: Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт: {ckpt_path}")
    return safe_torch_load(str(ckpt_path), device)


def resolve_intents(model_dir: Path, checkpoint: Dict[str, Any]) -> List[str]:
    intents = checkpoint.get("intents")
    if intents:
        return list(intents)

    id2label = checkpoint.get("id2label")
    if id2label:
        pairs = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])
        return [label for _, label in pairs]

    intent_path = model_dir / "intent_list.txt"
    if intent_path.exists():
        return [line.strip() for line in intent_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raise ValueError("Не удалось восстановить список интенсий из checkpoint или intent_list.txt")


def load_tokenizer(model_dir: Path, checkpoint: Dict[str, Any]) -> DistilBertTokenizerFast:
    model_name = checkpoint.get("model_name", "distilbert-base-uncased")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    special_tokens = checkpoint.get("special_tokens", ["[CTX]", "[UTT]", "[AFT]"])
    existing_vocab = tokenizer.get_vocab()
    missing_tokens = [tok for tok in special_tokens if tok not in existing_vocab]
    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})

    tokenizer.truncation_side = checkpoint.get("truncation_side", "left")
    return tokenizer


def load_samples_from_json(json_path: str) -> List[Dict[str, Any]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON должен содержать объект или список объектов.")


def predict(
    samples: List[Dict[str, Any]],
    model: nn.Module,
    tokenizer,
    intents: List[str],
    device: torch.device,
    top_k: int,
    max_len: int,
) -> List[Dict[str, Any]]:
    texts = [build_input_text(sample) for sample in samples]
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
        pred_idx = probs.argmax(dim=-1)

    results = []
    for i, sample in enumerate(samples):
        predictions = []
        for p, idx in zip(top_probs[i].tolist(), top_idx[i].tolist()):
            predictions.append({
                "intent": intents[idx],
                "probability": float(p),
            })

        best_idx = int(pred_idx[i].item())
        results.append(
            {
                "utterance": sample.get("utterance", sample.get("Phrase", "")),
                "context_before": sample.get("context_before", sample.get("context", sample.get("ContextBefore", None))),
                "context_after": sample.get("context_after", sample.get("ContextAfter", None)),
                "combined_text": texts[i],
                "predicted_intent": intents[best_idx],
                "predicted_probability": float(probs[i, best_idx].item()),
                "top_k": k,
                "predictions": predictions,
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Инференс для классификатора интенсий")
    parser.add_argument("--model-dir", default=None, help="Папка с best_model.pt")
    parser.add_argument("--utterance", default=None, help="Текст реплики")
    parser.add_argument("--context", default=None, help="Контекст до реплики")
    parser.add_argument("--context-after", default=None, help="Контекст после реплики")
    parser.add_argument("--input-json", default=None, help="JSON с одним объектом или списком объектов")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько top-предсказаний вывести")
    parser.add_argument("--max-len", type=int, default=None, help="Максимальная длина. По умолчанию берётся из checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство для инференса: {device}")

    model_dir = find_model_dir(args.model_dir)
    checkpoint = load_checkpoint(model_dir, device)
    intents = resolve_intents(model_dir, checkpoint)
    tokenizer = load_tokenizer(model_dir, checkpoint)

    model_name = checkpoint.get("model_name", "distilbert-base-uncased")
    dropout = float(checkpoint.get("dropout", 0.0))
    max_len = int(args.max_len or checkpoint.get("max_len", 192))

    model = DistilBertClassifier(model_name=model_name, num_classes=len(intents), dropout=dropout).to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if args.input_json:
        samples = load_samples_from_json(args.input_json)
    elif args.utterance is not None:
        samples = [
            {
                "utterance": args.utterance,
                "context": args.context,
                "context_after": args.context_after,
            }
        ]
    else:
        samples = SAMPLES

    print(f"Загружено интенсий: {len(intents)}")
    print(f"max_len = {max_len} | truncation_side = {tokenizer.truncation_side}")

    results = predict(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        intents=intents,
        device=device,
        top_k=args.top_k,
        max_len=max_len,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
