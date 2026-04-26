#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Вставь свой API-ключ сюда.
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"

# Файлы входа / выхода.
INPUT_TEXT_PATH = "source_text.txt"
INTENTS_TXT_PATH = "intent_list.txt"
OUTPUT_CSV_PATH = "dataset.csv"

# Дополнительный JSONL-лог с полными ответами моделей.
DEBUG_JSONL_PATH = "annotation_debug.jsonl"

# Три разные модели.
MODEL_1 = "gpt-5.4-mini"
MODEL_2 = "gpt-5.4-nano"
MODEL_JUDGE = "gpt-5.5"

# Ограничение на количество фраз.
# Для полной обработки поставь None.
# Для теста удобно поставить, например, 20.
MAX_SENTENCES: Optional[int] = None

# Если dataset.csv уже существует, скрипт продолжит с места остановки.
RESUME_IF_OUTPUT_EXISTS = True

# Минимальная длина фразы.
MIN_CHARS = 4
MIN_WORDS = 1

# Делить ли текст по точке с запятой.
SPLIT_ON_SEMICOLON = True

# Количество попыток при ошибках API.
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP_SECONDS = 2

# Пауза между обработкой фраз.
SLEEP_BETWEEN_ITEMS_SECONDS = 0.2

# Таймаут одного запроса.
REQUEST_TIMEOUT_SECONDS = 90


# ============================================================
# НОРМАЛИЗАЦИЯ ТЕКСТА
# ============================================================

TRANSLATE_TABLE = str.maketrans(
    {
        "\u00A0": " ",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
        "—": "—",
        "–": "-",
    }
)

ABBREVIATIONS = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Capt.",
    "Col.",
    "Gen.",
    "Lt.",
    "No.",
    "Fig.",
    "Vol.",
    "Ch.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
    "a.m.",
    "p.m.",
    "M.",
    "Mme.",
]

PROTECTED_DOT = "<DOT>"


def normalize_raw_text(text: str) -> str:
    """
    Общая нормализация исходного художественного текста.
    Сохраняет абзацы, но убирает случайные переносы внутри предложения.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATE_TABLE)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Удаление переносов с дефисом внутри слова:
    # exam-\nple -> example
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Удаляем лишние пробелы вокруг переносов.
    text = re.sub(r"[ \t]+", " ", text)

    # Нормализуем множественные пустые строки.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def split_into_blocks(text: str) -> List[str]:
    """
    Делит текст на блоки.

    Логика:
    - пустая строка = граница абзаца;
    - новая строка, начинающаяся с тире, часто является новой репликой;
    - одиночные переносы внутри обычного предложения склеиваются.
    """
    blocks: List[str] = []
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            block = " ".join(line.strip() for line in current_lines if line.strip())
            block = re.sub(r"\s+", " ", block).strip()
            if block:
                blocks.append(block)
            current_lines = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            flush()
            continue

        # Диалоговая строка с тире обычно является отдельной смысловой единицей.
        if re.match(r"^[—-]\s+", line) and current_lines:
            flush()
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return blocks


def protect_dots(text: str) -> str:
    """
    Защищает точки, которые не должны считаться концом предложения:
    - сокращения: Mr., Dr., etc.
    - инициалы: A. S. Pushkin
    - десятичные числа: 3.14
    """
    protected = text

    # Десятичные числа.
    protected = re.sub(r"(?<=\d)\.(?=\d)", PROTECTED_DOT, protected)

    # Сокращения.
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        pattern = re.compile(re.escape(abbr), flags=re.IGNORECASE)
        protected = pattern.sub(lambda m: m.group(0).replace(".", PROTECTED_DOT), protected)

    # Инициалы.
    protected = re.sub(
        r"\b([A-Za-zА-Яа-яЁё])\.",
        lambda m: m.group(1) + PROTECTED_DOT,
        protected,
    )

    return protected


def unprotect_dots(text: str) -> str:
    return text.replace(PROTECTED_DOT, ".")


def is_sentence_boundary(text: str, punct_start: int, punct_end: int) -> bool:
    """
    Проверяет, можно ли считать найденный знак препинания границей фразы.
    """
    punct = text[punct_start:punct_end]

    if ";" in punct:
        return SPLIT_ON_SEMICOLON

    # Ищем следующий значимый символ.
    k = punct_end
    while k < len(text) and text[k].isspace():
        k += 1

    if k >= len(text):
        return True

    next_char = text[k]

    # После точки / ! / ? обычно новое предложение начинается с:
    # - большой буквы;
    # - цифры;
    # - кавычки;
    # - тире;
    # - скобки.
    if next_char.isupper() or next_char.isdigit():
        return True

    if next_char in {'"', "'", "(", "[", "{", "—", "-"}:
        return True

    return False


def split_block_into_sentences(block: str) -> List[str]:
    """
    Делит один блок текста на фразы / предложения.

    Поддерживает:
    - . ! ?
    - многоточие ...
    - unicode-многоточие …
    - точку с запятой ;
    - закрывающие кавычки после знака препинания
    - сокращения и инициалы
    """
    text = protect_dots(block)
    sentences: List[str] = []

    start = 0
    i = 0

    while i < len(text):
        char = text[i]

        boundary_chars = ".!?…"
        if SPLIT_ON_SEMICOLON:
            boundary_chars += ";"

        if char in boundary_chars:
            j = i + 1

            # Съедаем последовательности вроде "...", "?!", "!!".
            while j < len(text) and text[j] in ".!?…":
                j += 1

            # Съедаем закрывающие кавычки и скобки.
            while j < len(text) and text[j] in {'"', "'", ")", "]", "}", "”", "’"}:
                j += 1

            if is_sentence_boundary(text, i, j):
                fragment = text[start:j].strip()
                fragment = unprotect_dots(fragment)
                fragment = clean_sentence(fragment)

                if is_valid_sentence(fragment):
                    sentences.append(fragment)

                # Пропускаем пробелы после границы.
                while j < len(text) and text[j].isspace():
                    j += 1

                start = j
                i = j
                continue

            i = j
            continue

        i += 1

    tail = text[start:].strip()
    tail = unprotect_dots(tail)
    tail = clean_sentence(tail)

    if is_valid_sentence(tail):
        sentences.append(tail)

    return sentences


def clean_sentence(sentence: str) -> str:
    """
    Финальная чистка одной фразы.
    """
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)

    # Убираем висящие разделители.
    sentence = sentence.strip(" \t\n\r")

    return sentence


def is_valid_sentence(sentence: str) -> bool:
    """
    Фильтр мусорных фрагментов.
    """
    if not sentence:
        return False

    if len(sentence) < MIN_CHARS:
        return False

    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", sentence)
    if len(words) < MIN_WORDS:
        return False

    # Не берём строки, состоящие только из знаков препинания.
    if re.fullmatch(r"[\W_]+", sentence):
        return False

    return True


def extract_sentences_from_fiction_text(text: str) -> List[str]:
    """
    Полный пайплайн нарезки художественного текста.
    """
    normalized = normalize_raw_text(text)
    blocks = split_into_blocks(normalized)

    result: List[str] = []
    for block in blocks:
        result.extend(split_block_into_sentences(block))

    return result


# ============================================================
# РАБОТА С ИНТЕНСИЯМИ И КОНТЕКСТОМ
# ============================================================

def load_intents(path: str) -> List[str]:
    """
    Загружает список интенсий из .txt.
    Формат: одна интенсия на строку.
    Пустые строки и строки с # игнорируются.
    """
    intent_path = Path(path)
    if not intent_path.exists():
        raise FileNotFoundError(f"Не найден файл со списком интенсий: {path}")

    intents: List[str] = []
    seen = set()

    for line in intent_path.read_text(encoding="utf-8").splitlines():
        intent = line.strip()

        if not intent:
            continue

        if intent.startswith("#"):
            continue

        if intent not in seen:
            intents.append(intent)
            seen.add(intent)

    if not intents:
        raise ValueError("Список интенсий пуст.")

    return intents


def build_phrase_with_context(
    phrase: str,
    context_before: str = "",
    context_after: str = "",
) -> str:
    """
    Формирует текст в том же стиле, который используется в train.py / inference.py.
    """
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


def make_examples(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Создаёт примеры:
    - Phrase
    - ContextBefore
    - ContextAfter
    - PhraseWithContext
    """
    examples: List[Dict[str, str]] = []

    for i, phrase in enumerate(sentences):
        context_before = sentences[i - 1] if i > 0 else ""
        context_after = sentences[i + 1] if i + 1 < len(sentences) else ""

        examples.append(
            {
                "Phrase": phrase,
                "ContextBefore": context_before,
                "ContextAfter": context_after,
                "PhraseWithContext": build_phrase_with_context(
                    phrase=phrase,
                    context_before=context_before,
                    context_after=context_after,
                ),
                "SourceSentenceIndex": str(i),
            }
        )

    return examples


# ============================================================
# OPENAI API
# ============================================================

def make_client() -> OpenAI:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        raise ValueError("Укажи OPENAI_API_KEY прямо в коде.")

    return OpenAI(api_key=OPENAI_API_KEY)


def intent_annotation_schema(intents: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": intents,
                "description": "Одна наиболее подходящая интенсия из списка.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Уверенность модели от 0 до 1.",
            },
            "rationale": {
                "type": "string",
                "description": "Короткое объяснение выбора на русском языке.",
            },
        },
        "required": ["intent", "confidence", "rationale"],
    }


def judge_schema(candidate_intents: List[str]) -> Dict[str, Any]:
    unique_candidates = list(dict.fromkeys(candidate_intents))

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "final_intent": {
                "type": "string",
                "enum": unique_candidates,
                "description": "Итоговая интенсия. Нужно выбрать только между ответом модели 1 и модели 2.",
            },
            "winner": {
                "type": "string",
                "enum": ["model_1", "model_2", "same_answer"],
                "description": "Какая модель дала лучший ответ.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Уверенность судьи от 0 до 1.",
            },
            "rationale": {
                "type": "string",
                "description": "Короткое объяснение решения судьи на русском языке.",
            },
        },
        "required": ["final_intent", "winner", "confidence", "rationale"],
    }


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Запасной парсер на случай, если модель вернула JSON с лишним текстом.
    При Structured Outputs обычно не нужен, но оставлен для устойчивости.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Не удалось найти JSON в ответе модели: {text[:500]}")

    return json.loads(text[start : end + 1])


def call_openai_json(
    client: OpenAI,
    model: str,
    instructions: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Универсальный вызов модели с JSON Schema.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                reasoning={
                    "effort": "low",
                },
                store=False,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            output_text = response.output_text
            return parse_json_from_text(output_text)

        except Exception as exc:
            last_error = exc

            if attempt < RETRY_ATTEMPTS:
                sleep_seconds = RETRY_BASE_SLEEP_SECONDS * attempt
                print(
                    f"Ошибка API / парсинга. Попытка {attempt}/{RETRY_ATTEMPTS}. "
                    f"Повтор через {sleep_seconds} сек. Ошибка: {exc}"
                )
                time.sleep(sleep_seconds)
            else:
                break

    raise RuntimeError(f"Не удалось получить корректный ответ от модели {model}: {last_error}")


def normalize_intent_choice(intent: str, intents: List[str]) -> str:
    """
    Проверяет, что выбранная интенсия есть в списке.
    Если отличается только регистр / пробелы, исправляет.
    """
    intent = str(intent or "").strip()

    if intent in intents:
        return intent

    lower_map = {x.lower().strip(): x for x in intents}
    key = intent.lower().strip()

    if key in lower_map:
        return lower_map[key]

    raise ValueError(f"Модель вернула интенсию вне списка: {intent}")


def clamp_probability(value: Any) -> float:
    try:
        probability = float(value)
    except Exception:
        probability = 0.0

    return max(0.0, min(1.0, probability))


def format_intents_for_prompt(intents: List[str]) -> str:
    return "\n".join(f"{i + 1}. {intent}" for i, intent in enumerate(intents))


def annotate_with_model(
    client: OpenAI,
    model: str,
    intents: List[str],
    phrase: str,
    context_before: str,
    context_after: str,
) -> Dict[str, Any]:
    instructions = """
Ты размечаешь художественный текст для датасета классификации интенсий.

Твоя задача:
- определить коммуникативную интенсию ОСНОВНОЙ ФРАЗЫ;
- использовать контекст до и после только для снятия неоднозначности;
- выбрать ровно одну интенсию из списка;
- не придумывать новые интенсии;
- не выбирать интенсию по теме текста, если коммуникативная функция фразы другая.

Интенсия — это коммуникативная направленность высказывания:
например, просьба, благодарность, совет, поддержка, уточнение, несогласие и т.д.

Отвечай строго по JSON-схеме.
"""

    prompt = f"""
Список допустимых интенсий:
{format_intents_for_prompt(intents)}

Контекст до основной фразы:
{context_before if context_before else "[контекст отсутствует]"}

Основная фраза:
{phrase}

Контекст после основной фразы:
{context_after if context_after else "[контекст отсутствует]"}

Выбери наиболее подходящую интенсию для основной фразы.
"""

    data = call_openai_json(
        client=client,
        model=model,
        instructions=instructions,
        user_prompt=prompt,
        schema_name="intent_annotation",
        schema=intent_annotation_schema(intents),
    )

    intent = normalize_intent_choice(data.get("intent", ""), intents)

    return {
        "intent": intent,
        "confidence": clamp_probability(data.get("confidence", 0)),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def judge_two_annotations(
    client: OpenAI,
    intents: List[str],
    phrase: str,
    context_before: str,
    context_after: str,
    model_1_result: Dict[str, Any],
    model_2_result: Dict[str, Any],
) -> Dict[str, Any]:
    intent_1 = normalize_intent_choice(model_1_result["intent"], intents)
    intent_2 = normalize_intent_choice(model_2_result["intent"], intents)

    candidate_intents = [intent_1, intent_2]

    instructions = """
Ты выступаешь как модель-судья при разметке датасета интенсий.

Тебе даны:
- исходная фраза;
- контекст до и после;
- ответ модели #1;
- ответ модели #2.

Твоя задача:
- выбрать, какой из двух ответов лучше соответствует основной фразе;
- НЕ выбирать третью интенсию;
- итоговая интенсия должна быть равна либо ответу модели #1, либо ответу модели #2;
- если обе модели выбрали одну и ту же интенсию, укажи same_answer.

Оценивай именно коммуникативную функцию основной фразы, а не общую тему фрагмента.
Отвечай строго по JSON-схеме.
"""

    prompt = f"""
Список всех допустимых интенсий:
{format_intents_for_prompt(intents)}

Контекст до основной фразы:
{context_before if context_before else "[контекст отсутствует]"}

Основная фраза:
{phrase}

Контекст после основной фразы:
{context_after if context_after else "[контекст отсутствует]"}

Ответ модели #1:
Интенсия: {intent_1}
Уверенность: {model_1_result.get("confidence")}
Объяснение: {model_1_result.get("rationale")}

Ответ модели #2:
Интенсия: {intent_2}
Уверенность: {model_2_result.get("confidence")}
Объяснение: {model_2_result.get("rationale")}

Выбери лучший ответ между моделью #1 и моделью #2.
"""

    data = call_openai_json(
        client=client,
        model=MODEL_JUDGE,
        instructions=instructions,
        user_prompt=prompt,
        schema_name="intent_judge",
        schema=judge_schema(candidate_intents),
    )

    final_intent = str(data.get("final_intent", "")).strip()

    if final_intent not in candidate_intents:
        raise ValueError(
            f"Судья выбрал интенсию вне двух кандидатов: {final_intent}. "
            f"Кандидаты: {candidate_intents}"
        )

    winner = str(data.get("winner", "")).strip()

    if intent_1 == intent_2:
        winner = "same_answer"

    if winner not in {"model_1", "model_2", "same_answer"}:
        winner = "same_answer" if intent_1 == intent_2 else "model_1"

    return {
        "final_intent": final_intent,
        "winner": winner,
        "confidence": clamp_probability(data.get("confidence", 0)),
        "rationale": str(data.get("rationale", "")).strip(),
    }


# ============================================================
# CSV / ЛОГИ
# ============================================================

CSV_COLUMNS = [
    # Колонки, нужные train.py
    "Phrase",
    "Intentionality",
    "PhraseWithContext",
    "ContextBefore",
    "ContextAfter",

    # Служебные колонки
    "SourceSentenceIndex",
    "Model1Name",
    "Model1Intent",
    "Model1Confidence",
    "Model1Rationale",
    "Model2Name",
    "Model2Intent",
    "Model2Confidence",
    "Model2Rationale",
    "JudgeModelName",
    "JudgeWinner",
    "JudgeIntent",
    "JudgeConfidence",
    "JudgeRationale",
]


def count_existing_rows(csv_path: str) -> int:
    path = Path(csv_path)

    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def open_csv_writer(csv_path: str, append: bool) -> tuple[Any, csv.DictWriter]:
    path = Path(csv_path)
    file_exists = path.exists()

    mode = "a" if append else "w"

    f = path.open(mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

    if not append or not file_exists:
        writer.writeheader()

    return f, writer


def append_debug_jsonl(path: str, data: Dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if len({MODEL_1, MODEL_2, MODEL_JUDGE}) != 3:
        raise ValueError("MODEL_1, MODEL_2 и MODEL_JUDGE должны быть разными моделями.")

    input_path = Path(INPUT_TEXT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной .txt файл: {INPUT_TEXT_PATH}")

    print("Загрузка списка интенсий...")
    intents = load_intents(INTENTS_TXT_PATH)
    print(f"Загружено интенсий: {len(intents)}")

    print("Загрузка художественного текста...")
    raw_text = input_path.read_text(encoding="utf-8")

    print("Нарезка текста на фразы...")
    sentences = extract_sentences_from_fiction_text(raw_text)

    if MAX_SENTENCES is not None:
        sentences = sentences[:MAX_SENTENCES]

    print(f"Выделено фраз: {len(sentences)}")

    if not sentences:
        raise ValueError("Не удалось выделить ни одной фразы из текста.")

    examples = make_examples(sentences)

    already_done = 0
    append_mode = False

    if RESUME_IF_OUTPUT_EXISTS and Path(OUTPUT_CSV_PATH).exists():
        already_done = count_existing_rows(OUTPUT_CSV_PATH)
        append_mode = already_done > 0

    if already_done:
        print(f"Найден существующий CSV. Уже обработано строк: {already_done}")
        print(f"Продолжаю с индекса: {already_done}")

    client = make_client()

    csv_file, writer = open_csv_writer(OUTPUT_CSV_PATH, append=append_mode)

    try:
        for idx, example in enumerate(examples):
            if idx < already_done:
                continue

            phrase = example["Phrase"]
            context_before = example["ContextBefore"]
            context_after = example["ContextAfter"]

            print("\n" + "=" * 80)
            print(f"Фраза {idx + 1}/{len(examples)}")
            print(f"Phrase: {phrase}")

            try:
                model_1_result = annotate_with_model(
                    client=client,
                    model=MODEL_1,
                    intents=intents,
                    phrase=phrase,
                    context_before=context_before,
                    context_after=context_after,
                )

                print(f"Модель #1 ({MODEL_1}): {model_1_result['intent']}")

                model_2_result = annotate_with_model(
                    client=client,
                    model=MODEL_2,
                    intents=intents,
                    phrase=phrase,
                    context_before=context_before,
                    context_after=context_after,
                )

                print(f"Модель #2 ({MODEL_2}): {model_2_result['intent']}")

                judge_result = judge_two_annotations(
                    client=client,
                    intents=intents,
                    phrase=phrase,
                    context_before=context_before,
                    context_after=context_after,
                    model_1_result=model_1_result,
                    model_2_result=model_2_result,
                )

                final_intent = judge_result["final_intent"]

                print(f"Судья ({MODEL_JUDGE}): {final_intent}")
                print(f"Победитель: {judge_result['winner']}")

                row = {
                    "Phrase": phrase,
                    "Intentionality": final_intent,
                    "PhraseWithContext": example["PhraseWithContext"],
                    "ContextBefore": context_before,
                    "ContextAfter": context_after,
                    "SourceSentenceIndex": example["SourceSentenceIndex"],

                    "Model1Name": MODEL_1,
                    "Model1Intent": model_1_result["intent"],
                    "Model1Confidence": model_1_result["confidence"],
                    "Model1Rationale": model_1_result["rationale"],

                    "Model2Name": MODEL_2,
                    "Model2Intent": model_2_result["intent"],
                    "Model2Confidence": model_2_result["confidence"],
                    "Model2Rationale": model_2_result["rationale"],

                    "JudgeModelName": MODEL_JUDGE,
                    "JudgeWinner": judge_result["winner"],
                    "JudgeIntent": judge_result["final_intent"],
                    "JudgeConfidence": judge_result["confidence"],
                    "JudgeRationale": judge_result["rationale"],
                }

                writer.writerow(row)
                csv_file.flush()

                append_debug_jsonl(
                    DEBUG_JSONL_PATH,
                    {
                        "index": idx,
                        "example": example,
                        "model_1_result": model_1_result,
                        "model_2_result": model_2_result,
                        "judge_result": judge_result,
                    },
                )

            except Exception as exc:
                print(f"Ошибка при обработке фразы #{idx}: {exc}")

                append_debug_jsonl(
                    DEBUG_JSONL_PATH,
                    {
                        "index": idx,
                        "example": example,
                        "error": str(exc),
                    },
                )

            time.sleep(SLEEP_BETWEEN_ITEMS_SECONDS)

    finally:
        csv_file.close()

    print("\n" + "=" * 80)
    print(f"Готово. Датасет сохранён в: {OUTPUT_CSV_PATH}")
    print(f"Лог сохранён в: {DEBUG_JSONL_PATH}")


if __name__ == "__main__":
    main()
