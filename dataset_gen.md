# Генерация датасета интенсий из художественного текста

Скрипт `generate_dataset.py` предназначен для автоматической генерации CSV-датасета для обучения классификатора интенсий на основе художественного текста.

На вход подаётся `.txt` файл с художественным текстом и `.txt` файл со списком интенсий.  
Скрипт выделяет из текста отдельные фразы, добавляет к каждой фразе контекст до и после, затем размечает фразы с помощью трёх разных моделей ChatGPT по методу судей.

Итоговый результат сохраняется в `dataset.csv`, который можно сразу использовать для обучения модели через `train.py`.

## Назначение скрипта

Скрипт решает задачу автоматической разметки данных.

Он выполняет следующие действия:

1. Загружает художественный текст из `.txt` файла.
2. Делит сплошной текст на отдельные фразы / предложения.
3. Для каждой фразы определяет:
   - само высказывание;
   - предложение до него;
   - предложение после него.
4. Загружает список допустимых интенсий.
5. Отправляет фразу и контекст в модель №1.
6. Отправляет ту же фразу и контекст в модель №2.
7. Передаёт ответы моделей №1 и №2 в модель №3.
8. Модель №3 выступает судьёй и выбирает наиболее подходящий ответ.
9. Сохраняет результат в CSV-файл.

---

## Структура проекта

Рекомендуемая структура файлов:

```text
project/
│
├── generate_dataset.py
├── source_text.txt
├── intent_list.txt
├── dataset.csv
└── annotation_debug.jsonl
````

Описание файлов:

| Файл                     | Назначение                                    |
| ------------------------ | --------------------------------------------- |
| `generate_dataset.py`    | Основной скрипт генерации датасета            |
| `source_text.txt`        | Входной художественный текст                  |
| `intent_list.txt`        | Список допустимых интенсий                    |
| `dataset.csv`            | Итоговый датасет                              |
| `annotation_debug.jsonl` | Технический лог с ответами моделей и ошибками |

---

## Установка зависимостей

Перед запуском необходимо установить зависимости:

```bash
pip install --upgrade openai pandas
```

---

## Настройка API-ключа

API-ключ задаётся прямо в коде в переменной:

```python
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
```

Нужно заменить строку `PASTE_YOUR_OPENAI_API_KEY_HERE` на свой ключ OpenAI.

Пример:

```python
OPENAI_API_KEY = "sk-..."
```

---

## Используемые модели

В скрипте используются три разные модели:

```python
MODEL_1 = "gpt-5.4-mini"
MODEL_2 = "gpt-5.4-nano"
MODEL_JUDGE = "gpt-5.5"
```

Логика работы:

| Модель        | Роль                    |
| ------------- | ----------------------- |
| `MODEL_1`     | Первая модель-разметчик |
| `MODEL_2`     | Вторая модель-разметчик |
| `MODEL_JUDGE` | Модель-судья            |

Модель №1 и модель №2 независимо выбирают наиболее подходящую интенсию.
Модель №3 получает оба ответа и выбирает лучший из них.

---

## Входной художественный текст

Входной текст должен находиться в файле:

```text
source_text.txt
```

Пример содержимого:

```text
He looked at her for a long time and then said, "I am sorry for what happened yesterday."

She did not answer at once. The room was quiet.

"Could you explain why you left so suddenly?" she asked.
```

Текст может быть сплошным.
Допускаются:

* переносы строк внутри предложений;
* диалоги с тире;
* кавычки;
* многоточия;
* вопросительные и восклицательные знаки;
* точка с запятой;
* абзацы.

Скрипт старается аккуратно обрабатывать художественный текст и не разрывать предложения в неправильных местах.

---

## Формат списка интенсий

Список интенсий должен находиться в файле:

```text
intent_list.txt
```

Формат: одна интенсия на строку.

Пример:

```text
Provide a person with information
Motivate a person for something
Ask the other person for something
Empathy / emotional support
Engage in small talk with a person
Express a polite disagreement with the interlocutor
Offer an opinion to a person
Admit a mistake to a person
Agree with another person
Clarify a complex idea for another person
Greet the other person
Seek clarification / curiosity
Apologize to a person
Give a compliment to a person
Ask for a persons opinion
Use humor in a conversation
Defend a person from something
Offer assistance to a person
Share feelings / thoughts
Express gratitude for something
Set personal boundaries clearly
Stand by one's opinion
Allow a person to make choices
Adopt a formal tone in a conversation
Express a desire to communicate
Ask for advice
Give a person constructive advice
Demonstrate politeness to another person
Express trust in a person
End the conversation in a polite way
Hold a person accountable
Persuade a person to change their perspective
Acknowledge the other person's point
Express willingness to compromise
Share positive emotions with the interlocutor
Contribute related ideas to the discussion
```

Пустые строки игнорируются.
Строки, начинающиеся с `#`, также игнорируются.

Пример с комментариями:

```text
# Основные коммуникативные интенсии
Ask for advice
Offer assistance to a person
Express gratitude for something
Seek clarification / curiosity
Apologize to a person
```

---

## Настройки внутри скрипта

Все параметры задаются прямо в `generate_dataset.py`.
Параметры командной строки не используются.

Основные настройки:

```python
INPUT_TEXT_PATH = "source_text.txt"
INTENTS_TXT_PATH = "intent_list.txt"
OUTPUT_CSV_PATH = "dataset.csv"
DEBUG_JSONL_PATH = "annotation_debug.jsonl"
```

| Параметр           | Значение                               |
| ------------------ | -------------------------------------- |
| `INPUT_TEXT_PATH`  | Путь к входному художественному тексту |
| `INTENTS_TXT_PATH` | Путь к списку интенсий                 |
| `OUTPUT_CSV_PATH`  | Путь к итоговому CSV-файлу             |
| `DEBUG_JSONL_PATH` | Путь к техническому логу               |

---

## Ограничение количества фраз

Для тестового запуска можно ограничить количество обрабатываемых фраз:

```python
MAX_SENTENCES = 20
```

Тогда будет обработано только первые 20 фраз.

Для обработки всего текста нужно указать:

```python
MAX_SENTENCES = None
```

---

## Деление по точке с запятой

В художественных текстах длинные высказывания могут разделяться не только точками, но и точками с запятой.

За это отвечает параметр:

```python
SPLIT_ON_SEMICOLON = True
```

Если значение `True`, скрипт может разделять текст по `;`.

Если значение `False`, точка с запятой не будет считаться границей фразы.

---

## Запуск скрипта

После подготовки файлов нужно запустить:

```bash
python generate_dataset.py
```

Скрипт не требует параметров командной строки.

---

## Пример полного запуска

### 1. Подготовить `source_text.txt`

```text
"Could you explain what you meant by that?" he asked.

She smiled and said, "Thank you for helping me yesterday."

"I am sorry," he whispered. "I should have told you the truth earlier."
```

### 2. Подготовить `intent_list.txt`

```text
Seek clarification / curiosity
Express gratitude for something
Apologize to a person
Offer assistance to a person
Share feelings / thoughts
```

### 3. Указать API-ключ в коде

```python
OPENAI_API_KEY = "sk-..."
```

### 4. Запустить генерацию

```bash
python generate_dataset.py
```

### 5. Получить результат

После выполнения появится файл:

```text
dataset.csv
```

---

## Формат итогового CSV-файла

Итоговый файл `dataset.csv` совместим с `train.py`.

Главные колонки:

| Колонка             | Описание                                           |
| ------------------- | -------------------------------------------------- |
| `Phrase`            | Основная фраза                                     |
| `Intentionality`    | Итоговая интенсия, выбранная моделью-судьёй        |
| `PhraseWithContext` | Фраза вместе с контекстом в структурированном виде |
| `ContextBefore`     | Предложение перед основной фразой                  |
| `ContextAfter`      | Предложение после основной фразы                   |

Эти колонки используются при обучении модели.

---

## Пример итогового CSV

```csv
Phrase,Intentionality,PhraseWithContext,ContextBefore,ContextAfter
"Could you explain what you meant by that?","Seek clarification / curiosity","[CTX] She looked confused. [UTT] Could you explain what you meant by that? [AFT] He waited for her answer.","She looked confused.","He waited for her answer."
"Thank you for helping me yesterday.","Express gratitude for something","[CTX] She smiled warmly. [UTT] Thank you for helping me yesterday. [AFT] He nodded in response.","She smiled warmly.","He nodded in response."
```

---

## Служебные колонки

Кроме основных колонок, скрипт сохраняет дополнительные данные:

| Колонка               | Описание                           |
| --------------------- | ---------------------------------- |
| `SourceSentenceIndex` | Номер фразы в исходном тексте      |
| `Model1Name`          | Название первой модели             |
| `Model1Intent`        | Интенсия, выбранная первой моделью |
| `Model1Confidence`    | Уверенность первой модели          |
| `Model1Rationale`     | Объяснение первой модели           |
| `Model2Name`          | Название второй модели             |
| `Model2Intent`        | Интенсия, выбранная второй моделью |
| `Model2Confidence`    | Уверенность второй модели          |
| `Model2Rationale`     | Объяснение второй модели           |
| `JudgeModelName`      | Название модели-судьи              |
| `JudgeWinner`         | Какая модель выбрана судьёй        |
| `JudgeIntent`         | Итоговая интенсия судьи            |
| `JudgeConfidence`     | Уверенность судьи                  |
| `JudgeRationale`      | Объяснение судьи                   |

Эти колонки не мешают работе `train.py`, так как он использует только нужные ему поля.

---

## Формат `PhraseWithContext`

Поле `PhraseWithContext` формируется с использованием специальных токенов:

```text
[CTX] контекст до фразы [UTT] основная фраза [AFT] контекст после фразы
```

Пример:

```text
[CTX] She looked confused. [UTT] Could you explain what you meant by that? [AFT] He waited for her answer.
```

Эти же специальные токены используются в `train.py` и `inference.py`.

---

## Как работает метод судей

Для каждой фразы выполняются три запроса:

### Шаг 1. Первая модель

Модель №1 получает:

* список интенсий;
* контекст до фразы;
* основную фразу;
* контекст после фразы.

Она выбирает одну наиболее подходящую интенсию.

### Шаг 2. Вторая модель

Модель №2 получает тот же запрос и также выбирает одну интенсию.

### Шаг 3. Модель-судья

Модель №3 получает:

* исходную фразу;
* контекст;
* ответ модели №1;
* ответ модели №2.

Судья должен выбрать только один из двух предложенных вариантов.
Он не может выбрать третью интенсию.

Именно ответ судьи сохраняется в колонку:

```text
Intentionality
```

---

## Продолжение после остановки

Если `dataset.csv` уже существует, скрипт может продолжить работу с места остановки.

За это отвечает параметр:

```python
RESUME_IF_OUTPUT_EXISTS = True
```

При повторном запуске скрипт считает, сколько строк уже есть в `dataset.csv`, и продолжает обработку со следующей фразы.

Это удобно, если:

* обработка была прервана;
* закончился лимит API;
* возникла временная ошибка сети;
* нужно продолжить генерацию позже.

---

## Технический лог

Дополнительно создаётся файл:

```text
annotation_debug.jsonl
```

В него записываются:

* исходная фраза;
* контекст;
* ответ модели №1;
* ответ модели №2;
* ответ судьи;
* ошибки, если они возникли.

Формат `.jsonl` означает, что каждая строка является отдельным JSON-объектом.

Этот файл нужен для проверки качества разметки и отладки.

---

## Обработка ошибок

Если при обработке отдельной фразы возникает ошибка, скрипт:

1. выводит сообщение об ошибке в консоль;
2. записывает ошибку в `annotation_debug.jsonl`;
3. переходит к следующей фразе.

То есть ошибка на одной фразе не останавливает всю генерацию датасета.

---

## Повторные попытки API

При временной ошибке API скрипт делает несколько попыток.

Настройки:

```python
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP_SECONDS = 2
```

Это означает, что при ошибке запрос будет повторён до 3 раз.

---

## Пауза между запросами

Чтобы снизить риск превышения лимитов API, используется небольшая пауза между обработкой фраз:

```python
SLEEP_BETWEEN_ITEMS_SECONDS = 0.2
```

При необходимости значение можно увеличить, например:

```python
SLEEP_BETWEEN_ITEMS_SECONDS = 1.0
```

---

## Проверка результата перед обучением

После генерации рекомендуется открыть `dataset.csv` и проверить:

1. Корректно ли выделены фразы.
2. Нет ли слишком коротких или бессмысленных фрагментов.
3. Соответствуют ли интенсии смыслу фраз.
4. Нет ли сильного перекоса в сторону одной интенсии.
5. Заполнены ли колонки:

   * `Phrase`;
   * `Intentionality`;
   * `ContextBefore`;
   * `ContextAfter`;
   * `PhraseWithContext`.

---

## Использование результата для обучения

После генерации датасета можно запустить обучение:

```bash
python train.py dataset.csv
```

Файл `dataset.csv` уже содержит обязательные колонки, которые ожидает `train.py`:

```text
Phrase
Intentionality
PhraseWithContext
ContextBefore
ContextAfter
```

---

## Типичный порядок работы

1. Подготовить художественный текст.
2. Сохранить его в `source_text.txt`.
3. Подготовить список интенсий.
4. Сохранить его в `intent_list.txt`.
5. Указать API-ключ в `generate_dataset.py`.
6. При необходимости ограничить количество фраз через `MAX_SENTENCES`.
7. Запустить:

```bash
python generate_dataset.py
```

8. Проверить `dataset.csv`.
9. Использовать его для обучения:

```bash
python train.py dataset.csv
```

---

## Возможные проблемы

### Скрипт пишет, что API-ключ не указан

Нужно заменить строку:

```python
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
```

на реальный API-ключ.

---

### Не найден `source_text.txt`

Файл с художественным текстом должен лежать рядом со скриптом или путь к нему должен быть указан в переменной:

```python
INPUT_TEXT_PATH = "source_text.txt"
```

---

### Не найден `intent_list.txt`

Файл со списком интенсий должен лежать рядом со скриптом или путь к нему должен быть указан в переменной:

```python
INTENTS_TXT_PATH = "intent_list.txt"
```

---

### В CSV мало строк

Проверь значение:

```python
MAX_SENTENCES
```

Если там стоит число, например `20`, будет обработано только 20 фраз.

Для полной обработки нужно указать:

```python
MAX_SENTENCES = None
```

---

### Много ошибок API

Можно увеличить паузу между запросами:

```python
SLEEP_BETWEEN_ITEMS_SECONDS = 1.0
```

Также можно увеличить количество повторных попыток:

```python
RETRY_ATTEMPTS = 5
```

---

### Фразы выделяются слишком мелко

Можно отключить деление по точке с запятой:

```python
SPLIT_ON_SEMICOLON = False
```

---

### Фразы выделяются слишком крупно

Можно оставить:

```python
SPLIT_ON_SEMICOLON = True
```

Также можно дополнительно доработать правила нарезки текста в функции:

```python
split_block_into_sentences()
```

---
```
