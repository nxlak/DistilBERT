# Классификатор интенсий на основе DistilBERT

Проект предназначен для обучения и использования нейросетевой модели, которая определяет интенсию пользовательского высказывания.  
Модель построена на базе `distilbert-base-uncased` и решает задачу мультиклассовой классификации: для каждой входной фразы выбирается одна наиболее вероятная интенсия.

---

## Структура проекта

Рекомендуемая структура файлов:

```text
project/
│
├── train.py
├── inference.py
├── dataset.csv
├── intent_list.txt


Файл `dataset.csv` используется для обучения.
Папка `distil_multiclass_model_ctx` создаётся автоматически после запуска обучения и содержит обученную модель, метрики и графики.

---

## Установка зависимостей

Перед запуском необходимо установить зависимости:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

Если используется видеокарта NVIDIA, рекомендуется установить версию `torch` с поддержкой CUDA.

---

## Формат датасета

Датасет должен храниться в CSV-файле.

По умолчанию `train.py` ищет файл датасета рядом со скриптом в следующем порядке:

```text
dataset.csv
dataset_cleaned.csv
dataset_.csv
```

Также путь к датасету можно передать явно при запуске.

### Обязательные колонки

В датасете обязательно должны быть следующие колонки:

| Колонка          | Описание                                     |
| ---------------- | -------------------------------------------- |
| `Phrase`         | Основная фраза / высказывание                |
| `Intentionality` | Название интенсии, к которой относится фраза |

### Необязательные колонки

Дополнительно могут использоваться колонки с контекстом:

| Колонка             | Описание                        |
| ------------------- | ------------------------------- |
| `PhraseWithContext` | Фраза вместе с контекстом       |
| `ContextBefore`     | Контекст до основной реплики    |
| `ContextAfter`      | Контекст после основной реплики |

Если в строке заполнены `ContextBefore` или `ContextAfter`, скрипт формирует текст для обучения в структурированном виде:

```text
[CTX] контекст до реплики [UTT] основная реплика [AFT] контекст после реплики
```

Если контекста нет, используется только значение из `Phrase`.

---

## Пример CSV-датасета

Пример файла `dataset.csv`:

```csv
Phrase,Intentionality,PhraseWithContext,ContextBefore,ContextAfter
"Could you explain this part again?","Seek clarification / curiosity","","We are discussing the project timeline.",""
"Thank you for helping me yesterday.","Express gratitude for something","","",""
"I think we should choose another solution.","Offer an opinion to a person","","The team is comparing two possible approaches.",""
"I am sorry for being late.","Apologize to a person","","",""
"If you need help, I can assist you.","Offer assistance to a person","","",""
```

Минимальный допустимый вариант датасета:

```csv
Phrase,Intentionality
"Could you explain this part again?","Seek clarification / curiosity"
"Thank you for helping me yesterday.","Express gratitude for something"
"I am sorry for being late.","Apologize to a person"
```

---

## Формат списка интенсий

Пример `intent_list.txt`:

```text
Admit a mistake to a person
Agree with another person
Apologize to a person
Ask for advice
Ask the other person for something
Empathy / emotional support
Express gratitude for something
Offer assistance to a person
Seek clarification / curiosity
```

Вручную создавать `intent_list.txt` перед обучением не требуется.
Список интенсий извлекается из колонки `Intentionality` в датасете и сохраняется вместе с моделью.

Файл `intent_list.txt` нужен для инференса как дополнительный способ восстановить список классов, если он не был найден в сохранённом чекпоинте модели.

---

## Обучение модели

### Запуск с датасетом по умолчанию

Если файл `dataset.csv` лежит рядом с `train.py`, обучение запускается командой:

```bash
python train.py
```

### Запуск с указанием пути к датасету

```bash
python train.py path/to/dataset.csv
```

Например:

```bash
python train.py data/my_dataset.csv
```

---

## Что происходит во время обучения

Скрипт `train.py` выполняет следующие шаги:

1. Загружает CSV-датасет.
2. Проверяет наличие обязательных колонок `Phrase` и `Intentionality`.
3. Формирует текст для обучения с учётом контекста.
4. Нормализует текст:

   * приводит к нижнему регистру;
   * заменяет нестандартные кавычки и тире;
   * удаляет лишние пробелы.
5. Удаляет пустые строки.
6. Удаляет классы, в которых меньше 3 примеров.
7. Разделяет данные на:

   * обучающую выборку;
   * валидационную выборку;
   * тестовую выборку.
8. Загружает токенизатор и модель DistilBERT.
9. Дообучает модель под задачу классификации интенсий.
10. Сохраняет лучшую модель по значению `Val Macro F1`.
11. Оценивает модель на тестовой выборке.
12. Сохраняет метрики, графики и результаты предсказаний.

---

## Основные параметры обучения

Основные параметры заданы внутри `train.py`:

```python
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "./distil_multiclass_model_ctx"

MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 5
FREEZE_EPOCHS = 1
MIN_SAMPLES_PER_CLASS = 3
```

Описание важных параметров:

| Параметр                  | Значение                                             |
| ------------------------- | ---------------------------------------------------- |
| `MODEL_NAME`              | Базовая модель DistilBERT                            |
| `MODEL_DIR`               | Папка для сохранения модели и результатов            |
| `MAX_LEN`                 | Максимальная длина входного текста в токенах         |
| `BATCH_SIZE`              | Размер батча                                         |
| `EPOCHS`                  | Максимальное количество эпох                         |
| `LEARNING_RATE`           | Скорость обучения                                    |
| `DROPOUT`                 | Вероятность отключения нейронов для регуляризации    |
| `EARLY_STOPPING_PATIENCE` | Через сколько эпох без улучшения остановить обучение |
| `MIN_SAMPLES_PER_CLASS`   | Минимальное количество примеров для класса           |

---

## Результаты обучения

После обучения в папке `distil_multiclass_model_ctx` сохраняются следующие файлы:

| Файл                         | Описание                                |
| ---------------------------- | --------------------------------------- |
| `best_model.pt`              | Лучшая сохранённая модель               |
| `intent_list.txt`            | Список интенсий                         |
| `training_history.csv`       | История обучения по эпохам              |
| `training_history.png`       | Графики изменения loss и метрик         |
| `test_results.csv`           | Предсказания модели на тестовой выборке |
| `metrics_test.json`          | Итоговые метрики на тестовой выборке    |
| `classification_report.json` | Подробный отчёт по каждому классу       |
| `confusion_matrix.png`       | Матрица ошибок в виде изображения       |
| `confusion_matrix.csv`       | Матрица ошибок в табличном виде         |
| `class_distribution.csv`     | Распределение примеров по классам       |
| `train_config.json`          | Конфигурация обучения                   |

---

## Использование модели для инференса

После обучения можно использовать `inference.py`.

Скрипт автоматически ищет модель в одной из папок:

```text
distil_multiclass_model_ctx
distil_multiclass_model
```

Также путь к модели можно указать вручную через параметр `--model-dir`.

---

## Инференс одной фразы

Пример запуска:

```bash
python inference.py --utterance "Could you explain this part again?"
```

Пример с контекстом:

```bash
python inference.py \
  --utterance "Could you explain this part again?" \
  --context "We are discussing the project timeline."
```

Пример с контекстом до и после реплики:

```bash
python inference.py \
  --utterance "I should like your opinion on that." \
  --context "People of my age don't really know anything about those times." \
  --context-after "We can only read about them in books."
```

---

## Инференс с указанием папки модели

```bash
python inference.py \
  --model-dir distil_multiclass_model_ctx \
  --utterance "Thank you for helping me yesterday."
```

---

## Вывод нескольких наиболее вероятных интенсий

По умолчанию скрипт выводит 5 наиболее вероятных интенсий.

Количество можно изменить параметром `--top-k`:

```bash
python inference.py \
  --utterance "Thank you for helping me yesterday." \
  --top-k 3
```

---

## Инференс из JSON-файла

Можно передать один объект или список объектов в JSON-файле.

Пример `input.json`:

```json
[
  {
    "utterance": "Could you explain this part again?",
    "context": "We are discussing the project timeline."
  },
  {
    "utterance": "Thank you for helping me yesterday.",
    "context": ""
  },
  {
    "utterance": "If you need help, I can assist you.",
    "context_before": "",
    "context_after": ""
  }
]
```

Запуск:

```bash
python inference.py --input-json input.json
```

---

## Формат входных данных для инференса

Для каждого примера можно использовать следующие поля:

| Поле             | Описание                                                 |
| ---------------- | -------------------------------------------------------- |
| `utterance`      | Основная реплика                                         |
| `context`        | Контекст до реплики                                      |
| `context_before` | Контекст до реплики                                      |
| `context_after`  | Контекст после реплики                                   |
| `Phrase`         | Альтернативное название поля для реплики                 |
| `ContextBefore`  | Альтернативное название поля для контекста до реплики    |
| `ContextAfter`   | Альтернативное название поля для контекста после реплики |

Если указано поле `context_before`, оно используется как контекст до реплики.
Если его нет, используется поле `context`.

---

## Пример результата инференса

Пример вывода:

```json
[
  {
    "utterance": "Could you explain this part again?",
    "context_before": "We are discussing the project timeline.",
    "context_after": null,
    "combined_text": "[ctx] we are discussing the project timeline. [utt] could you explain this part again?",
    "predicted_intent": "Seek clarification / curiosity",
    "predicted_probability": 0.8731,
    "top_k": 5,
    "predictions": [
      {
        "intent": "Seek clarification / curiosity",
        "probability": 0.8731
      },
      {
        "intent": "Ask for advice",
        "probability": 0.0524
      },
      {
        "intent": "Ask the other person for something",
        "probability": 0.0318
      }
    ]
  }
]
```

Главные поля результата:

| Поле                    | Описание                                     |
| ----------------------- | -------------------------------------------- |
| `predicted_intent`      | Интенсия, выбранная моделью                  |
| `predicted_probability` | Вероятность выбранной интенсии               |
| `predictions`           | Список наиболее вероятных интенсий           |
| `combined_text`         | Итоговый текст, который был передан в модель |

---

## Особенности работы с контекстом

Если у фразы есть контекст, он добавляется к входному тексту с помощью специальных токенов:

```text
[CTX] — контекст до реплики
[UTT] — основная реплика
[AFT] — контекст после реплики
```

Например:

```text
[CTX] The team is discussing the deadline. [UTT] Could you explain this part again? [AFT] The speaker looks confused.
```

Эти специальные токены добавляются в токенизатор во время обучения и сохраняются вместе с моделью.

---

## Важные замечания

1. В датасете каждая строка должна соответствовать одной фразе и одной интенсии.
2. Названия интенсий в колонке `Intentionality` должны быть написаны единообразно.
3. Классы, в которых меньше 3 примеров, автоматически удаляются перед обучением.
4. Если классы сильно несбалансированы, модель может хуже распознавать редкие интенсии.
5. Для корректного инференса необходимо использовать ту же папку модели, которая была создана после обучения.
6. Файл `intent_list.txt` должен находиться внутри папки модели, например:

```text
distil_multiclass_model_ctx/intent_list.txt
```

7. Основной файл модели должен называться:

```text
best_model.pt
```

---

## Быстрый пример полного запуска

### 1. Подготовить датасет

Создать файл:

```text
dataset.csv
```

рядом с `train.py`.

Минимальный пример:

```csv
Phrase,Intentionality
"Could you explain this part again?","Seek clarification / curiosity"
"Thank you for your help.","Express gratitude for something"
"I am sorry for the mistake.","Apologize to a person"
"If you need help, I can assist you.","Offer assistance to a person"
```

### 2. Запустить обучение

```bash
python train.py
```

### 3. Проверить, что появилась папка модели

```text
distil_multiclass_model_ctx/
```

Внутри неё должны быть:

```text
best_model.pt
intent_list.txt
metrics_test.json
training_history.png
```

### 4. Запустить инференс

```bash
python inference.py --utterance "Could you explain this part again?"
```

### 5. Получить предсказание

Скрипт выведет JSON с наиболее вероятной интенсией и списком top-k предсказаний.

---

## Назначение файлов

### `train.py`

Скрипт для обучения модели.
Он принимает CSV-датасет, обучает классификатор и сохраняет модель вместе с результатами оценки.

### `inference.py`

Скрипт для применения обученной модели.
Он принимает одну фразу, фразу с контекстом или JSON-файл с несколькими примерами и возвращает предсказанную интенсию.

```
```
