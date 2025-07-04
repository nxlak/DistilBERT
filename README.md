# DistilBERT

* **train.py** — скрипт для подготовки данных, обучения и валидации модели, сохранения результатов.
* **inference.py** — скрипт для загрузки обученной модели и классификации новых текстов.

---

### Требования

* Python 3.8+
* CUDA (для работы на GPU) или CPU

#### Необходимые пакеты

```
pandas
numpy
torch torchvision torchaudio
transformers
scikit-learn
nlpaug
nltk
tqdm
openpyxl
matplotlib
seaborn
```

## 1. Подготовка данных

Положите файл `correct_data.xlsx` и `correct_ant.xlsx` в корень проекта.
* `correct_data.xlsx` - датасет
* `correct_ant.xlsx` - список антонимов

---

## 2. Обучение модели

Запустите скрипт `train.py`:

```bash
python train.py
```

   * `training_history.csv` и `training_history.png` (график потерь и точности).
   * `test_results.csv` с метриками на test выборке.
   * `intent_list.txt` — список всех меток.

Результаты будут сохранены в папке `bert_model/`.

---

## 3. Инференс

С помощью скрипта `inference.py` можно классифицировать произвольный набор текстов.

```bash
python inference.py
```

Что делает `inference.py`:

1. Загружает `best_model.pt` из `bert_model/`.
2. Загружает токенизатор и строит модель `AdvancedBERTClassifier`.
3. Приводит входные тексты к нормализованному виду.
4. Для каждого текста выводит Top‑10 предсказаний с вероятностями.
5. Сохраняет полный DataFrame результатов в `bert_model/inference_results.csv`.

Чтобы классифицировать свои тексты, отредактируйте список `TEXTS` в начале скрипта или замените его на чтение из файла.

---
