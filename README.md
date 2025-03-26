(Due to technical issues, the search service is temporarily unavailable.)

# Обзор файлов для реализации DistilBERT в анализе интенций

## Установка и настройка окружения

### 1. Установка Jupyter Notebook
```bash
# Установка через pip
pip install jupyterlab

# Запуск Jupyter Notebook
jupyter notebook
```

### 2. Установка библиотек
```bash
pip install pandas numpy matplotlib sklearn tqdm torch torchmetrics transformers
```

---

## Файл #1: `preprocessing2.ipynb`
**Общая суть**: Многоцелевая модель для классификации интенций с использованием:
- DistilBERT в качестве базовой модели
- 52 независимых классификаторов ("голов") 
- Кастомной функции потерь
- Балансировки данных через преобразование меток

---

### In[1]: Импорт библиотек
```python
import pandas as pd
from sklearn.model_selection import train_test_split
...
from torchmetrics import Accuracy
```
**Что делает**:
- Импортирует необходимые компоненты:
  - `pandas` для работы с данными
  - `sklearn` для разделения данных
  - `transformers` для работы с DistilBERT
  - `torch` для нейросетей
  - `torchmetrics` для расчета точности

---

### In[2]-In[4]: Загрузка и разделение данных
```python
df1 = pd.read_csv("data.csv")
df_train, df_test = train_test_split(df1, test_size=0.4)
```
**Что делает**:
1. Загружает основной датасет (`data.csv`)
2. Разделяет данные на:
   - 60% тренировочных
   - 40% тестовых (из них 70% становятся валидационными)

---

### In[5]-In[6]: Анализ данных
```python
df_train.describe()
df_train['Intentionality'].hist()
```
**Что делает**:
- Показывает статистику по данным
- Строит гистограмму распределения меток

---

### In[7]: Преобразование меток
```python
def tint_coding(...):
    result = [0] * n
    ...
```
**Что делает**:
- Преобразует текстовые метки в вектор из:
  - `1` для целевого класса
  - `-1` для антонима
  - `0` для остальных

---

### In[8]: Dataset класс
```python
class TorchSet(...):
    def __getitem__(self, idx):
        ...
```
**Что делает**:
- Создает кастомный Dataset для PyTorch
- Токенизирует текст при инициализации
- Возвращает словарь с:
  - `input_ids` — индексы токенов
  - `attention_mask` — маска внимания
  - `labels` — вектор меток

---

### In[10]-In[12]: Архитектура модели
```python
class Classifier(...):
    # 3-х слойная сеть
    ...

class Tint(...):
    # 52 независимых классификатора
    ...
```
**Особенности**:
- Средний пулинг выхода BERT
- Инициализация Хе и Xavier
- Dropout для регуляризации

---

### In[13]-In[15]: Функция потерь и обучение
```python
class TintLoss(...):
    def forward(...):
        return torch.log(...)

def model_train(...):
    # Цикл обучения на GPU
```
**Особенности**:
- Кастомная функция потерь для мультиклассовой классификации
- Clip gradients на 30.0
- Расчет accuracy для каждой "головы"

---

### In[17]-In[18]: Инициализация модели
```python
model = DistilBertModel.from_pretrained(...)
tokenizer = DistilBertTokenizer(...)
```
**Важно**:
- Используется предобученная версия `distilbert-base-uncased`
- Батч размер 50 для DataLoader

---

## Файл #2: `preprocessing4.ipynb`
**Основные отличия**:
- Балансировка данных через добавление "ложных" примеров
- Отдельные DataLoader'ы для каждой "головы"
- Индивидуальные оптимизаторы для каждой головы
- Поочередное обучение голов с перемешиванием

---

### In[7]: Новый метод балансировки
```python
false_labels = cur_df[...].sample(len(true_labels)*4)
```
**Что делает**:
- Добавляет в 4 раза больше "нулевых" примеров
- Балансирует классы для каждой пары антонимов

---

### In[16]: Процесс обучения
```python
for i in idx_list:  # Перемешанный порядок голов
    optimizers[i].zero_grad()
    ...
```
**Особенности**:
- Разные learning rate для BERT (0.01) и голов (0.1)
- Градиенты обновляются только для текущей головы
- Шаффл порядка обработки голов

---

### In[19]: Параметры обучения
```python
param = {'num_epoch': 1, 'lr': 0.1}
```

---

