#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
import nlpaug.augmenter.word as naw
import nltk

# Настройки
nltk.download('wordnet', quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# ========== КОНФИГУРАЦИЯ ===========
SEED = 42
MAX_LEN = 128
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 35
MODEL_DIR = "./bert_model"
MODEL_NAME = 'bert-base-uncased'
EARLY_STOPPING_PATIENCE = 7 
AUGMENTATION_FACTOR = 4  
LEARNING_RATE = 2e-5
NUM_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01  
MIN_CLASS_WEIGHT = 0.5  

torch.manual_seed(SEED)
np.random.seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

# ========== Загрузка и подготовка данных ===========
def load_data(path="correct_data.xlsx"):
    df = pd.read_excel(path, engine="openpyxl")
    column_mapping = {
        'Example': 'Phrase',
        'Intension': 'Intentionality',
        'Sentence': 'Phrase',
        'Label': 'Intentionality'
    }
    for old, new in column_mapping.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    required_columns = ['Phrase', 'Intentionality']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Отсутствуют обязательные колонки: {missing}")
    
    df = df[required_columns].dropna()
    df['label_idx'], intents = pd.factorize(df['Intentionality'])
    return df, intents.tolist()

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s,.!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def augment_text(text, num_aug=3):
    augmented = []
    augmenters = [
        naw.SynonymAug(aug_src='wordnet'),
        naw.RandomWordAug(action="swap"),
        naw.RandomWordAug(action="delete", aug_p=0.1),
        naw.RandomWordAug(action="crop", aug_p=0.1), 
    ]
    
    try:
        bert_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            action="substitute",
            device=device.type
        )
        augmenters.append(bert_aug)
    except Exception as e:
        print(f"Предупреждение: BERT-аугментация недоступна: {e}")
    
    unique_texts = set()
    attempts = 0
    max_attempts = num_aug * 2
    
    while len(unique_texts) < num_aug and attempts < max_attempts:
        aug = np.random.choice(augmenters)
        try:
            aug_text = aug.augment(text)
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            
            aug_text = normalize_text(aug_text)
            if aug_text != text and aug_text not in unique_texts:
                unique_texts.add(aug_text)
        except Exception as e:
            print(f"Ошибка аугментации: {e}")
        attempts += 1
    
    return list(unique_texts)

def balance_dataset(df, min_samples=50, max_multiplier=2.0):
    augmented = []
    class_counts = df['label_idx'].value_counts()
    max_count = class_counts.max()
    max_augmented = int(max_count * max_multiplier)
    
    for label, count in class_counts.items():
        if count < min_samples:
            subset = df[df.label_idx == label]
            augmented_samples = []
            
            for _, row in subset.iterrows():
                augmented_texts = augment_text(row['Phrase'], num_aug=AUGMENTATION_FACTOR)
                for aug_text in augmented_texts:
                    augmented_samples.append({
                        'Phrase': aug_text,
                        'label_idx': label
                    })
            
            num_needed = min(min_samples - count, max_augmented - count, len(augmented_samples))
            augmented.extend(augmented_samples[:num_needed])
            
            if num_needed < min_samples - count:
                additional = min_samples - count - num_needed
                extra_samples = subset.sample(n=additional, replace=True)
                for _, row in extra_samples.iterrows():
                    augmented.append({
                        'Phrase': row['Phrase'],
                        'label_idx': label
                    })
    
    return pd.concat([df, pd.DataFrame(augmented)])

# ========== Улучшенная модель BERT ===========
class AdvancedBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout1 = nn.Dropout(dropout)
        
        hidden_size = bert_model.config.hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        for layer in self.bert.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        pooled_output = outputs.pooler_output
        x = self.dropout1(pooled_output)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.classifier(x)
        return logits

# ========== Датасет с сохранением текстов ===========
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'index': idx
        }

# ========== Визуализация ===========
def plot_training_history(history, path):
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Accuracy', color='blue')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # График F1-score
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='F1 Score', color='green')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, classes, path):
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='coolwarm',
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": 8, "color": "black", "alpha": 0.7}
    )
    
    plt.title('Confusion Matrix', fontsize=18, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    if len(classes) > 50:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.figtext(
            0.5, 0.01, 
            f"Total classes: {len(classes)} | Matrix shows counts of predictions",
            ha="center", 
            fontsize=12, 
            bbox={"facecolor": "orange", "alpha": 0.1, "pad": 5}
        )
    else:
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    cm_path = path.replace('.png', '.csv')
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_path)
    print(f"Числовая матрица сохранена в: {cm_path}")

# ========== Обучение и оценка ===========
def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Загрузка данных
    print("Загрузка данных...")
    df, intents = load_data()
    num_classes = len(intents)
    print(f"Найдено {num_classes} классов: {', '.join(intents[:5])}...")  
    
    print("Нормализация текста...")
    df['Phrase'] = df['Phrase'].apply(normalize_text)
    
    print("Балансировка классов...")
    df = balance_dataset(df, min_samples=50)
    print(f"Размер датасета после балансировки: {len(df)} примеров")
    
    # Разделение данных
    print("Разделение данных...")
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df['label_idx']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df['label_idx']
    )
    print(f"Размеры выборок: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Инициализация BERT
    print("Загрузка токенизатора и модели BERT...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME)
    
    # Создание датасетов
    print("Создание датасетов...")
    train_dataset = BERTDataset(
        texts=train_df['Phrase'].tolist(),
        labels=train_df['label_idx'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_dataset = BERTDataset(
        texts=val_df['Phrase'].tolist(),
        labels=val_df['label_idx'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_dataset = BERTDataset(
        texts=test_df['Phrase'].tolist(),
        labels=test_df['label_idx'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    
    # Инициализация модели
    print("Инициализация модели...")
    model = AdvancedBERTClassifier(
        bert_model=bert_model,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)
    
    all_labels = train_df['label_idx'].values
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(all_labels), 
        y=all_labels
    )
    class_weights = np.maximum(class_weights, MIN_CLASS_WEIGHT)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Обучение
    print("\n" + "="*50)
    print("Начало обучения")
    print("="*50)
    print(f"Используется смешанная точность: {'Да' if device.type == 'cuda' else 'Нет'}")
    
    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'intents': intents
            }, os.path.join(MODEL_DIR, 'best_model.pt'))
            print(f"Модель сохранена (лучший F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Ранняя остановка на эпохе {epoch}")
                break
    
    print("\nЗагрузка лучшей модели для тестирования...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    intents = checkpoint['intents']
    
    # Оценка на тестовом наборе
    model.eval()
    all_preds, all_labels, all_texts = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Тестирование"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            indices = batch['index'].cpu().numpy()
            batch_texts = [test_dataset.texts[i] for i in indices]
            all_texts.extend(batch_texts)
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=intents)
    
    print("\n" + "="*50)
    print("Финальные метрики модели")
    print("="*50)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    results_df = pd.DataFrame({
        'text': all_texts,
        'true_label': [intents[i] for i in all_labels],
        'predicted_label': [intents[i] for i in all_preds],
        'correct': [a == b for a, b in zip(all_labels, all_preds)]
    })
    
    results_df.to_csv(os.path.join(MODEL_DIR, 'test_results.csv'), index=False)
    
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(os.path.join(MODEL_DIR, 'training_history.csv'), index=False)
    
    with open(os.path.join(MODEL_DIR, 'intent_list.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(intents))
    
    plot_training_history(history, os.path.join(MODEL_DIR, 'training_history.png'))
    #plot_confusion_matrix(all_labels, all_preds, intents, os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    print(f"\nВсе результаты сохранены в {MODEL_DIR}")

if __name__ == '__main__':
    train_model()