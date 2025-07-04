#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Константы ======
MODEL_DIR = "./bert_model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Тексты для тестирования
TEXTS = [
    "Hello! How are you? Great to see you!",
    "Let me know how I can help you.",
    "You are truly talented at what you do. I admire your skills!",
    "You have already achieved so much, and I believe you can handle this. You are stronger than you think!",
    "Remember, every step towards your goal is already a success. You can achieve everything you set your mind to!",
    "I'm always happy to chat with you! Don't hesitate to share your thoughts.",
    "I'm here for you. If you need to talk, I'm ready to listen.",
    "Don't worry; everything will be okay! I believe in you.",
    "You're on the right track! Keep it up, and you'll definitely succeed!",
    "I'm with you and ready to support you in this.",
    "I'm really sorry if I hurt you. It was unintentional.",
    "I understand how hard it is for you right now. You're not alone in this.",
    "I understand your point of view, but let's consider this option as well.",
    "I'm ready to forgive you. Let's move forward.",
    "If you need help, just let me know—I'm happy to assist!",
    "If you need support or help, just say so.",
    "Let's try to find a solution that works for both of us.",
    "Try looking at it from a different angle—it might open new opportunities.",
    "I'm happy to help you grow in this area. Let's discuss how I can assist.",
    "Did you know there are several resources that can help you with this?",
    "In my opinion, this could be a good idea because...",
    "I feel so good today! I hope you're in a great mood too!",
    "I value your opinion and would love to hear more about what you think.",
    "You did an amazing job! This is truly impressive.",
    "Thank you for your help! I really appreciate it.",
    "I'm so excited about what we're going to do! It's going to be amazing!",
    "Let me explain this differently; it might make it clearer.",
    "Every detail matters; let's pay attention to them.",
    "Maybe we should think about how this has affected your situation?",
    "You did a great job on this project, but maybe we should pay attention to...",
    "I'm genuinely interested in learning more about your experience in this area.",
    "I've made a decision on this matter, and I believe it's the best way forward.",
    "Let's try to find a solution that works for both of us.",
    "Don't worry; I'll wait until you're ready to talk.",
    "I understand other perspectives, but I still believe my position is correct.",
    "I believe everything will turn out well! We have every chance of success.",
    "You're right about this aspect; it's really important to consider it.",
    "I'm interested to hear your opinion on this matter. What do you think?",
    "Maybe we should try this approach? It often works in similar situations.",
    "Let me explain each step of this process in detail.",
    "I'm here for you to support you in tough times.",
    "I respect your space and won't intrude.",
    "I'm ready to assist you at any time."
]

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

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s,.!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_model_and_tokenizer():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    bert_model = BertModel.from_pretrained(MODEL_NAME)
    
    model = AdvancedBERTClassifier(
        bert_model=bert_model,
        num_classes=len(checkpoint['intents']),
        dropout=0.3
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, checkpoint['intents']

def predict_single_text(model, tokenizer, text):
    text = normalize_text(text)
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
    
    return probs

def predict_batch(model, tokenizer, texts):
    results = []
    for text in tqdm(texts, desc="Обработка текстов"):
        probs = predict_single_text(model, tokenizer, text)
        results.append(probs)
    return np.array(results)

def main():
    model, tokenizer, intent_list = load_model_and_tokenizer()
    print(f"Модель загружена. Доступно {len(intent_list)} интентов")
    print(f"Устройство: {DEVICE}")
    
    print(f"\nОбработка {len(TEXTS)} текстов...")
    probs = predict_batch(model, tokenizer, TEXTS)
    
    results_rows = []
    
    print("\n" + "="*50)
    print("Результаты классификации:")
    print("="*50)
    
    for i, (text, row) in enumerate(zip(TEXTS, probs)):
        top10 = np.argsort(-row)[:10]
        
        for rank, idx in enumerate(top10, 1):
            results_rows.append({
                "text_id": i+1,
                "text": text,
                "intent": intent_list[idx],
                "probability": row[idx],
                "rank": rank
            })
        
        print(f"\n=== Текст #{i+1} ===")
        print(f"«{text}»")
        print("\nТоп-10 предсказаний:")
        for j, idx in enumerate(top10, 1):
            print(f"{j:2}. {intent_list[idx]:<50} {row[idx]:.4f}")
    
    results_df = pd.DataFrame(results_rows)

    results_path = os.path.join(MODEL_DIR, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nПолные результаты сохранены в: {results_path}")

if __name__ == "__main__":
    from tqdm.auto import tqdm 
    main()