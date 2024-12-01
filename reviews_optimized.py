import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import gradio as gr

# Логування завантаження даних
print("Завантаження даних...")
train_data = pd.read_csv('amazon_review_polarity_csv/train.csv', header=None, names=['class', 'title', 'text'])
test_data = pd.read_csv('amazon_review_polarity_csv/test.csv', header=None, names=['class', 'title', 'text'])

def clean_text(text):
    text = text.replace('\\"', '"').replace('\\n', ' ')
    return text.lower()

frac = 0.01
print(f"Вибір підмножини даних ({int(frac*100)}%)...")
train_data = train_data.sample(frac=frac, random_state=42)
test_data = test_data.sample(frac=frac, random_state=42)

print(f"Train data size after sampling: {len(train_data)}")
print(f"Test data size after sampling: {len(test_data)}")

print("Заповнення пропущених значень...")
train_data['title'] = train_data['title'].fillna("")
train_data['text'] = train_data['text'].fillna("")
test_data['title'] = test_data['title'].fillna("")
test_data['text'] = test_data['text'].fillna("")

print("Об'єднання та очищення тексту...")
train_data['text'] = (train_data['title'] + " " + train_data['text']).apply(clean_text)
test_data['text'] = (test_data['title'] + " " + test_data['text']).apply(clean_text)

X_train = train_data['text']
y_train = train_data['class'] - 1
X_test = test_data['text']
y_test = test_data['class'] - 1

print("Створення PyTorch датасету...")
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("Завантаження DistilBERT токенайзера...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_texts, val_texts, train_labels, val_labels = train_test_split(
    X_train.tolist(), y_train.tolist(), test_size=0.1, random_state=42
)

train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

print("Створення DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("Ініціалізація моделі...")
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

model = SentimentClassifier(n_classes=2)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовується пристрій: {device}")
model = model.to(device)

# Збереження та завантаження моделі
model_path = "sentiment_model.pt"

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

choice = input("Enter 'train' to train the model or 'load' to load a pre-trained model: ").strip().lower()

if choice == 'train':
    print("Розпочато тренування моделі...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience, trials = 2, 0
    print(f"Number of samples in train_loader: {len(train_dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    for epoch in range(5):
        print(f"Епоха {epoch + 1}...")
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Середня втрата для епохи {epoch + 1}: {avg_loss:.4f}")

        print("Обчислення валідаційної втрати...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trials = 0
            save_model(model, model_path)
        else:
            trials += 1
            if trials >= patience:
                print("Early stopping triggered")
                break
elif choice == 'load':
    if os.path.exists(model_path):
        print("Завантаження навченої моделі...")
        load_model(model, model_path)
    else:
        print("Не знайдено навченої моделі. Спочатку виконайте тренування.")

print("Запуск інтерфейсу для класифікації...")
def classify_review(review):
    tokens = tokenizer(review, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1).tolist()[0]
    return {"Negative": probs[0], "Positive": probs[1]}

gr.Interface(fn=classify_review, inputs="text", outputs="label").launch()
