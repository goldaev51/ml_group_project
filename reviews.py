import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textattack.augmentation import WordNetAugmenter
import gradio as gr

# Uploading and processing data
train_data = pd.read_csv('amazon_review_polarity_csv/train.csv', header=None, names=['class', 'title', 'text'])
test_data = pd.read_csv('amazon_review_polarity_csv/test.csv', header=None, names=['class', 'title', 'text'])

def clean_text(text):
    text = text.replace('\\"', '"').replace('\\n', ' ')
    return text.lower()

# Fill in missing values in the title and text columns
train_data['title'] = train_data['title'].fillna("")
train_data['text'] = train_data['text'].fillna("")
test_data['title'] = test_data['title'].fillna("")
test_data['text'] = test_data['text'].fillna("")

# Merge and clean up text
train_data['text'] = (train_data['title'] + " " + train_data['text']).apply(clean_text)
test_data['text'] = (test_data['title'] + " " + test_data['text']).apply(clean_text)

train_data = train_data.drop(columns=['title'])
test_data = test_data.drop(columns=['title'])

X_train = train_data['text']
y_train = train_data['class'] - 1
X_test = test_data['text']
y_test = test_data['class'] - 1

# Data analysis
train_lengths = X_train.apply(len)
plt.hist(train_lengths, bins=30, alpha=0.7)
plt.title("Distribution of Review Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

for label in [0, 1]:
    subset = train_data[train_data['class'] == label + 1]
    text = ' '.join(subset['text'].tolist())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"WordCloud for class {label}")
    plt.axis('off')
    plt.show()

# Text augmentation
augmenter = WordNetAugmenter()
augmented_texts = [augmenter.augment(text) for text in X_train[:10]]
print("Augmented Samples:", augmented_texts[:3])

# Text vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Balancing classes
ros = RandomOverSampler(random_state=42)
X_train_tfidf_balanced, y_train_balanced = ros.fit_resample(X_train_tfidf, y_train)


# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(X_train_tfidf_balanced, y_train_balanced)
# lr_predictions = lr_model.predict(X_test_tfidf)
# print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
#
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train_tfidf_balanced, y_train_balanced)
# svm_predictions = svm_model.predict(X_test_tfidf)
# print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
#
# rf_model = RandomForestClassifier(n_estimators=100)
# rf_model.fit(X_train_tfidf_balanced, y_train_balanced)
# rf_predictions = rf_model.predict(X_test_tfidf)
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))

# Optimization of hyperparameters
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=cv)
grid.fit(X_train_tfidf_balanced, y_train_balanced)
print("Best Parameters:", grid.best_params_)

# Creating a PyTorch model
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_texts, val_texts, train_labels, val_labels = train_test_split(
    X_train.tolist(), y_train.tolist(), test_size=0.1, random_state=42
)

train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

model = SentimentClassifier(n_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# BERT training
# optimizer = AdamW(model.parameters(), lr=2e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):  # Три епохи
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


# Interactive interface
def classify_review(review):
    tokens = tokenizer(review, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1).tolist()[0]
    return {"Negative": probs[0], "Positive": probs[1]}

gr.Interface(fn=classify_review, inputs="text", outputs="label").launch()
