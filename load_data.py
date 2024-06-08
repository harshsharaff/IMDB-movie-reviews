import os
import tarfile
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def download_and_extract(url, download_path, extract_path):
    if not os.path.exists(download_path):
        urllib.request.urlretrieve(url, download_path)

    
    if not os.path.exists(extract_path):
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)


url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_path = "aclImdb_v1.tar.gz"
extract_path = "aclImdb"

download_and_extract(url, download_path, extract_path)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_imdb_data(data_dir):
    texts, labels = [], []
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            dir_path = os.path.join(data_dir, "aclImdb", split, sentiment)
            if not os.path.exists(dir_path):
                continue
            for filename in os.listdir(dir_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(dir_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                    labels.append(1 if sentiment == 'pos' else 0)
    return texts, labels

texts, labels = load_imdb_data(extract_path)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 128
BATCH_SIZE = 32

train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LEN)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for batch in test_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    break  

def train_knn(train_texts, train_labels, test_texts, test_labels):
    knn_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipeline.fit(train_texts, train_labels)
    knn_predictions = knn_pipeline.predict(test_texts)
    knn_accuracy = accuracy_score(test_labels, knn_predictions)
    return knn_accuracy

# Logistic Regression Classifier
def train_logistic_regression(train_texts, train_labels, test_texts, test_labels):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    train_features = tfidf_vectorizer.fit_transform(train_texts)
    test_features = tfidf_vectorizer.transform(test_texts)
    
    logistic_classifier = LogisticRegression(max_iter=1000)
    logistic_classifier.fit(train_features, train_labels)

    logistic_predictions = logistic_classifier.predict(test_features)

    logistic_accuracy = accuracy_score(test_labels, logistic_predictions)

    return logistic_accuracy


knn_acc = train_knn(train_texts, train_labels, test_texts, test_labels)
logistic_acc = train_logistic_regression(train_texts, train_labels, test_texts, test_labels)

print("kNN Accuracy:", knn_acc)
print("Logistic Regression Accuracy:", logistic_acc)
