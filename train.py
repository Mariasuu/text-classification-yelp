import re
import string
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



# NLTK setup

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")



# Предобработка текста

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(tokens)



# Загрузка данных

def load_data(sample_size_train=50000, sample_size_test=10000):
    ds = load_dataset("Yelp/yelp_review_full")

    train_df = pd.DataFrame(ds["train"]).sample(sample_size_train, random_state=42)
    test_df = pd.DataFrame(ds["test"]).sample(sample_size_test, random_state=42)

    train_df["clean_text"] = train_df["text"].apply(preprocess_text)
    test_df["clean_text"] = test_df["text"].apply(preprocess_text)

    return train_df, test_df


# TF-IDF модель

def train_tfidf(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "TF-IDF + LogisticRegression")


# BERT эмбеддинги
def get_bert_embeddings(texts, tokenizer, model, batch_size=64):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.numpy())

    return np.vstack(embeddings)


def train_bert(train_df, test_df):
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()

    train_sample = train_df.sample(5000, random_state=42)
    test_sample = test_df.sample(1000, random_state=42)

    X_train = get_bert_embeddings(list(train_sample["clean_text"]), tokenizer, bert_model)
    X_test = get_bert_embeddings(list(test_sample["clean_text"]), tokenizer, bert_model)

    y_train = train_sample["label"]
    y_test = test_sample["label"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return evaluate_model(y_test, y_pred, "BERT + LogisticRegression")


# Оценка модели

def evaluate_model(y_true, y_pred, name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# MAIN
def main():
    train_df, test_df = load_data()

    tfidf_results = train_tfidf(train_df, test_df)
    bert_results = train_bert(train_df, test_df)

    results = pd.DataFrame([tfidf_results, bert_results])
    print("\n=== Итог ===")
    print(results)


if __name__ == "__main__":
    main()