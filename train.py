import pandas as pd
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import wandb
import pickle

def load_data(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].apply(lambda x: clean_text(x))
    X = df["text"]
    y = df["label_num"]
    return X, y

filepath = "spam_ham_dataset.csv"
X, y = load_data(filepath)
vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)