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