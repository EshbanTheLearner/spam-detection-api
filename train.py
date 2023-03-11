import pandas as pd
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import wandb
import joblib

wandb.init()

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

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)
y_probas = clf.predict_proba(X_test)

wandb.sklearn.plot_learning_curve(clf, X_test, y_test)
wandb.sklearn.plot_roc(y_test, y_probas)
wandb.sklearn.plot_confusion_matrix(y_test, y_preds, labels=clf.classes_)
wandb.sklearn.plot_precision_recall(y_test, y_probas)

joblib.dump(clf, "./models/count_vectorizer.joblib")
joblib.dump(clf, "./models/naivebayes_clf.joblib")