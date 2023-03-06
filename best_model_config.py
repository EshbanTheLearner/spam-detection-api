import numpy as np
import pandas as pd

from preprocess import clean_text

import optuna

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def objective(trial):
    df = pd.read_csv("spam_ham_dataset.csv")
    
    X = df["text"]
    y = df["label_num"]

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(X)
    counts = count_vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(counts, y, test_size=0.20, random_state=42)

    classifier_name = trial.suggest_categorical(
        "classifier",
        [
            "LogisticRegression",
            "RandomForest",
            "SVC"
        ]
    )

    if classifier_name == "LogisticRegression":
        lr_penalty = trial.suggest_categorical(
            "penalty", 
            [
                "l1",
                "l2",
                "elasticnet"
            ],
            log=True
        )
        classifier_obj = LogisticRegression(penalty=lr_penalty, random_state=42)
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    elif classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")
    
    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    acc = score.mean()
    return acc
