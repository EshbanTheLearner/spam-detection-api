import pandas as pd

from preprocess import clean_text

import optuna

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def objective(trial):
    df = pd.read_csv("spam_ham_dataset.csv")
    df["text"] = df["text"].apply(lambda x: clean_text(x))
    X = df["text"]
    y = df["label_num"]

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(X)
    counts = count_vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(counts, y, test_size=0.20, random_state=42)

    classifier_name = trial.suggest_categorical(
        "classifier",
        [
            "RandomForest",
            "SVC",
            "NaiveBayes"
        ]
    )

    if classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    elif classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")
    else:
        classifier_obj = MultinomialNB()
    
    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    acc = score.mean()
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\nBest Trial:\n{study.best_trial}")
print(f"\nBest Params:\n{study.best_params}")