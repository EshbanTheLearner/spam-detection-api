import numpy as np
import pandas as pd

from preprocess import clean_text

import optuna

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB