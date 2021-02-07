import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    feats = data.columns.drop(parameters['feat_to_remove'])
    X = data.loc[:, feats]
    y = data.loc[:, 'diabetes_mellitus']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'],
            random_state=parameters['random_state'])

    return [X_train, X_test, y_train, y_test]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> XGBClassifier:
    model = XGBClassifier(n_jobs=-1, num_boost_round=100, seed=parameters['random_state'])
    model.fit(X_train, y_train)

    return model

def evaluate_model(classifier: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = classifier.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has AUC score: %.3f", score)

