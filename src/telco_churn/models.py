"""Model training helpers: SMOTE resampling, classifier factories, evaluation."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from telco_churn.config import RANDOM_STATE, TARGET_PP


def resample(X_train, y_train, sampling_strategy=1.0, k_neighbors=5, seed=RANDOM_STATE):
    """Apply SMOTE on the training data only."""
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def make_rf(random_state=RANDOM_STATE, **kwargs) -> RandomForestClassifier:
    defaults = dict(n_jobs=-1, random_state=random_state)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def make_lr(random_state=RANDOM_STATE, **kwargs) -> LogisticRegression:
    defaults = dict(max_iter=2000, random_state=random_state)
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def make_xgb(random_state=RANDOM_STATE, **kwargs) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        random_state=random_state,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def evaluate_topk(y_true, y_proba, k=TARGET_PP) -> dict:
    """Evaluate predictions where exactly the top-k by probability are positive."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(-y_proba, kind="mergesort")[:k]
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[order] = 1
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {
        "pp": int(k),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "threshold": float(np.sort(y_proba)[::-1][k - 1]),
    }


def fit_smote_predict(clf, X_train, y_train, X_test, sampling_strategy, k_neighbors, seed):
    """Resample with SMOTE, fit the classifier, return (clf, test probabilities)."""
    X_res, y_res = resample(X_train, y_train, sampling_strategy, k_neighbors, seed)
    clf.fit(X_res, y_res)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_proba