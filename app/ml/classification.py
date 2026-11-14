"""
Classification models
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier  # type: ignore[import-not-found]
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=random_state, use_label_encoder=False,
            eval_metric="logloss", verbosity=0
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1
        )
    return models


def evaluate_classification(
    y_true, y_pred, y_proba: Optional[np.ndarray] = None, average: str = "weighted"
) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    try:
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["report"] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return metrics
