"""
Explainable AI - Feature importance, Permutation, SHAP (when feasible)
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


def get_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Tree-based or linear feature importance"""
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            imp = np.abs(coef).mean(axis=0)
        else:
            imp = np.abs(coef)
    if imp is None:
        return []

    pairs = sorted(zip(feature_names, imp), key=lambda x: abs(x[1]), reverse=True)
    total = sum(abs(v) for _, v in pairs) or 1
    return [
        {"feature": str(f), "importance": round(float(v), 4), "pct": round(abs(v) / total * 100, 2)}
        for f, v in pairs[:20]
    ]


def get_permutation_importance(
    model, X, y, feature_names: List[str], n_repeats: int = 5, random_state: int = 42
) -> List[Dict[str, Any]]:
    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1
        )
        pairs = sorted(zip(feature_names, result.importances_mean), key=lambda x: abs(x[1]), reverse=True)
        return [
            {"feature": str(f), "importance": round(float(v), 4)}
            for f, v in pairs[:15]
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_shap_values(model, X, feature_names: List[str], max_samples: int = 100) -> Optional[Dict]:
    """SHAP explanations when possible and dataset is manageable"""
    try:
        import shap
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_s = X[idx]
        else:
            X_s = X

        # Prefer TreeExplainer for tree models
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_s)
        else:
            explainer = shap.Explainer(model, X_s)
            sv = explainer(X_s)
            if hasattr(sv, "values"):
                sv = sv.values

        if isinstance(sv, list):
            # multi-class: take mean abs
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            sv = np.abs(sv)

        mean_abs = np.mean(sv, axis=0)
        pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
        return {
            "method": "shap",
            "features": [
                {"feature": str(f), "mean_abs_shap": round(float(v), 4)}
                for f, v in pairs[:15]
            ]
        }
    except Exception as e:
        return {"error": f"SHAP unavailable: {str(e)}", "method": "none"}
