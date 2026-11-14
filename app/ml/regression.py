"""
Regression models
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor  # pyright: ignore[reportMissingImports]
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # pyright: ignore[reportMissingImports]
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_regressors(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbosity=0
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1
        )
    return models


def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "actual": y_true.tolist()[:200],  # limit for response size
        "predicted": y_pred.tolist()[:200],
    }
