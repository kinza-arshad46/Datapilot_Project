"""
Reproducible Preprocessing Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class DataPilotPreprocessor:
    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scaler: str = "standard",
        encoding: str = "onehot",
        drop_duplicates: bool = True,
        handle_outliers: bool = False,
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaler = scaler
        self.encoding = encoding
        self.drop_duplicates = drop_duplicates
        self.handle_outliers = handle_outliers
        self.pipeline = None
        self.feature_names_ = None
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self.target_col_ = None
        self.problem_type_ = None

    def _detect_columns(self, df: pd.DataFrame, target: Optional[str] = None):
        cols = [c for c in df.columns if c != target]
        self.numerical_cols_ = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = df[cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        # Remaining as categorical
        remaining = [c for c in cols if c not in self.numerical_cols_ and c not in self.categorical_cols_]
        self.categorical_cols_.extend(remaining)

    def _build_pipeline(self):
        num_steps = [("imputer", SimpleImputer(strategy=self.numerical_strategy))]
        if self.scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif self.scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif self.scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        cat_steps = [("imputer", SimpleImputer(strategy=self.categorical_strategy))]
        if self.encoding == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        else:
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        transformers = []
        if self.numerical_cols_:
            transformers.append(("num", Pipeline(num_steps), self.numerical_cols_))
        if self.categorical_cols_:
            transformers.append(("cat", Pipeline(cat_steps), self.categorical_cols_))

        self.pipeline = ColumnTransformer(transformers, remainder="drop")

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        self.target_col_ = target
        self.problem_type_ = problem_type

        data = df.copy()
        if self.drop_duplicates:
            data = data.drop_duplicates()

        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found")

        y = data[target]
        X = data.drop(columns=[target])

        # Encode target for classification
        if problem_type == "classification":
            if y.dtype == object or str(y.dtype) == "category":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
                self.target_encoder_ = le
            else:
                self.target_encoder_ = None
        else:
            self.target_encoder_ = None

        self._detect_columns(data, target)
        self._build_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if problem_type == "classification" and y.nunique() > 1 else None
        )

        X_train_t = self.pipeline.fit_transform(X_train)
        X_test_t = self.pipeline.transform(X_test)

        # Feature names
        try:
            self.feature_names_ = list(self.pipeline.get_feature_names_out())
        except Exception:
            self.feature_names_ = [f"f{i}" for i in range(X_train_t.shape[1])]

        return X_train_t, X_test_t, np.array(y_train), np.array(y_test), self.feature_names_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted")
        X = df.copy()
        if self.target_col_ and self.target_col_ in X.columns:
            X = X.drop(columns=[self.target_col_])
        # Ensure columns
        for c in self.numerical_cols_ + self.categorical_cols_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.numerical_cols_ + self.categorical_cols_]
        return self.pipeline.transform(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DataPilotPreprocessor":
        return joblib.load(path)
