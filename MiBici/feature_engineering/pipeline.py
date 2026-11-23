"""Encadena pasos de características dentro de un Pipeline de sklearn."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from MiBici.feature_engineering.base import FeatureStep

TARGET = "salidas"


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Aplica ``steps`` en orden; cada paso ve las columnas de los anteriores."""

    def __init__(self, steps: Sequence[FeatureStep] = ()):
        self.steps = steps

    def fit(self, X, y=None):
        self.categorical_columns()  # valida declaraciones antes de ajustar
        frame = X.copy()
        if y is not None:
            frame[TARGET] = np.asarray(y)
        for step in self.steps:
            step.fit(frame)
            frame = self._apply(step, frame)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        frame = X.drop(columns=[TARGET], errors="ignore")
        for step in self.steps:
            frame = self._apply(step, frame)
        return frame

    @staticmethod
    def _apply(step, frame):
        result = step.transform(frame)
        if len(result) != len(frame) or not result.index.equals(frame.index):
            raise ValueError(f"{step!r} cambió el número u orden de las filas")
        return result

    def _declared(self, kind):
        declared, dropped = [], set()
        for step in self.steps:
            for column in getattr(step, kind):
                if column in declared:
                    raise ValueError(f"La columna {column!r} se declara dos veces")
                declared.append(column)
            dropped.update(step.drops)
        return [column for column in declared if column not in dropped]

    def categorical_columns(self):
        categorical, numeric = self._declared("categorical"), self._declared("numeric")
        repeated = set(categorical) & set(numeric)
        if repeated:
            raise ValueError(f"Columnas categóricas y numéricas: {sorted(repeated)}")
        return categorical

    def numeric_columns(self):
        return self._declared("numeric")

    def build_column_transformer(self):
        """Codifica lo declarado con la misma configuración que build_preprocessor."""
        transformers = []
        if self.categorical_columns():
            transformers.append((
                "categoria",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                self.categorical_columns(),
            ))
        if self.numeric_columns():
            transformers.append(("numero", StandardScaler(), self.numeric_columns()))
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def run_steps(frame: pd.DataFrame, steps: Sequence[FeatureStep]) -> pd.DataFrame:
    """Ajusta y aplica los pasos sobre un mismo frame (útil fuera de un modelo)."""
    for step in steps:
        step.fit(frame)
        frame = FeaturePipeline._apply(step, frame)
    return frame
