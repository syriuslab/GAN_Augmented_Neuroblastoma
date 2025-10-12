from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

def load_dataset(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def prepare_splits(df: pd.DataFrame, target_col: str, drop_cols: Optional[List[str]] = None,
                   test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42, scale: bool = True) -> DataSplit:
    drop_cols = drop_cols or []
    assert target_col in df.columns, f"target_col '{target_col}' not in columns: {df.columns.tolist()}"
    y = df[target_col].values
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    feature_names = X.columns.tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=random_state)

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values

    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
