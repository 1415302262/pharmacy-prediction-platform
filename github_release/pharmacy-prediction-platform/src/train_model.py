from __future__ import annotations

import copy
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=8,
    )
    model.fit(X_train, y_train)
    return model


def train_svr(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    use_gpu: bool = False,
    **overrides,
) -> XGBRegressor:
    params = dict(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=SEED,
        n_jobs=8,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        objective="reg:squarederror",
    )
    params.update(overrides)
    model = XGBRegressor(**params)
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    use_gpu: bool = False,
    **overrides,
) -> CatBoostRegressor:
    params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        depth=8,
        learning_rate=0.03,
        iterations=2000,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        verbose=False,
        task_type="GPU" if use_gpu else "CPU",
    )
    params.update(overrides)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    return model


class FingerprintMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(1024, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@torch.no_grad()
def predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for (xb,) in loader:
        xb = xb.to(device)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    use_gpu: bool = False,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 12,
) -> Tuple[nn.Module, Dict[str, list], str]:
    set_global_seed(SEED)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = FingerprintMLP(input_dim=X_train.shape[1]).to(device)

    train_dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    history = {"train_rmse": [], "valid_rmse": []}
    best_state = copy.deepcopy(model.state_dict())
    best_valid_rmse = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        train_pred = predict_mlp(model, X_train, device)
        valid_pred = predict_mlp(model, X_valid, device)
        train_rmse = float(math.sqrt(mean_squared_error(y_train, train_pred)))
        valid_rmse = float(math.sqrt(mean_squared_error(y_valid, valid_pred)))
        history["train_rmse"].append(train_rmse)
        history["valid_rmse"].append(valid_rmse)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, str(device)


def optimize_ensemble_weights(y_true: np.ndarray, prediction_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    model_names = list(prediction_dict.keys())
    prediction_matrix = np.column_stack([prediction_dict[name] for name in model_names])

    def objective(weights: np.ndarray) -> float:
        blended = prediction_matrix @ weights
        return math.sqrt(mean_squared_error(y_true, blended))

    initial = np.full(len(model_names), 1.0 / len(model_names), dtype=np.float64)
    bounds = [(0.0, 1.0)] * len(model_names)
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        weights = initial
    else:
        weights = np.clip(result.x, 0.0, 1.0)
        weights = weights / weights.sum()

    return {name: float(weight) for name, weight in zip(model_names, weights)}


def blend_predictions(prediction_dict: Dict[str, np.ndarray], weight_dict: Dict[str, float]) -> np.ndarray:
    blended = np.zeros_like(next(iter(prediction_dict.values())), dtype=np.float64)
    for model_name, prediction in prediction_dict.items():
        blended += weight_dict[model_name] * prediction
    return blended.astype(np.float32)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_torch_model(model: nn.Module, path: str | Path, metadata: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def save_json(data: Dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
