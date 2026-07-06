"""Machine-learning forecasters: XGBoost and an LSTM (PyTorch).

Both consume the 23-dim feature matrix from data_loader.make_features. Features are
all lagged, so predicting price[t] from row t leaks no future information. The fixed
and rolling protocols differ only in how much history the model is trained on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .data_loader import Dataset, make_features
from .metrics import evaluate

warnings.filterwarnings("ignore")
SEED = 42

XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                  objective="reg:squarederror")


def _xy(dataset: Dataset) -> tuple[pd.DataFrame, pd.Series]:
    feat = make_features(dataset.price, dataset.exog)
    return feat.drop(columns="price"), feat["price"]


def run_xgboost(dataset: Dataset, rolling: bool = False) -> dict:
    """Train XGBoost; predict the test window (fixed) or refit each step (rolling)."""
    X, y = _xy(dataset)
    ts = dataset.test_size
    if not rolling:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[:-ts], y.iloc[:-ts])
        preds = model.predict(X.iloc[-ts:])
        actuals = y.iloc[-ts:].to_numpy()
        resid_sd = np.std(y.iloc[:-ts] - model.predict(X.iloc[:-ts]))
    else:
        preds, actuals = [], []
        n = len(X)
        for i in range(n - ts, n):
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X.iloc[:i], y.iloc[:i])
            preds.append(float(model.predict(X.iloc[i:i + 1])[0]))
            actuals.append(float(y.iloc[i]))
        preds = np.array(preds); actuals = np.array(actuals)
        resid_sd = np.std(actuals - preds)
    lo, hi = preds - 1.96 * resid_sd, preds + 1.96 * resid_sd
    out = {"forecast": np.asarray(preds), "lower": lo, "upper": hi,
           **evaluate(actuals, preds, lo, hi)}
    return out


def xgb_feature_importance(dataset: Dataset) -> pd.Series:
    """Gain-based feature importance from a model fit on all-but-test data."""
    X, y = _xy(dataset)
    ts = dataset.test_size
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X.iloc[:-ts], y.iloc[:-ts])
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False)


def tune_xgboost(dataset: Dataset, n_splits: int = 3) -> dict:
    """Light TimeSeriesSplit CV over a small grid; returns best params (RMSE)."""
    X, y = _xy(dataset)
    ts = dataset.test_size
    Xtr, ytr = X.iloc[:-ts], y.iloc[:-ts]
    grid = [(d, lr) for d in (2, 3, 4) for lr in (0.05, 0.1)]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best, best_rmse = None, np.inf
    for depth, lr in grid:
        rmses = []
        for tr_idx, val_idx in tscv.split(Xtr):
            m = XGBRegressor(**{**XGB_PARAMS, "max_depth": depth, "learning_rate": lr})
            m.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
            pred = m.predict(Xtr.iloc[val_idx])
            rmses.append(np.sqrt(np.mean((pred - ytr.iloc[val_idx].to_numpy()) ** 2)))
        if np.mean(rmses) < best_rmse:
            best_rmse, best = np.mean(rmses), {"max_depth": depth, "learning_rate": lr}
    return {"best_params": best, "cv_rmse": float(best_rmse)}


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def run_lstm(dataset: Dataset, seq_len: int = 12, max_epochs: int = 100,
             patience: int = 10) -> dict:
    """Train a 2-layer LSTM on 12-step sequences; predict the test window.

    Per the paper, the LSTM is not retrained for the rolling protocol, so the
    rolling and fixed forecasts are identical (the same trained model is reused).
    """
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = _xy(dataset)
    ts = dataset.test_size
    feat_cols = X.columns

    # standardize on training portion only
    Xtr_raw = X.iloc[:-ts]
    x_mean, x_std = Xtr_raw.mean(), Xtr_raw.std().replace(0, 1)
    y_mean, y_std = y.iloc[:-ts].mean(), y.iloc[:-ts].std()
    Xs = ((X - x_mean) / x_std).to_numpy(dtype=np.float32)
    ys = ((y - y_mean) / y_std).to_numpy(dtype=np.float32)

    def build_sequences(end: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys_ = [], []
        for t in range(seq_len, end):
            xs.append(Xs[t - seq_len:t])
            ys_.append(ys[t])
        return np.array(xs), np.array(ys_)

    n = len(Xs)
    train_end = n - ts
    Xseq, yseq = build_sequences(train_end)
    # 10% validation tail
    n_val = max(1, int(0.1 * len(Xseq)))
    Xtr, ytr = Xseq[:-n_val], yseq[:-n_val]
    Xval, yval = Xseq[-n_val:], yseq[-n_val:]

    class LSTMForecaster(nn.Module):
        def __init__(self, n_feat: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True,
                                dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):  # noqa: ANN001
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster(len(feat_cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xval_t = torch.tensor(Xval); yval_t = torch.tensor(yval)

    best_val, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xval_t), yval_t))
        if vloss < best_val:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in
                                                 model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # predict test window
    model.eval()
    preds = []
    with torch.no_grad():
        for t in range(train_end, n):
            seq = torch.tensor(Xs[t - seq_len:t][None, :, :])
            preds.append(float(model(seq)[0]) * y_std + y_mean)
    preds = np.array(preds)
    actuals = y.iloc[-ts:].to_numpy()
    resid_sd = np.std(actuals - preds)
    lo, hi = preds - 1.96 * resid_sd, preds + 1.96 * resid_sd
    return {"forecast": preds, "lower": lo, "upper": hi,
            **evaluate(actuals, preds, lo, hi)}
