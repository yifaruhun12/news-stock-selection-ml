from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .threshold_strategy import ThresholdPreparedData


@dataclass
class RegularizedLogisticModel:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray


def _numeric_matrix(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            for name in feature_names
        ]
    )


def fit_regularized_logistic(
    frame: pd.DataFrame,
    labels: pd.Series,
    feature_names: tuple[str, ...],
    *,
    l2_penalty: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> RegularizedLogisticModel:
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")
    matrix = _numeric_matrix(frame, feature_names)
    target = pd.to_numeric(labels, errors="coerce").to_numpy(float)
    valid_target = np.isfinite(target)
    matrix = matrix[valid_target]
    target = target[valid_target]
    if len(target) < 20 or len(np.unique(target)) < 2:
        raise ValueError("logistic training requires both classes and 20 rows")

    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    lower_bounds = np.quantile(matrix, 0.01, axis=0)
    upper_bounds = np.quantile(matrix, 0.99, axis=0)
    matrix = np.clip(matrix, lower_bounds, upper_bounds)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0

    for _ in range(max_iterations):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = (
            design.T @ (probability - target) / len(target)
            + l2_penalty * penalty @ coefficients
        )
        hessian = (
            (design.T * weights) @ design / len(target)
            + l2_penalty * penalty
        )
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < tolerance:
            break

    return RegularizedLogisticModel(
        feature_names=feature_names,
        medians=medians,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        means=means,
        scales=scales,
        coefficients=coefficients,
    )


def predict_logistic_probability(
    model: RegularizedLogisticModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    matrix = _numeric_matrix(frame, model.feature_names)
    matrix = np.where(np.isfinite(matrix), matrix, model.medians)
    matrix = np.clip(matrix, model.lower_bounds, model.upper_bounds)
    standardized = (matrix - model.means) / model.scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    linear = np.clip(design @ model.coefficients, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def build_intraday_tp_sl_labels(
    prepared: ThresholdPreparedData,
    *,
    take_profit: float,
    stop_loss: float,
) -> pd.DataFrame:
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be positive")
    calendar = pd.DatetimeIndex(prepared.calendar)
    date_to_index = {
        pd.Timestamp(date): index for index, date in enumerate(calendar)
    }
    records: list[dict[str, object]] = []
    for row in prepared.signals.itertuples(index=False):
        ticker = str(row.TICKER_SYMBOL)
        entry_date = pd.Timestamp(row.entry_date)
        entry_price = prepared.open_prices.at[entry_date, ticker]
        record = {
            "SIGNAL_DATE": pd.Timestamp(row.SIGNAL_DATE),
            "TICKER_SYMBOL": ticker,
            "entry_date": entry_date,
            "entry_price": (
                float(entry_price) if np.isfinite(entry_price) else None
            ),
            "label": np.nan,
            "label_date": pd.NaT,
            "label_reason": "invalid_entry",
            "label_holding_trading_days": np.nan,
        }
        if not np.isfinite(entry_price) or entry_price <= 0:
            records.append(record)
            continue
        take_profit_price = float(entry_price) * (1.0 + take_profit)
        stop_loss_price = float(entry_price) * (1.0 - stop_loss)
        start_index = date_to_index[entry_date]
        record["label_reason"] = "censored_at_sample_end"
        for index in range(start_index, len(calendar)):
            date = pd.Timestamp(calendar[index])
            high = prepared.high_prices.at[date, ticker]
            low = prepared.low_prices.at[date, ticker]
            volume = prepared.volumes.at[date, ticker]
            valid = (
                np.isfinite(high)
                and high > 0
                and np.isfinite(low)
                and low > 0
                and np.isfinite(volume)
                and volume > 0
            )
            if not valid:
                continue
            hit_take_profit = high >= take_profit_price
            hit_stop_loss = low <= stop_loss_price
            if not (hit_take_profit or hit_stop_loss):
                continue
            # Match the production backtest's conservative OHLC convention.
            label = 0 if hit_stop_loss else 1
            record.update(
                {
                    "label": label,
                    "label_date": date,
                    "label_reason": (
                        "stop_loss"
                        if hit_stop_loss
                        else "take_profit"
                    ),
                    "label_holding_trading_days": index - start_index,
                }
            )
            break
        records.append(record)
    return pd.DataFrame(records)


def binary_auc(labels: pd.Series, probabilities: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "label": pd.to_numeric(labels, errors="coerce"),
            "probability": pd.to_numeric(probabilities, errors="coerce"),
        }
    ).dropna()
    positives = int(frame["label"].eq(1).sum())
    negatives = int(frame["label"].eq(0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = frame["probability"].rank(method="average")
    positive_rank_sum = float(ranks.loc[frame["label"].eq(1)].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
