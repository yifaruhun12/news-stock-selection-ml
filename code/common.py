from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_strategy.backtest import _performance_metrics
from news_strategy.threshold_strategy_report import (
    prepare_threshold_backtest_data,
)


OOS_START = pd.Timestamp("2023-01-01")
SCORE_THRESHOLD = 2.0


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value.relative_to(PACKAGE_ROOT))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    return value


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_prepared_data():
    signals = pd.read_csv(
        DATA_DIR / "signals_event_surprise.csv",
        dtype={"TICKER_SYMBOL": "string"},
        parse_dates=["SIGNAL_DATE"],
    )
    signals["TICKER_SYMBOL"] = (
        signals["TICKER_SYMBOL"].astype("string").str.zfill(6)
    )
    signals = signals.loc[
        pd.to_numeric(signals["stock_score"], errors="coerce").ge(
            SCORE_THRESHOLD
        )
    ].copy()
    tickers = set(signals["TICKER_SYMBOL"].dropna().unique())

    prices = pd.read_csv(
        DATA_DIR / "market_prices.csv",
        dtype={"TICKER_SYMBOL": "string"},
        parse_dates=["TRADE_DATE"],
    )
    prices["TICKER_SYMBOL"] = (
        prices["TICKER_SYMBOL"].astype("string").str.zfill(6)
    )
    prices = prices.loc[prices["TICKER_SYMBOL"].isin(tickers)].copy()

    extremes = pd.read_csv(
        DATA_DIR / "intraday_extremes.csv",
        dtype={"TICKER_SYMBOL": "string"},
        parse_dates=["TRADE_DATE"],
    )
    extremes["TICKER_SYMBOL"] = (
        extremes["TICKER_SYMBOL"].astype("string").str.zfill(6)
    )
    prices = prices.merge(
        extremes,
        on=["TICKER_SYMBOL", "TRADE_DATE"],
        how="left",
        validate="one_to_one",
    )
    required = ["HIGHEST_PRICE", "LOWEST_PRICE"]
    for column in required:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    tolerance = 1e-8
    invalid_mask = (
        prices["HIGHEST_PRICE"].lt(
            prices[["OPEN_PRICE", "CLOSE_PRICE"]].max(axis=1) - tolerance
        )
        | prices["LOWEST_PRICE"].gt(
            prices[["OPEN_PRICE", "CLOSE_PRICE"]].min(axis=1) + tolerance
        )
    )
    prices.loc[invalid_mask, required] = np.nan
    coverage = float(prices[required].notna().all(axis=1).mean())
    if coverage < 0.999:
        raise RuntimeError(f"usable intraday OHLC coverage is only {coverage:.2%}")

    benchmark = pd.read_csv(
        DATA_DIR / "benchmark_csi800.csv",
        parse_dates=["TRADE_DATE"],
    )
    prepared = prepare_threshold_backtest_data(
        signals,
        prices,
        benchmark,
        minimum_score_threshold=SCORE_THRESHOLD,
    )
    audit = {
        "signal_rows": int(len(signals)),
        "candidate_tickers": int(len(tickers)),
        "market_rows": int(len(prices)),
        "invalid_ohlc_rows": int(invalid_mask.sum()),
        "usable_ohlc_coverage": coverage,
    }
    return prepared, audit


def yearly_metrics(daily: pd.DataFrame) -> list[dict]:
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
    rows = []
    for year, group in frame.groupby("year"):
        wealth = (1.0 + group["strategy_return"]).cumprod()
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(wealth.iloc[-1] - 1.0),
                "benchmark_return": float(
                    (1.0 + group["benchmark_return"]).prod() - 1.0
                ),
                "maximum_drawdown": float(
                    (wealth / wealth.cummax() - 1.0).min()
                ),
            }
        )
    return rows


def oos_backtest_summary(result, *, include_signal_count: bool = False) -> dict:
    daily = result.daily_returns.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
    daily = daily.loc[daily["trade_date"].ge(OOS_START)].copy()

    trades = result.trades.copy()
    if not trades.empty:
        trades["entry_date"] = pd.to_datetime(
            trades["entry_date"]
        ).dt.normalize()
        trades = trades.loc[trades["entry_date"].ge(OOS_START)].copy()
    realized = (
        trades.loc[trades["exit_reason"].isin(["take_profit", "stop_loss"])]
        if not trades.empty
        else trades
    )
    completed = (
        trades.loc[trades["exit_reason"].ne("sample_end")]
        if not trades.empty
        else trades
    )
    excess = daily["strategy_return"] - daily["benchmark_return"]
    excess_std = excess.std(ddof=1)

    summary = {
        "oos_start": daily["trade_date"].min(),
        "oos_end": daily["trade_date"].max(),
        "observations": int(len(daily)),
        "entry_trades": int(len(trades)),
        "realized_trades": int(len(realized)),
        "take_profit_exits": (
            int(realized["exit_reason"].eq("take_profit").sum())
            if not realized.empty
            else 0
        ),
        "stop_loss_exits": (
            int(realized["exit_reason"].eq("stop_loss").sum())
            if not realized.empty
            else 0
        ),
        "realized_trade_win_rate": (
            float(realized["gross_position_return"].gt(0).mean())
            if not realized.empty
            else None
        ),
        "completed_trade_win_rate": (
            float(completed["gross_position_return"].gt(0).mean())
            if not completed.empty
            else None
        ),
        "average_completed_holding_trading_days": (
            float(completed["holding_trading_days"].mean())
            if not completed.empty
            else None
        ),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "annualized_arithmetic_trading_cost": float(
            daily["trading_cost"].mean() * 252.0
        ),
        "information_ratio": (
            float(excess.mean() / excess_std * np.sqrt(252.0))
            if np.isfinite(excess_std) and excess_std > 0
            else None
        ),
        **{
            f"strategy_{key}": value
            for key, value in _performance_metrics(
                daily["strategy_return"]
            ).items()
        },
        **{
            f"gross_{key}": value
            for key, value in _performance_metrics(
                daily["gross_return"]
            ).items()
        },
        **{
            f"benchmark_{key}": value
            for key, value in _performance_metrics(
                daily["benchmark_return"]
            ).items()
        },
    }
    if include_signal_count:
        summary["signal_rows_after_filter"] = int(
            result.summary["signal_rows_above_threshold"]
        )
    return summary


def metric_validation(
    summary: dict,
    expected: dict[str, float | int],
    *,
    tolerance: float = 1e-12,
) -> dict:
    checks = {}
    for key, expected_value in expected.items():
        actual_value = summary[key]
        if isinstance(expected_value, int):
            passed = int(actual_value) == expected_value
        else:
            passed = bool(
                np.isclose(
                    float(actual_value),
                    float(expected_value),
                    atol=tolerance,
                    rtol=0.0,
                )
            )
        checks[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "match": passed,
        }
    checks["all_expected_metrics_match"] = all(
        item["match"] for item in checks.values() if isinstance(item, dict)
    )
    return checks
