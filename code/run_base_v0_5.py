from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
for path in (str(SRC_DIR), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from code.common import (
    OOS_START,
    OUTPUT_DIR,
    load_prepared_data,
    metric_validation,
    oos_backtest_summary,
    write_frame,
    write_json,
    yearly_metrics,
)
from news_strategy.threshold_strategy_report import run_threshold_stop_backtest


STRATEGY_OUTPUT = OUTPUT_DIR / "base_v0_5"
EXPECTED = {
    "strategy_annualized_return": 0.22272672308262043,
    "strategy_cumulative_return": 0.5521955529201259,
    "strategy_sharpe": 0.8856895433792447,
    "strategy_max_drawdown": -0.4298814822324283,
    "realized_trade_win_rate": 0.40331491712707185,
    "entry_trades": 370,
}


def main() -> dict:
    prepared, data_audit = load_prepared_data()
    same_window_signals = prepared.signals.loc[
        prepared.signals["SIGNAL_DATE"].ge(OOS_START)
    ].copy()
    result = run_threshold_stop_backtest(
        replace(prepared, signals=same_window_signals),
        score_threshold=2.0,
        take_profit=0.15,
        stop_loss=0.08,
        max_position_weight=0.10,
        one_way_cost_bps=2.0,
        stop_execution="next_open",
    )
    summary = oos_backtest_summary(result, include_signal_count=True)

    daily = result.daily_returns.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
    daily = daily.loc[daily["trade_date"].ge(OOS_START)].copy()
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
    daily["drawdown"] = (
        daily["strategy_nav"] / daily["strategy_nav"].cummax() - 1.0
    )

    trades = result.trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades = trades.loc[trades["entry_date"].ge(OOS_START)].copy()

    write_frame(daily, STRATEGY_OUTPUT / "daily_returns.csv")
    write_frame(trades, STRATEGY_OUTPUT / "trades.csv")
    write_frame(result.open_positions, STRATEGY_OUTPUT / "open_positions.csv")
    write_frame(same_window_signals, STRATEGY_OUTPUT / "signals.csv")

    expected_checks = metric_validation(summary, EXPECTED)
    realized = trades.loc[
        trades["exit_reason"].isin(["take_profit", "stop_loss"])
    ]
    recomputed_cumulative = float(
        (1.0 + daily["strategy_return"]).prod() - 1.0
    )
    recomputed_win_rate = float(
        realized["gross_position_return"].gt(0).mean()
    )
    validation = {
        "signal_key_duplicates": int(
            same_window_signals.duplicated(
                ["SIGNAL_DATE", "TICKER_SYMBOL"]
            ).sum()
        ),
        "recomputed_cumulative_return": recomputed_cumulative,
        "cumulative_return_matches_summary": bool(
            np.isclose(
                recomputed_cumulative,
                summary["strategy_cumulative_return"],
                atol=1e-12,
            )
        ),
        "recomputed_realized_trade_win_rate": recomputed_win_rate,
        "win_rate_matches_summary": bool(
            np.isclose(
                recomputed_win_rate,
                summary["realized_trade_win_rate"],
                atol=1e-12,
            )
        ),
        "expected_report_metrics": expected_checks,
    }
    validation["all_checks_pass"] = bool(
        validation["signal_key_duplicates"] == 0
        and validation["cumulative_return_matches_summary"]
        and validation["win_rate_matches_summary"]
        and expected_checks["all_expected_metrics_match"]
    )

    payload = {
        "strategy": "Base v0.5",
        "status": "completed",
        "rules": {
            "signal": "EventSurprise >= 2.0",
            "entry": "next trading-day open",
            "take_profit": 0.15,
            "stop_loss": 0.08,
            "exit_trigger": "trading-day close",
            "exit": "next tradable trading-day open",
            "max_position_weight": 0.10,
            "max_industry_weight": None,
        },
        "data_audit": data_audit,
        "summary": summary,
        "yearly": [
            row
            for row in yearly_metrics(daily)
            if row["year"] >= OOS_START.year
        ],
        "validation": validation,
        "artifacts": {
            "daily_returns": STRATEGY_OUTPUT / "daily_returns.csv",
            "trades": STRATEGY_OUTPUT / "trades.csv",
            "open_positions": STRATEGY_OUTPUT / "open_positions.csv",
            "signals": STRATEGY_OUTPUT / "signals.csv",
        },
    }
    write_json(payload, STRATEGY_OUTPUT / "summary.json")
    write_json(validation, STRATEGY_OUTPUT / "validation.json")
    print(json.dumps({"strategy": payload["strategy"], **summary}, default=str))
    if not validation["all_checks_pass"]:
        raise RuntimeError("Base v0.5 did not reproduce the report metrics")
    return payload


if __name__ == "__main__":
    main()
