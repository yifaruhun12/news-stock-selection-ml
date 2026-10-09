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
    DATA_DIR,
    OOS_START,
    OUTPUT_DIR,
    load_prepared_data,
    metric_validation,
    oos_backtest_summary,
    write_frame,
    write_json,
    yearly_metrics,
)
from news_strategy.ml_filter import (
    binary_auc,
    fit_regularized_logistic,
    predict_logistic_probability,
)
from news_strategy.threshold_strategy_report import run_threshold_stop_backtest


STRATEGY_OUTPUT = OUTPUT_DIR / "logistic_l2_industry_cap_30pct"
TEST_YEARS = (2023, 2024, 2025)
L2_PENALTY = 0.10
RETENTION_FRACTION = 0.50

BASE_FEATURES = (
    "stock_score",
    "log_news_count",
    "log_event_count",
    "strength_component",
    "event_component",
    "novelty_component",
    "credibility_component",
    "confidence_component",
    "mean_impact",
    "mean_novelty",
    "mean_credibility",
    "mean_confidence",
    "score_rank_pct",
    "pre_5d_excess_return",
    "pre_20d_excess_return",
    "short_long_excess_gap",
    "event_density",
    "event_z",
    "price_expectation_z",
    "event_expectation_interaction",
)
INDUSTRY_PRICE_FEATURES = (
    "industry_pre_5d_excess_return",
    "industry_pre_20d_excess_return",
    "stock_minus_industry_pre_5d_excess",
    "stock_minus_industry_pre_20d_excess",
    "industry_pre_5d_excess_z",
    "industry_pre_20d_excess_z",
    "industry_surprise_score",
    "industry_price_available",
)
INDUSTRY_CONTEXT_FEATURES = (
    "industry_mapping_proxy",
    "industry_known",
)
EXPECTED = {
    "strategy_annualized_return": 0.15019528579175723,
    "strategy_cumulative_return": 0.3579305192171971,
    "strategy_sharpe": 0.7135623886480632,
    "strategy_max_drawdown": -0.21477697945738672,
    "realized_trade_win_rate": 0.32786885245901637,
    "entry_trades": 254,
}


def walk_forward(
    dataset: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_frames = []
    coefficient_frames = []
    folds = []
    for year in TEST_YEARS:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        train = dataset.loc[
            dataset["label"].notna()
            & dataset["label_date"].lt(test_start)
            & dataset["SIGNAL_DATE"].lt(test_start)
        ]
        test = dataset.loc[
            dataset["SIGNAL_DATE"].ge(test_start)
            & dataset["SIGNAL_DATE"].lt(test_end)
        ].copy()
        model = fit_regularized_logistic(
            train,
            train["label"],
            features,
            l2_penalty=L2_PENALTY,
        )
        train_probability = predict_logistic_probability(model, train)
        threshold = float(
            np.quantile(train_probability, 1.0 - RETENTION_FRACTION)
        )
        test["ml_probability"] = predict_logistic_probability(model, test)
        test["selected_top50"] = test["ml_probability"].ge(threshold)
        test["test_year"] = year
        prediction_frames.append(test)
        coefficient_frames.append(
            pd.DataFrame(
                {
                    "test_year": year,
                    "feature": ("intercept", *features),
                    "coefficient": model.coefficients,
                }
            )
        )
        folds.append(
            {
                "test_year": year,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "selection_threshold": threshold,
                "selected_rows": int(test["selected_top50"].sum()),
                "train_signal_end": train["SIGNAL_DATE"].max(),
                "train_label_end": train["label_date"].max(),
                "no_signal_date_leakage": bool(
                    train["SIGNAL_DATE"].max() < test_start
                ),
                "no_label_availability_leakage": bool(
                    train["label_date"].max() < test_start
                ),
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    labeled = predictions.dropna(subset=["label"])
    validation = {
        "feature_count": len(features),
        "folds": folds,
        "oos_auc": binary_auc(
            labeled["label"], labeled["ml_probability"]
        ),
        "prediction_key_duplicates": int(
            predictions.duplicated(
                ["SIGNAL_DATE", "TICKER_SYMBOL"]
            ).sum()
        ),
        "all_folds_no_signal_date_leakage": all(
            fold["no_signal_date_leakage"] for fold in folds
        ),
        "all_folds_no_label_availability_leakage": all(
            fold["no_label_availability_leakage"] for fold in folds
        ),
    }
    return predictions, coefficients, validation


def main() -> dict:
    dataset = pd.read_csv(
        DATA_DIR / "logistic_industry_feature_dataset.csv",
        dtype={"TICKER_SYMBOL": "string"},
        parse_dates=["SIGNAL_DATE", "entry_date", "label_date"],
    )
    dataset["TICKER_SYMBOL"] = (
        dataset["TICKER_SYMBOL"].astype("string").str.zfill(6)
    )
    category_features = tuple(
        sorted(
            column
            for column in dataset.columns
            if column.startswith("industry_category_")
        )
    )
    features = (
        *BASE_FEATURES,
        *INDUSTRY_PRICE_FEATURES,
        *INDUSTRY_CONTEXT_FEATURES,
        *category_features,
    )

    prepared, data_audit = load_prepared_data()
    base_signals = prepared.signals.loc[
        prepared.signals["SIGNAL_DATE"].ge(OOS_START)
    ].copy()
    predictions, coefficients, model_validation = walk_forward(
        dataset, features
    )
    selected = predictions.loc[
        predictions["selected_top50"],
        [
            "SIGNAL_DATE",
            "TICKER_SYMBOL",
            "ml_probability",
            "industry",
        ],
    ]
    selected_signals = base_signals.merge(
        selected,
        on=["SIGNAL_DATE", "TICKER_SYMBOL"],
        how="inner",
        validate="one_to_one",
    )
    result = run_threshold_stop_backtest(
        replace(prepared, signals=selected_signals),
        score_threshold=2.0,
        take_profit=0.25,
        stop_loss=0.10,
        max_position_weight=0.10,
        one_way_cost_bps=2.0,
        stop_execution="intraday_ohlc",
        max_industry_weight=0.30,
    )
    summary = oos_backtest_summary(result)
    summary.update(
        {
            "maximum_industry_entry_weight": result.summary[
                "maximum_industry_entry_weight"
            ],
            "maximum_observed_industry_weight": result.summary[
                "maximum_observed_industry_weight"
            ],
            "maximum_industry_weight_after_new_order": result.summary[
                "maximum_industry_weight_after_new_order"
            ],
            "industry_cap_violation_count": result.summary[
                "industry_cap_violation_count"
            ],
        }
    )

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

    write_frame(predictions, STRATEGY_OUTPUT / "predictions.csv")
    write_frame(coefficients, STRATEGY_OUTPUT / "coefficients.csv")
    write_frame(selected_signals, STRATEGY_OUTPUT / "selected_signals.csv")
    write_frame(daily, STRATEGY_OUTPUT / "daily_returns.csv")
    write_frame(trades, STRATEGY_OUTPUT / "trades.csv")
    write_frame(result.open_positions, STRATEGY_OUTPUT / "open_positions.csv")

    realized = trades.loc[
        trades["exit_reason"].isin(["take_profit", "stop_loss"])
    ]
    recomputed_cumulative = float(
        (1.0 + daily["strategy_return"]).prod() - 1.0
    )
    recomputed_win_rate = float(
        realized["gross_position_return"].gt(0).mean()
    )
    expected_checks = metric_validation(summary, EXPECTED)
    validation = {
        **model_validation,
        "selected_signal_rows": int(len(selected_signals)),
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
        "industry_cap_respected": bool(
            summary["industry_cap_violation_count"] == 0
            and summary["maximum_industry_weight_after_new_order"]
            <= 0.30 + 1e-10
        ),
        "expected_report_metrics": expected_checks,
    }
    validation["all_checks_pass"] = bool(
        validation["prediction_key_duplicates"] == 0
        and validation["all_folds_no_signal_date_leakage"]
        and validation["all_folds_no_label_availability_leakage"]
        and validation["cumulative_return_matches_summary"]
        and validation["win_rate_matches_summary"]
        and validation["industry_cap_respected"]
        and expected_checks["all_expected_metrics_match"]
    )

    payload = {
        "strategy": "Logistic L2 + 30% industry entry cap",
        "status": "completed",
        "rules": {
            "signal": "EventSurprise >= 2.0",
            "model": "L2 Logistic Regression",
            "training": "annual expanding walk-forward",
            "test_years": TEST_YEARS,
            "retention_fraction": RETENTION_FRACTION,
            "entry": "next trading-day open",
            "take_profit": 0.25,
            "stop_loss": 0.10,
            "exit": "intraday OHLC trigger",
            "max_position_weight": 0.10,
            "max_industry_weight": 0.30,
        },
        "features": features,
        "data_audit": data_audit,
        "summary": summary,
        "yearly": [
            row
            for row in yearly_metrics(daily)
            if row["year"] >= OOS_START.year
        ],
        "validation": validation,
        "artifacts": {
            "predictions": STRATEGY_OUTPUT / "predictions.csv",
            "coefficients": STRATEGY_OUTPUT / "coefficients.csv",
            "selected_signals": STRATEGY_OUTPUT / "selected_signals.csv",
            "daily_returns": STRATEGY_OUTPUT / "daily_returns.csv",
            "trades": STRATEGY_OUTPUT / "trades.csv",
            "open_positions": STRATEGY_OUTPUT / "open_positions.csv",
        },
    }
    write_json(payload, STRATEGY_OUTPUT / "summary.json")
    write_json(validation, STRATEGY_OUTPUT / "validation.json")
    print(json.dumps({"strategy": payload["strategy"], **summary}, default=str))
    if not validation["all_checks_pass"]:
        raise RuntimeError("Logistic L2 did not reproduce the report metrics")
    return payload


if __name__ == "__main__":
    main()
