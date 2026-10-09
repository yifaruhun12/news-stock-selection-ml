from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    signals_with_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    selected_positions: pd.DataFrame
    summary: dict[str, float | int | None]


def _trade_dates_from_prices(prices: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(prices["TRADE_DATE"]).dt.normalize().unique())


def _normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    working = prices.copy()
    working["TRADE_DATE"] = pd.to_datetime(working["TRADE_DATE"]).dt.normalize()
    working["TICKER_SYMBOL"] = working["TICKER_SYMBOL"].astype(str)
    for column in ("OPEN_PRICE", "CLOSE_PRICE", "TURNOVER_VOL"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.drop_duplicates(
        ["TICKER_SYMBOL", "TRADE_DATE"], keep="last"
    ).sort_values(["TICKER_SYMBOL", "TRADE_DATE"])


def add_forward_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    if signals.empty or prices.empty:
        return signals.copy()
    trade_dates = _trade_dates_from_prices(prices)
    trade_date_values = np.asarray(trade_dates, dtype="datetime64[ns]")
    price_frame = _normalized_prices(prices)
    output = signals.copy()
    output["SIGNAL_DATE"] = pd.to_datetime(output["SIGNAL_DATE"]).dt.normalize()
    output["TICKER_SYMBOL"] = output["TICKER_SYMBOL"].astype(str)

    signal_values = output["SIGNAL_DATE"].to_numpy(dtype="datetime64[ns]")
    entry_indices = np.searchsorted(trade_date_values, signal_values, side="right")
    exit_indices = entry_indices + holding_days - 1
    valid_date_indices = (
        (entry_indices < len(trade_date_values))
        & (exit_indices < len(trade_date_values))
    )
    entry_values = np.full(
        len(output), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    exit_values = np.full(
        len(output), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    entry_values[valid_date_indices] = trade_date_values[
        entry_indices[valid_date_indices]
    ]
    exit_values[valid_date_indices] = trade_date_values[
        exit_indices[valid_date_indices]
    ]
    output["entry_date"] = entry_values
    output["exit_date"] = exit_values

    entry_prices = price_frame[
        ["TICKER_SYMBOL", "TRADE_DATE", "OPEN_PRICE", "TURNOVER_VOL"]
    ].rename(
        columns={
            "TRADE_DATE": "entry_date",
            "OPEN_PRICE": "entry_open",
            "TURNOVER_VOL": "entry_volume",
        }
    )
    exit_prices = price_frame[
        ["TICKER_SYMBOL", "TRADE_DATE", "CLOSE_PRICE"]
    ].rename(
        columns={
            "TRADE_DATE": "exit_date",
            "CLOSE_PRICE": "exit_close",
        }
    )
    output = output.merge(
        entry_prices,
        on=["TICKER_SYMBOL", "entry_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        exit_prices,
        on=["TICKER_SYMBOL", "exit_date"],
        how="left",
        validate="many_to_one",
    )
    valid_prices = (
        output["entry_open"].gt(0)
        & output["entry_volume"].gt(0)
        & np.isfinite(output["exit_close"])
    )
    output["forward_return"] = np.where(
        valid_prices,
        output["exit_close"] / output["entry_open"] - 1.0,
        np.nan,
    )
    return output.drop(
        columns=["entry_open", "entry_volume", "exit_close"]
    )


def _factor_diagnostics(signals: pd.DataFrame) -> dict[str, float | int | None]:
    valid = signals.dropna(subset=["stock_score", "forward_return"]).copy()
    if valid.empty:
        return {
            "signal_rows_with_returns": 0,
            "rank_ic_mean": None,
            "rank_ic_ir": None,
            "top_decile_mean_return": None,
            "bottom_decile_mean_return": None,
            "top_minus_bottom": None,
        }

    daily_ic = (
        valid.groupby("SIGNAL_DATE")
        .apply(
            lambda frame: (
                frame["stock_score"].rank().corr(frame["forward_return"].rank())
                if len(frame) >= 2
                else np.nan
            ),
            include_groups=False,
        )
        .dropna()
    )
    daily_top_bottom: list[tuple[float, float]] = []
    for _, frame in valid.groupby("SIGNAL_DATE"):
        if len(frame) < 10:
            continue
        cut = max(len(frame) // 10, 1)
        ordered = frame.sort_values("stock_score")
        daily_top_bottom.append(
            (
                float(ordered.tail(cut)["forward_return"].mean()),
                float(ordered.head(cut)["forward_return"].mean()),
            )
        )
    top_mean = (
        float(np.mean([value[0] for value in daily_top_bottom]))
        if daily_top_bottom
        else None
    )
    bottom_mean = (
        float(np.mean([value[1] for value in daily_top_bottom]))
        if daily_top_bottom
        else None
    )
    ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else np.nan
    return {
        "signal_rows_with_returns": int(len(valid)),
        "rank_ic_mean": float(daily_ic.mean()) if not daily_ic.empty else None,
        "rank_ic_ir": (
            float(daily_ic.mean() / ic_std)
            if np.isfinite(ic_std) and ic_std > 0
            else None
        ),
        "top_decile_mean_return": top_mean,
        "bottom_decile_mean_return": bottom_mean,
        "top_minus_bottom": (
            float(top_mean - bottom_mean)
            if top_mean is not None and bottom_mean is not None
            else None
        ),
    }


def _performance_metrics(daily_returns: pd.Series) -> dict[str, float | None]:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }
    wealth = (1.0 + daily_returns).cumprod()
    cumulative = float(wealth.iloc[-1] - 1.0)
    periods = len(daily_returns)
    annualized = float((1.0 + cumulative) ** (252.0 / periods) - 1.0)
    volatility = float(daily_returns.std(ddof=1) * np.sqrt(252.0))
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252.0))
        if periods > 1 and daily_returns.std(ddof=1) > 0
        else None
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    holding_days: int,
    top_n: int,
    one_way_cost_bps: float,
) -> BacktestResult:
    signals_with_returns = add_forward_returns(signals, prices, holding_days)
    diagnostics = _factor_diagnostics(signals_with_returns)
    trade_dates = _trade_dates_from_prices(prices)
    cost_rate = one_way_cost_bps / 10_000.0

    selected_frames: list[pd.DataFrame] = []
    for signal_date, frame in signals_with_returns.groupby("SIGNAL_DATE"):
        candidates = (
            frame.dropna(subset=["forward_return"])
            .loc[lambda value: value["stock_score"] > 0]
            .nlargest(top_n, "stock_score")
            .copy()
        )
        if candidates.empty:
            continue
        candidates["cohort_weight"] = 1.0 / holding_days / len(candidates)
        selected_frames.append(candidates)

    selected_positions = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    daily_contributions = pd.Series(
        0.0,
        index=pd.DatetimeIndex(trade_dates),
        dtype=float,
    )
    if not selected_positions.empty:
        date_to_index = {
            pd.Timestamp(day): index for index, day in enumerate(trade_dates)
        }
        active = selected_positions[
            ["TICKER_SYMBOL", "entry_date", "cohort_weight"]
        ].copy()
        active["entry_index"] = active["entry_date"].map(date_to_index)
        active = active.loc[active["entry_index"].notna()].copy()
        active = active.loc[active.index.repeat(holding_days)].reset_index(drop=True)
        active["holding_offset"] = np.tile(
            np.arange(holding_days), len(active) // holding_days
        )
        active["trade_index"] = (
            active["entry_index"].astype(int) + active["holding_offset"]
        )
        active = active.loc[active["trade_index"] < len(trade_dates)].copy()
        active["trade_date"] = pd.to_datetime(
            np.asarray(trade_dates, dtype="datetime64[ns]")[
                active["trade_index"].to_numpy()
            ]
        )

        price_frame = _normalized_prices(prices)
        price_frame["previous_close"] = price_frame.groupby(
            "TICKER_SYMBOL"
        )["CLOSE_PRICE"].shift(1)
        active = active.merge(
            price_frame[
                [
                    "TICKER_SYMBOL",
                    "TRADE_DATE",
                    "OPEN_PRICE",
                    "CLOSE_PRICE",
                    "previous_close",
                ]
            ],
            left_on=["TICKER_SYMBOL", "trade_date"],
            right_on=["TICKER_SYMBOL", "TRADE_DATE"],
            how="left",
            validate="many_to_one",
        )
        active["period_return"] = np.where(
            active["holding_offset"].eq(0),
            active["CLOSE_PRICE"] / active["OPEN_PRICE"] - 1.0,
            active["CLOSE_PRICE"] / active["previous_close"] - 1.0,
        )
        active["period_return"] = active["period_return"].where(
            np.isfinite(active["period_return"])
        )
        active["period_return"] = (
            active["period_return"]
            - active["holding_offset"].eq(0).astype(float) * cost_rate
            - active["holding_offset"].eq(holding_days - 1).astype(float)
            * cost_rate
        )
        contributions = (
            active.dropna(subset=["period_return"])
            .assign(
                contribution=lambda frame: frame["cohort_weight"]
                * frame["period_return"]
            )
            .groupby("trade_date")["contribution"]
            .sum()
        )
        daily_contributions.loc[contributions.index] = contributions

    daily = pd.DataFrame(
        {
            "trade_date": daily_contributions.index,
            "strategy_return": daily_contributions.to_numpy(),
        }
    )
    benchmark_frame = benchmark.copy()
    benchmark_frame["TRADE_DATE"] = pd.to_datetime(
        benchmark_frame["TRADE_DATE"]
    ).dt.normalize()
    benchmark_frame = benchmark_frame.sort_values("TRADE_DATE")
    benchmark_frame["benchmark_return"] = pd.to_numeric(
        benchmark_frame["CLOSE_INDEX"], errors="coerce"
    ).pct_change()
    daily = daily.merge(
        benchmark_frame[["TRADE_DATE", "benchmark_return"]],
        left_on="trade_date",
        right_on="TRADE_DATE",
        how="left",
    ).drop(columns="TRADE_DATE")
    daily["excess_return"] = (
        daily["strategy_return"] - daily["benchmark_return"].fillna(0.0)
    )
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (
        1.0 + daily["benchmark_return"].fillna(0.0)
    ).cumprod()

    summary: dict[str, float | int | None] = {
        **diagnostics,
        "signal_dates": int(signals["SIGNAL_DATE"].nunique()) if not signals.empty else 0,
        "stocks_selected": int(len(selected_positions)),
        "daily_return_observations": int(len(daily)),
        **{
            f"strategy_{key}": value
            for key, value in _performance_metrics(daily["strategy_return"]).items()
        },
        **{
            f"benchmark_{key}": value
            for key, value in _performance_metrics(
                daily["benchmark_return"].fillna(0.0)
            ).items()
        },
    }
    return BacktestResult(
        signals_with_returns=signals_with_returns,
        daily_returns=daily,
        selected_positions=selected_positions,
        summary=summary,
    )


def run_netted_overlap_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    holding_days: int,
    top_n: int,
    one_way_cost_bps: float,
) -> BacktestResult:
    """Backtest a hold-through alternative with net portfolio turnover.

    Consecutive cohorts can hold the same stock. This alternative aggregates
    them into one daily target weight and carries the previous close's target
    through the next overnight period. It therefore changes the original
    close-exit/next-open-entry exposure and is a turnover diagnostic, not merely
    an accounting correction.
    """

    signals_with_returns = add_forward_returns(signals, prices, holding_days)
    diagnostics = _factor_diagnostics(signals_with_returns)
    trade_dates = _trade_dates_from_prices(prices)
    cost_rate = one_way_cost_bps / 10_000.0

    selected_frames: list[pd.DataFrame] = []
    for _, frame in signals_with_returns.groupby("SIGNAL_DATE"):
        candidates = (
            frame.dropna(subset=["forward_return"])
            .loc[lambda value: value["stock_score"] > 0]
            .nlargest(top_n, "stock_score")
            .copy()
        )
        if candidates.empty:
            continue
        candidates["cohort_weight"] = 1.0 / holding_days / len(candidates)
        selected_frames.append(candidates)
    selected_positions = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )

    trade_index = pd.DatetimeIndex(trade_dates)
    daily = pd.DataFrame({"trade_date": trade_index})
    daily["strategy_return"] = 0.0
    daily["turnover"] = 0.0
    daily["trading_cost"] = 0.0
    if not selected_positions.empty:
        date_to_index = {
            pd.Timestamp(day): index for index, day in enumerate(trade_dates)
        }
        active = selected_positions[
            ["TICKER_SYMBOL", "entry_date", "cohort_weight"]
        ].copy()
        active["entry_index"] = active["entry_date"].map(date_to_index)
        active = active.loc[active["entry_index"].notna()].copy()
        active = active.loc[active.index.repeat(holding_days)].reset_index(drop=True)
        active["holding_offset"] = np.tile(
            np.arange(holding_days), len(active) // holding_days
        )
        active["trade_index"] = (
            active["entry_index"].astype(int) + active["holding_offset"]
        )
        active = active.loc[active["trade_index"] < len(trade_dates)].copy()
        active["trade_date"] = pd.to_datetime(
            np.asarray(trade_dates, dtype="datetime64[ns]")[
                active["trade_index"].to_numpy()
            ]
        )
        target_long = (
            active.groupby(["trade_date", "TICKER_SYMBOL"], as_index=False)[
                "cohort_weight"
            ]
            .sum()
            .rename(columns={"cohort_weight": "target_weight"})
        )
        target = (
            target_long.pivot(
                index="trade_date",
                columns="TICKER_SYMBOL",
                values="target_weight",
            )
            .reindex(trade_index)
            .fillna(0.0)
        )
        old_weight = target.shift(1).fillna(0.0)

        price_frame = _normalized_prices(prices)
        price_frame = price_frame.loc[
            price_frame["TICKER_SYMBOL"].isin(target.columns)
        ].copy()
        price_frame["previous_close"] = price_frame.groupby(
            "TICKER_SYMBOL"
        )["CLOSE_PRICE"].shift(1)
        price_frame["overnight_return"] = (
            price_frame["OPEN_PRICE"] / price_frame["previous_close"] - 1.0
        )
        price_frame["intraday_return"] = (
            price_frame["CLOSE_PRICE"] / price_frame["OPEN_PRICE"] - 1.0
        )

        def _return_matrix(column: str) -> pd.DataFrame:
            return (
                price_frame.pivot(
                    index="TRADE_DATE",
                    columns="TICKER_SYMBOL",
                    values=column,
                )
                .reindex(index=trade_index, columns=target.columns)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

        overnight = _return_matrix("overnight_return")
        intraday = _return_matrix("intraday_return")
        gross_return = (
            (old_weight * overnight).sum(axis=1)
            + (target * intraday).sum(axis=1)
        )
        turnover = (target - old_weight).abs().sum(axis=1)
        trading_cost = turnover * cost_rate
        daily["strategy_return"] = (gross_return - trading_cost).to_numpy()
        daily["turnover"] = turnover.to_numpy()
        daily["trading_cost"] = trading_cost.to_numpy()

    benchmark_frame = benchmark.copy()
    benchmark_frame["TRADE_DATE"] = pd.to_datetime(
        benchmark_frame["TRADE_DATE"]
    ).dt.normalize()
    benchmark_frame = benchmark_frame.sort_values("TRADE_DATE")
    benchmark_frame["benchmark_return"] = pd.to_numeric(
        benchmark_frame["CLOSE_INDEX"], errors="coerce"
    ).pct_change()
    daily = daily.merge(
        benchmark_frame[["TRADE_DATE", "benchmark_return"]],
        left_on="trade_date",
        right_on="TRADE_DATE",
        how="left",
    ).drop(columns="TRADE_DATE")
    daily["excess_return"] = (
        daily["strategy_return"] - daily["benchmark_return"].fillna(0.0)
    )
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_nav"] = (
        1.0 + daily["benchmark_return"].fillna(0.0)
    ).cumprod()
    summary: dict[str, float | int | None] = {
        **diagnostics,
        "signal_dates": int(signals["SIGNAL_DATE"].nunique()) if not signals.empty else 0,
        "stocks_selected": int(len(selected_positions)),
        "daily_return_observations": int(len(daily)),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "cumulative_trading_cost": float(daily["trading_cost"].sum()),
        **{
            f"strategy_{key}": value
            for key, value in _performance_metrics(daily["strategy_return"]).items()
        },
        **{
            f"benchmark_{key}": value
            for key, value in _performance_metrics(
                daily["benchmark_return"].fillna(0.0)
            ).items()
        },
    }
    return BacktestResult(
        signals_with_returns=signals_with_returns,
        daily_returns=daily,
        selected_positions=selected_positions,
        summary=summary,
    )
