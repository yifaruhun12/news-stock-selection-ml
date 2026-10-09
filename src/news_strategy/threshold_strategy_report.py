from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import _normalized_prices, _performance_metrics


@dataclass
class ThresholdPreparedData:
    signals: pd.DataFrame
    calendar: pd.DatetimeIndex
    open_prices: pd.DataFrame
    close_prices: pd.DataFrame
    high_prices: pd.DataFrame
    low_prices: pd.DataFrame
    volumes: pd.DataFrame
    benchmark_returns: pd.Series


@dataclass
class ThresholdBacktestResult:
    daily_returns: pd.DataFrame
    trades: pd.DataFrame
    open_positions: pd.DataFrame
    summary: dict[str, float | int | None]


def prepare_threshold_backtest_data(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    minimum_score_threshold: float,
) -> ThresholdPreparedData:
    signal_frame = signals.copy()
    signal_frame["SIGNAL_DATE"] = pd.to_datetime(
        signal_frame["SIGNAL_DATE"]
    ).dt.normalize()
    signal_frame["TICKER_SYMBOL"] = (
        signal_frame["TICKER_SYMBOL"].astype(str).str.zfill(6)
    )
    signal_frame["stock_score"] = pd.to_numeric(
        signal_frame["stock_score"], errors="coerce"
    )
    signal_frame = signal_frame.dropna(
        subset=["SIGNAL_DATE", "TICKER_SYMBOL", "stock_score"]
    )

    benchmark_frame = benchmark.copy()
    benchmark_frame["TRADE_DATE"] = pd.to_datetime(
        benchmark_frame["TRADE_DATE"]
    ).dt.normalize()
    benchmark_frame["CLOSE_INDEX"] = pd.to_numeric(
        benchmark_frame["CLOSE_INDEX"], errors="coerce"
    )
    benchmark_frame = benchmark_frame.drop_duplicates(
        "TRADE_DATE", keep="last"
    ).sort_values("TRADE_DATE")
    benchmark_frame["benchmark_return"] = benchmark_frame[
        "CLOSE_INDEX"
    ].pct_change(fill_method=None)
    calendar = pd.DatetimeIndex(
        benchmark_frame["TRADE_DATE"].dropna().unique()
    ).sort_values()

    first_signal_date = signal_frame["SIGNAL_DATE"].min()
    first_entry_index = int(
        calendar.searchsorted(first_signal_date, side="right")
    )
    if first_entry_index >= len(calendar):
        raise ValueError("no trading day is available after the first signal")
    evaluation_dates = calendar[first_entry_index:]

    signal_values = signal_frame["SIGNAL_DATE"].to_numpy(
        dtype="datetime64[ns]"
    )
    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    entry_indices = np.searchsorted(
        calendar_values, signal_values, side="right"
    )
    valid_entries = entry_indices < len(calendar_values)
    entry_dates = np.full(
        len(signal_frame),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    entry_dates[valid_entries] = calendar_values[
        entry_indices[valid_entries]
    ]
    signal_frame["entry_date"] = entry_dates
    signal_frame = signal_frame.loc[
        signal_frame["entry_date"].notna()
        & signal_frame["stock_score"].ge(minimum_score_threshold)
    ].copy()
    signal_frame = (
        signal_frame.sort_values(
            ["SIGNAL_DATE", "TICKER_SYMBOL", "stock_score"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["SIGNAL_DATE", "TICKER_SYMBOL"], keep="first")
        .sort_values(["entry_date", "stock_score"], ascending=[True, False])
    )
    candidate_tickers = sorted(signal_frame["TICKER_SYMBOL"].unique())
    if not candidate_tickers:
        raise ValueError("no signal reaches the minimum score threshold")

    price_frame = _normalized_prices(prices)
    price_frame["TICKER_SYMBOL"] = (
        price_frame["TICKER_SYMBOL"].astype(str).str.zfill(6)
    )
    selected_prices = price_frame.loc[
        price_frame["TICKER_SYMBOL"].isin(candidate_tickers)
        & price_frame["TRADE_DATE"].isin(evaluation_dates)
    ]

    def _matrix(column: str) -> pd.DataFrame:
        if column not in selected_prices.columns:
            return pd.DataFrame(
                np.nan,
                index=evaluation_dates,
                columns=candidate_tickers,
                dtype=float,
            )
        return (
            selected_prices.pivot(
                index="TRADE_DATE",
                columns="TICKER_SYMBOL",
                values=column,
            )
            .reindex(index=evaluation_dates, columns=candidate_tickers)
            .astype(float)
        )

    benchmark_returns = (
        benchmark_frame.set_index("TRADE_DATE")["benchmark_return"]
        .reindex(evaluation_dates)
        .fillna(0.0)
    )
    return ThresholdPreparedData(
        signals=signal_frame,
        calendar=evaluation_dates,
        open_prices=_matrix("OPEN_PRICE"),
        close_prices=_matrix("CLOSE_PRICE"),
        high_prices=_matrix("HIGHEST_PRICE"),
        low_prices=_matrix("LOWEST_PRICE"),
        volumes=_matrix("TURNOVER_VOL"),
        benchmark_returns=benchmark_returns,
    )


def run_threshold_stop_backtest(
    prepared: ThresholdPreparedData,
    *,
    score_threshold: float,
    take_profit: float,
    stop_loss: float,
    max_position_weight: float,
    one_way_cost_bps: float,
    stop_execution: str = "next_open",
    max_holding_trading_days: int | None = None,
    max_industry_weight: float | None = None,
    industry_column: str = "industry",
) -> ThresholdBacktestResult:
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be positive")
    if not 0 < max_position_weight <= 1:
        raise ValueError("max_position_weight must be in (0, 1]")
    if one_way_cost_bps < 0:
        raise ValueError("one_way_cost_bps must be non-negative")
    if max_industry_weight is not None and not 0 < max_industry_weight <= 1:
        raise ValueError("max_industry_weight must be in (0, 1]")
    if (
        max_holding_trading_days is not None
        and max_holding_trading_days <= 0
    ):
        raise ValueError("max_holding_trading_days must be positive")
    if (
        max_holding_trading_days is not None
        and stop_execution != "intraday_ohlc"
    ):
        raise ValueError(
            "max_holding_trading_days currently requires intraday_ohlc "
            "execution"
        )
    if stop_execution not in {
        "next_open",
        "same_close",
        "intraday_ohlc",
    }:
        raise ValueError(
            "stop_execution must be 'next_open', 'same_close', or "
            "'intraday_ohlc'"
        )

    signals = prepared.signals.loc[
        prepared.signals["stock_score"].ge(score_threshold)
    ].copy()
    if (
        max_industry_weight is not None
        and industry_column not in signals.columns
    ):
        raise ValueError(
            f"industry cap requires signal column {industry_column!r}"
        )
    entry_candidates = {
        pd.Timestamp(date): frame.sort_values(
            "stock_score", ascending=False
        )
        for date, frame in signals.groupby("entry_date")
    }
    columns = prepared.open_prices.columns
    open_prices = prepared.open_prices
    close_prices = prepared.close_prices
    high_prices = prepared.high_prices
    low_prices = prepared.low_prices
    volumes = prepared.volumes
    if stop_execution == "intraday_ohlc" and (
        high_prices.isna().all().all() or low_prices.isna().all().all()
    ):
        raise ValueError(
            "intraday_ohlc execution requires HIGHEST_PRICE and LOWEST_PRICE"
        )
    overnight_returns = (
        open_prices / prepared.close_prices.shift(1) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intraday_returns = (
        close_prices / open_prices - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    close_weights = pd.Series(0.0, index=columns)
    cash_weight = 1.0
    active: dict[str, dict[str, object]] = {}
    pending_exits: dict[str, dict[str, object]] = {}
    trade_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    intraday_ambiguous_exits = 0
    industry_cap_violation_count = 0
    maximum_industry_weight_after_new_order = 0.0
    cost_rate = one_way_cost_bps / 10_000.0
    date_to_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(prepared.calendar)
    }
    terminal_date = pd.Timestamp(prepared.calendar[-1])

    for trade_date in prepared.calendar:
        trade_date = pd.Timestamp(trade_date)
        overnight = overnight_returns.loc[trade_date]
        open_factor = float(
            cash_weight + (close_weights * (1.0 + overnight)).sum()
        )
        if not np.isfinite(open_factor) or open_factor <= 0:
            raise RuntimeError(f"invalid open portfolio value on {trade_date}")
        pretrade_weights = (
            close_weights * (1.0 + overnight) / open_factor
        )
        pretrade_cash = cash_weight / open_factor
        target_weights = pretrade_weights.copy()
        target_cash = float(pretrade_cash)

        is_terminal = trade_date == terminal_date
        if is_terminal:
            requested_exits = {
                ticker: {
                    "reason": "sample_end",
                    "trigger_date": trade_date,
                }
                for ticker in active
            }
        elif stop_execution == "next_open":
            requested_exits = dict(pending_exits)
        else:
            requested_exits = {}

        executed_exit_tickers: set[str] = set()
        sold_weight = 0.0
        for ticker, exit_request in requested_exits.items():
            if ticker not in active:
                pending_exits.pop(ticker, None)
                continue
            exit_open = open_prices.at[trade_date, ticker]
            exit_volume = volumes.at[trade_date, ticker]
            if not (
                np.isfinite(exit_open)
                and exit_open > 0
                and np.isfinite(exit_volume)
                and exit_volume > 0
            ):
                if not is_terminal:
                    pending_exits[ticker] = exit_request
                continue
            weight = float(target_weights.at[ticker])
            sold_weight += weight
            target_cash += weight
            target_weights.at[ticker] = 0.0
            position = active.pop(ticker)
            pending_exits.pop(ticker, None)
            executed_exit_tickers.add(ticker)
            trade_records.append(
                {
                    "TICKER_SYMBOL": ticker,
                    "signal_date": position["signal_date"],
                    "signal_score": position["signal_score"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_weight": position["entry_weight"],
                    "exit_trigger_date": exit_request["trigger_date"],
                    "exit_date": trade_date,
                    "exit_price": float(exit_open),
                    "gross_position_return": (
                        float(exit_open) / float(position["entry_price"]) - 1.0
                    ),
                    "holding_trading_days": (
                        date_to_index[trade_date]
                        - date_to_index[pd.Timestamp(position["entry_date"])]
                    ),
                    "exit_reason": exit_request["reason"],
                    "status": "closed",
                }
            )

        bought_weight = 0.0
        entry_count = 0
        if not is_terminal and trade_date in entry_candidates:
            candidates = entry_candidates[trade_date]
            for row in candidates.itertuples(index=False):
                ticker = str(row.TICKER_SYMBOL)
                if ticker in active or ticker in executed_exit_tickers:
                    continue
                entry_open = open_prices.at[trade_date, ticker]
                entry_volume = volumes.at[trade_date, ticker]
                if not (
                    np.isfinite(entry_open)
                    and entry_open > 0
                    and np.isfinite(entry_volume)
                    and entry_volume > 0
                ):
                    continue
                current_turnover = sold_weight + bought_weight
                affordable = (
                    target_cash - current_turnover * cost_rate
                ) / (1.0 + cost_rate)
                raw_industry = (
                    getattr(row, industry_column)
                    if industry_column in candidates.columns
                    else None
                )
                industry = (
                    str(raw_industry)
                    if pd.notna(raw_industry) and str(raw_industry).strip()
                    else "__UNKNOWN__"
                )
                industry_room = 1.0
                if max_industry_weight is not None:
                    current_industry_weight = sum(
                        float(target_weights.at[active_ticker])
                        for active_ticker, position in active.items()
                        if position.get("industry") == industry
                    )
                    industry_room = max(
                        max_industry_weight - current_industry_weight,
                        0.0,
                    )
                allocation = float(
                    min(
                        max_position_weight,
                        industry_room,
                        max(affordable, 0.0),
                    )
                )
                if allocation <= 1e-10:
                    if affordable <= 1e-10:
                        break
                    continue
                if max_industry_weight is not None:
                    post_order_industry_weight = (
                        current_industry_weight + allocation
                    )
                    maximum_industry_weight_after_new_order = max(
                        maximum_industry_weight_after_new_order,
                        post_order_industry_weight,
                    )
                    if (
                        post_order_industry_weight
                        > max_industry_weight + 1e-10
                    ):
                        industry_cap_violation_count += 1
                target_weights.at[ticker] = allocation
                target_cash -= allocation
                bought_weight += allocation
                entry_count += 1
                active[ticker] = {
                    "signal_date": pd.Timestamp(row.SIGNAL_DATE),
                    "signal_score": float(row.stock_score),
                    "entry_date": trade_date,
                    "entry_price": float(entry_open),
                    "entry_weight": allocation,
                    "industry": industry,
                }

        maximum_industry_entry_weight = max(
            (
                sum(
                    float(target_weights.at[ticker])
                    for ticker, position in active.items()
                    if position.get("industry") == industry
                )
                for industry in {
                    position.get("industry")
                    for position in active.values()
                }
            ),
            default=0.0,
        )
        turnover = sold_weight + bought_weight
        cost_fraction = turnover * cost_rate
        net_cash = target_cash - cost_fraction
        if net_cash < -1e-10:
            raise RuntimeError(f"transaction cost exceeds cash on {trade_date}")

        intraday = intraday_returns.loc[trade_date]
        intraday_factors = 1.0 + intraday
        intraday_exit_values: dict[str, float] = {}
        intraday_exit_records: list[tuple[str, str, float]] = []
        if stop_execution == "intraday_ohlc" and not is_terminal:
            for ticker, position in list(active.items()):
                high_price = high_prices.at[trade_date, ticker]
                low_price = low_prices.at[trade_date, ticker]
                current_open = open_prices.at[trade_date, ticker]
                current_volume = volumes.at[trade_date, ticker]
                has_valid_extremes = (
                    np.isfinite(high_price)
                    and high_price > 0
                    and np.isfinite(low_price)
                    and low_price > 0
                    and np.isfinite(current_open)
                    and current_open > 0
                    and np.isfinite(current_volume)
                    and current_volume > 0
                )
                entry_price = float(position["entry_price"])
                take_profit_price = entry_price * (1.0 + take_profit)
                stop_loss_price = entry_price * (1.0 - stop_loss)
                hit_take_profit = (
                    has_valid_extremes and high_price >= take_profit_price
                )
                hit_stop_loss = (
                    has_valid_extremes and low_price <= stop_loss_price
                )
                if hit_take_profit or hit_stop_loss:
                    # OHLC data cannot determine the intraday order of the two
                    # hits. Use the conservative stop-loss-first convention.
                    if hit_take_profit and hit_stop_loss:
                        intraday_ambiguous_exits += 1
                    reason, exit_price = (
                        ("stop_loss", stop_loss_price)
                        if hit_stop_loss
                        else ("take_profit", take_profit_price)
                    )
                else:
                    holding_day_number = (
                        date_to_index[trade_date]
                        - date_to_index[
                            pd.Timestamp(position["entry_date"])
                        ]
                        + 1
                    )
                    close_price = close_prices.at[trade_date, ticker]
                    if not (
                        max_holding_trading_days is not None
                        and holding_day_number >= max_holding_trading_days
                        and np.isfinite(current_open)
                        and current_open > 0
                        and np.isfinite(close_price)
                        and close_price > 0
                        and np.isfinite(current_volume)
                        and current_volume > 0
                    ):
                        continue
                    reason = "max_holding_period"
                    exit_price = float(close_price)
                intraday_factors.at[ticker] = (
                    exit_price / float(current_open)
                )
                intraday_exit_records.append((ticker, reason, exit_price))

        gross_close_factor = float(
            target_cash + (target_weights * intraday_factors).sum()
        )
        net_close_factor = float(
            max(net_cash, 0.0)
            + (target_weights * intraday_factors).sum()
        )
        if not np.isfinite(net_close_factor) or net_close_factor <= 0:
            raise RuntimeError(f"invalid close portfolio value on {trade_date}")
        gross_factor = open_factor * gross_close_factor
        net_factor = open_factor * net_close_factor
        close_weights = (
            target_weights * intraday_factors / net_close_factor
        )
        cash_weight = max(net_cash, 0.0) / net_close_factor

        close_exit_notional = 0.0
        close_exit_count = 0
        if stop_execution == "intraday_ohlc":
            for ticker, reason, exit_price in intraday_exit_records:
                position = active.pop(ticker)
                weight = float(close_weights.at[ticker])
                close_exit_notional += weight
                close_exit_count += 1
                cash_weight += weight
                close_weights.at[ticker] = 0.0
                trade_records.append(
                    {
                        "TICKER_SYMBOL": ticker,
                        "signal_date": position["signal_date"],
                        "signal_score": position["signal_score"],
                        "entry_date": position["entry_date"],
                        "entry_price": position["entry_price"],
                        "entry_weight": position["entry_weight"],
                        "exit_trigger_date": trade_date,
                        "exit_date": trade_date,
                        "exit_price": exit_price,
                        "gross_position_return": (
                            exit_price / float(position["entry_price"]) - 1.0
                        ),
                        "holding_trading_days": (
                            date_to_index[trade_date]
                            - date_to_index[
                                pd.Timestamp(position["entry_date"])
                            ]
                        ),
                        "exit_reason": reason,
                        "status": "closed",
                    }
                )
        elif stop_execution == "same_close" and not is_terminal:
            close_exit_requests: list[
                tuple[str, str, float]
            ] = []
            for ticker, position in active.items():
                close_price = close_prices.at[trade_date, ticker]
                if not np.isfinite(close_price) or close_price <= 0:
                    continue
                position_return = (
                    float(close_price)
                    / float(position["entry_price"])
                    - 1.0
                )
                if position_return >= take_profit:
                    close_exit_requests.append(
                        (ticker, "take_profit", float(close_price))
                    )
                elif position_return <= -stop_loss:
                    close_exit_requests.append(
                        (ticker, "stop_loss", float(close_price))
                    )
            for ticker, reason, close_price in close_exit_requests:
                position = active.pop(ticker)
                weight = float(close_weights.at[ticker])
                close_exit_notional += weight
                close_exit_count += 1
                cash_weight += weight
                close_weights.at[ticker] = 0.0
                trade_records.append(
                    {
                        "TICKER_SYMBOL": ticker,
                        "signal_date": position["signal_date"],
                        "signal_score": position["signal_score"],
                        "entry_date": position["entry_date"],
                        "entry_price": position["entry_price"],
                        "entry_weight": position["entry_weight"],
                        "exit_trigger_date": trade_date,
                        "exit_date": trade_date,
                        "exit_price": close_price,
                        "gross_position_return": (
                            close_price
                            / float(position["entry_price"])
                            - 1.0
                        ),
                        "holding_trading_days": (
                            date_to_index[trade_date]
                            - date_to_index[
                                pd.Timestamp(position["entry_date"])
                            ]
                        ),
                        "exit_reason": reason,
                        "status": "closed",
                    }
                )
        daily_records.append(
            {
                "trade_date": trade_date,
                "gross_return": gross_factor - 1.0,
                "strategy_return": net_factor - 1.0,
                "turnover": turnover,
                "trading_cost": gross_factor - net_factor,
                "holdings_count": len(active),
                "cash_weight": cash_weight,
                "entries": entry_count,
                "entry_notional": bought_weight,
                "exits": (
                    len(executed_exit_tickers) + close_exit_count
                ),
                "exit_notional": sold_weight,
                "maximum_industry_entry_weight": (
                    maximum_industry_entry_weight
                ),
                "maximum_industry_weight": (
                    max(
                        (
                            sum(
                                float(close_weights.at[ticker])
                                for ticker, position in active.items()
                                if position.get("industry") == industry
                            )
                            for industry in {
                                position.get("industry")
                                for position in active.values()
                            }
                        ),
                        default=0.0,
                    )
                ),
            }
        )

        if not is_terminal and stop_execution == "next_open":
            for ticker, position in active.items():
                if ticker in pending_exits:
                    continue
                close_price = close_prices.at[trade_date, ticker]
                if not np.isfinite(close_price) or close_price <= 0:
                    continue
                position_return = (
                    float(close_price) / float(position["entry_price"]) - 1.0
                )
                if position_return >= take_profit:
                    pending_exits[ticker] = {
                        "reason": "take_profit",
                        "trigger_date": trade_date,
                    }
                elif position_return <= -stop_loss:
                    pending_exits[ticker] = {
                        "reason": "stop_loss",
                        "trigger_date": trade_date,
                    }

    open_position_records: list[dict[str, object]] = []
    for ticker, position in active.items():
        terminal_close = close_prices.at[terminal_date, ticker]
        open_position_records.append(
            {
                "TICKER_SYMBOL": ticker,
                **position,
                "mark_date": terminal_date,
                "mark_price": (
                    float(terminal_close)
                    if np.isfinite(terminal_close)
                    else None
                ),
                "status": "unliquidated_missing_terminal_open",
            }
        )
    open_positions = pd.DataFrame(open_position_records)
    trades = pd.DataFrame(trade_records)
    daily = pd.DataFrame(daily_records)
    daily["benchmark_return"] = prepared.benchmark_returns.reindex(
        daily["trade_date"]
    ).to_numpy()
    daily["excess_return"] = (
        daily["strategy_return"] - daily["benchmark_return"]
    )
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["gross_nav"] = (1.0 + daily["gross_return"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
    daily["drawdown"] = (
        daily["strategy_nav"] / daily["strategy_nav"].cummax() - 1.0
    )

    realized = trades.loc[
        trades["exit_reason"].isin(["take_profit", "stop_loss"])
    ] if not trades.empty else trades
    completed = trades.loc[
        trades["exit_reason"].ne("sample_end")
    ] if not trades.empty else trades
    time_exits = trades.loc[
        trades["exit_reason"].eq("max_holding_period")
    ] if not trades.empty else trades
    excess_std = daily["excess_return"].std(ddof=1)
    summary: dict[str, float | int | None] = {
        "score_threshold": score_threshold,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "max_position_weight": max_position_weight,
        "max_industry_weight": max_industry_weight,
        "industry_column": (
            industry_column if max_industry_weight is not None else None
        ),
        "one_way_cost_bps": one_way_cost_bps,
        "stop_execution": stop_execution,
        "max_holding_trading_days": max_holding_trading_days,
        "intraday_ambiguous_exits": intraday_ambiguous_exits,
        "signal_rows_above_threshold": int(len(signals)),
        "signal_dates_above_threshold": int(
            signals["SIGNAL_DATE"].nunique()
        ),
        "entry_trades": int(
            len(trades) + len(open_positions)
        ),
        "closed_trades": int(len(trades)),
        "take_profit_exits": int(
            (trades["exit_reason"] == "take_profit").sum()
        ) if not trades.empty else 0,
        "stop_loss_exits": int(
            (trades["exit_reason"] == "stop_loss").sum()
        ) if not trades.empty else 0,
        "max_holding_period_exits": int(
            (trades["exit_reason"] == "max_holding_period").sum()
        ) if not trades.empty else 0,
        "sample_end_exits": int(
            (trades["exit_reason"] == "sample_end").sum()
        ) if not trades.empty else 0,
        "unliquidated_positions": int(len(open_positions)),
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
        "max_holding_period_exit_win_rate": (
            float(time_exits["gross_position_return"].gt(0).mean())
            if not time_exits.empty
            else None
        ),
        "average_max_holding_period_exit_return": (
            float(time_exits["gross_position_return"].mean())
            if not time_exits.empty
            else None
        ),
        "average_holding_trading_days": (
            float(realized["holding_trading_days"].mean())
            if not realized.empty
            else None
        ),
        "median_holding_trading_days": (
            float(realized["holding_trading_days"].median())
            if not realized.empty
            else None
        ),
        "average_completed_holding_trading_days": (
            float(completed["holding_trading_days"].mean())
            if not completed.empty
            else None
        ),
        "median_completed_holding_trading_days": (
            float(completed["holding_trading_days"].median())
            if not completed.empty
            else None
        ),
        "average_holdings": float(daily["holdings_count"].mean()),
        "maximum_holdings": int(daily["holdings_count"].max()),
        "maximum_observed_industry_weight": float(
            daily["maximum_industry_weight"].max()
        ),
        "maximum_industry_entry_weight": float(
            daily["maximum_industry_entry_weight"].max()
        ),
        "maximum_industry_weight_after_new_order": (
            maximum_industry_weight_after_new_order
        ),
        "industry_cap_violation_count": industry_cap_violation_count,
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "total_turnover": float(daily["turnover"].sum()),
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "annualized_arithmetic_trading_cost": float(
            daily["trading_cost"].mean() * 252.0
        ),
        "information_ratio": (
            float(
                daily["excess_return"].mean()
                / excess_std
                * np.sqrt(252.0)
            )
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
    return ThresholdBacktestResult(
        daily_returns=daily,
        trades=trades,
        open_positions=open_positions,
        summary=summary,
    )
