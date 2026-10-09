# News-Based Stock Selection Backtest Package

## Download the Complete Runnable Package

[Download the full project ZIP](https://github.com/yifaruhun12/news-stock-selection-ml/releases/download/v1.0.0/News_Stock_Selection_English_Guide.zip) from [Releases](https://github.com/yifaruhun12/news-stock-selection-ml/releases/tag/v1.0.0). It contains the source code, historical data, and reference results. Extract the ZIP and follow the commands below from the extracted project folder.

This repository presents the source code, English documentation, and compact result tables for browsing. The large market-data CSV files are supplied in the release ZIP.

This package reproduces the two strategies in the final research report using the included local data snapshots. The machine learning strategy uses 56 features, L2-regularized logistic regression, annual expanding-window walk-forward validation, and a 30% industry weight cap on new entries.

## Reported Backtest Results

| Strategy | Backtest Period | Annualized Return | Cumulative Return | Sharpe Ratio | Maximum Drawdown | Win Rate |
|---|---|---:|---:|---:|---:|---:|
| Base v0.5 | 2023-01-03 to 2025-04-15 | 22.27% | 55.22% | 0.886 | -42.99% | 40.33% |
| Logistic L2 + 30% Industry Entry Cap | 2023-01-03 to 2025-04-15 | 15.02% | 35.79% | 0.714 | -21.48% | 32.79% |

These are historical backtest results, not live trading returns. Maximum drawdown is shown as a negative percentage; a value closer to zero indicates a smaller drawdown. The Sharpe ratio is reported as a ratio, not a percentage.

## Installation and Quick Start

1. Install Python 3.12.
2. Extract the ZIP archive and open a terminal in `final_backtest_package_v0_5_logistic/`, the folder containing `run_all.py` and `requirements.txt`.
3. Install the dependencies and run both backtests:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

4. Review the combined results and reproduction status:

```text
outputs/final_comparison.csv
outputs/reproduction_status.json
```

On Windows, the included batch files provide the same workflow: run `安装依赖.bat` to install dependencies, then run `运行全部回测.bat` to execute both backtests. The command-line instructions above do not require using these batch files.

The pinned dependencies are:

```text
numpy==2.5.1
pandas==3.0.3
```

The backtest programs read local files from `data/`. They do not connect to a database or require the original research directory. Each run regenerates the daily returns, trade records, signals or selected signals, and validation outputs. Rerunning the programs overwrites output files with the same names.

## Package Structure

```text
code/
  run_base_v0_5.py          Base v0.5 backtest entry point
  run_logistic_l2.py        Logistic L2 backtest entry point
  common.py                Shared data loading, metrics, and validation
src/news_strategy/         Frozen backtest and model implementations
data/                     Local data snapshots required for reproduction
outputs/                  Included reference results and regenerated outputs
requirements.txt          Pinned Python dependency versions
run_all.py                Data verification and execution of both backtests
```

## Data Files

The included CSV files contain the data used by the two final backtests. They are limited to the securities and fields required by the strategies; no additional downloads or database access are needed.

| File | Purpose |
|---|---|
| `signals_event_surprise.csv` | EventSurprise threshold signals and stock identifiers |
| `market_prices.csv` | Daily opening prices, closing prices, and trading volumes for candidate stocks |
| `intraday_extremes.csv` | Intraday high and low prices for candidate stocks |
| `benchmark_csi800.csv` | CSI 800 benchmark price data |
| `logistic_industry_feature_dataset.csv` | Labels and 56 features for annual walk-forward logistic regression |
| `DATA_MANIFEST.csv` | Data file row counts, sizes, date ranges, and SHA-256 checksums |

## Reproduction Checks

Both backtest entry points include checks against the final report's reference metrics. Before running the strategies, `run_all.py` also checks the data files against the manifest.

After a successful run:

- `outputs/base_v0_5/validation.json` should contain `"all_checks_pass": true`.
- `outputs/logistic_l2_industry_cap_30pct/validation.json` should contain `"all_checks_pass": true`.
- `outputs/reproduction_status.json` should contain `"status": "passed"`.

Changing dependency versions or data files may cause exact reproduction checks to fail. Use the included `requirements.txt` and `data/DATA_MANIFEST.csv` as the reference for the expected environment and data snapshot.

## How to Read the Results

- **Annualized return:** The annualized strategy return over the reported backtest period.
- **Cumulative return:** The compounded strategy return over the full reported period.
- **Maximum drawdown:** The largest decline in strategy net asset value from a previous peak, recorded as a negative percentage.
- **Win rate:** The reported realized-trade win rate. In the supplied implementation, this measure uses trades closed by take-profit or stop-loss exits.
- **30% industry entry cap:** A restriction on industry exposure when placing new entries. It should not be interpreted as a guarantee that marked-to-market industry weights can never exceed 30% between entry decisions.

The logistic strategy's smaller drawdown accompanies a lower annualized return than the base strategy. The comparison describes the complete strategy configurations; it does not isolate the contribution of the machine learning filter from the industry cap.
