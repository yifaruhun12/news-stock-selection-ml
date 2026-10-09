# 数据说明

本目录保存两套最终回测真正读取的本地数据快照。

| 文件 | 用途 |
|---|---|
| `signals_event_surprise.csv` | EventSurprise 阈值信号和股票代码 |
| `market_prices.csv` | 候选股票的日开盘、收盘与成交量 |
| `intraday_extremes.csv` | 候选股票的日内最高价和最低价 |
| `benchmark_csi800.csv` | 中证800基准行情 |
| `logistic_industry_feature_dataset.csv` | Logistic 年度滚动训练所需的标签和56项特征 |
| `DATA_MANIFEST.csv` | 文件行数、大小、日期范围和 SHA256 校验值 |

数据已按最终策略实际需要裁剪，不包含无关股票或未使用字段。所有回测
均直接读取这些 CSV，不需要重新下载或连接数据库。
