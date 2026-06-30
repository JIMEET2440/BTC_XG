# BTC-XGB-SignalEngine
 
A binary classification system that predicts short-term Bitcoin price direction using gradient-boosted trees (XGBoost), with an underlying multi-timeframe analysis design.
 
## Overview
 
BTC-XGB-SignalEngine treats Bitcoin trading signal generation as a supervised classification problem: given the current market state, predict whether price will move up or down on the next candle. Rather than relying on a single resolution of price data, the project is architected around a **multi-timeframe fusion** philosophy — the idea that short-term price action is best interpreted in the context of broader trends visible at coarser timeframes. A 5-minute candle pattern means something different inside a bullish 1-hour trend than inside a bearish one, and the model is designed to learn from that combined context rather than from any single timeframe in isolation.
 
## Core approach
 
**Multi-timeframe data fusion**
The pipeline is built to ingest and align BTC/USDT OHLCV data across five resolutions — 3m, 5m, 15m, 30m, and 1h — giving the model simultaneous access to short-term noise and longer-term trend structure.
 
**Feature engineering**
A broad technical indicator suite is computed directly from price and volume:
- **RSI** (Relative Strength Index) — momentum and overbought/oversold conditions
- **MACD** (Moving Average Convergence Divergence) — trend direction and momentum shifts, including the signal line crossover
- **CCI** (Commodity Channel Index) — deviation of price from its statistical mean
- **Bollinger Bands** — volatility bands around a rolling mean
- **VWAP** (Volume Weighted Average Price) — cumulative volume-weighted price benchmark
- **EMA-based lag sequences** — exponential moving averages of open/high/low/close, with up to 10 lagged historical values per feature, giving the tree-based model temporal memory it doesn't have natively
**Labeling**
The target is a binary classification: whether the next candle's close is higher than the current candle's close.
 
**Modeling**
An `XGBClassifier` is trained on the engineered, scaled feature set to learn patterns associated with upward price movement.
 
**Signal generation and backtesting**
Model predictions are converted into a simple long-position strategy — entering when the model predicts an upward move (filtered by an above-average volume condition) and exiting when it predicts a downward move. A backtesting routine then walks through the resulting trade signals to compute cumulative PnL, giving a rough sense of how the classifier's predictions would translate into trading outcomes.
 
## Tech stack
 
- `pandas`, `numpy` — data processing
- `scikit-learn` — preprocessing (`MinMaxScaler`, `train_test_split`)
- `xgboost` — core classification model
- `matplotlib` — visualization
## Results
 
From the original full-pipeline run (5-minute timeframe data, EMA lag features, 90/10 train-test split):
 
| Metric | Value |
|---|---|
| Train accuracy | 0.902 |
| Test accuracy | 0.844 |
| Backtest cumulative PnL (on $1000 notional per trade) | $51,210.17 |
 
The backtest result comes from simulating entries on model-predicted upward signals (filtered by an above-average 20-period volume condition) and exits on predicted downward signals, walking through the resulting trade pairs to compute realized PnL.
 
## Current scope
 
The full pipeline is designed for multi-timeframe fusion across 3m/5m/15m/30m/1h data. In the current state of this repository, only the **5-minute** timeframe dataset is used end-to-end — the multi-timeframe ingestion structure is in place, but the other resolutions are not yet wired into the final feature set. Extending the model to genuinely fuse all five timeframes is the natural next step for this project.
