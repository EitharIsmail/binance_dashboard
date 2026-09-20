# Binance Direction Classifier

An end-to-end MLOps system that trains, tracks, registers, and serves machine learning models predicting short-term Bitcoin price direction (Bull / Neutral / Bear) from Binance market data.

This repository has two independent stages, each with its own README:

| Folder | What it does |
|---|---|
| [`3_Pipeline/`](./3_Pipeline/README.md) | Offline: downloads market data, engineers features, trains and benchmarks 10 classifiers per horizon, tracks everything in MLflow, and promotes winners through a model registry. |
| [`4_Deploy_Online/`](./4_Deploy_Online/README.md) | Online: a Dockerized FastAPI service that loads whatever the registry currently marks "Production" and serves live predictions over HTTP. |

---

## Mental model

The project is built around one deliberate separation: **training is expensive and infrequent; serving is cheap and constant.** These never run in the same process, and they don't even need to run on the same machine. The only thing connecting them is the **MLflow Model Registry** — a shared, named catalog that the pipeline writes to and the API reads from. Neither side needs to know the other exists; they only need to agree on a naming convention (`{symbol}_{horizon}_classifier`) and a tracking URI.

```
┌─────────────────────────┐          ┌──────────────────────┐           ┌─────────────────────────┐
│   Pipeline (offline)    │  writes  │   MLflow Registry    │   reads   │    Deploy_Online (live) │
│  acquire → clean →      │ ───────► │  (versions, stages,  │  ───────► │  FastAPI + Docker,      │
│  feature-engineer →     │          │   metrics per model) │           │  loads "Production"     │
│  train → evaluate →     │          │                      │           │  model per horizon      │
│  register → promote     │          │                      │           │                         │
└─────────────────────────┘          └──────────────────────┘           └─────────────────────────┘
        run on demand                                                        runs continuously
     (python main.py)                                                    (docker run / uvicorn)
```

A second deliberate design choice runs through both halves: **every prediction target is a `(symbol, horizon)` pair, not just a symbol.** "Will BTC move in the next 30 minutes" and "will BTC move in the next 4 hours" are genuinely different learning problems — different label distributions, different feature relevance, different models can win. The system never tries to serve one model for every horizon; it trains and registers one independent model per horizon, and the API's only real user-facing choice is *which horizon* to ask about.

## Why this shape, specifically

- **Training writes to a registry instead of a file** so that "which model is live" is a queryable, versioned decision — not a hardcoded path. Promoting a better model doesn't require touching the serving code at all.
- **A promotion gate (`deploy_if_better`), not automatic promotion**, exists because a model that looks good on one run's test set isn't automatically better than what's currently live — every new candidate has to beat the incumbent by a real margin before it takes over.
- **The API never imports anything from the training pipeline except the label-mapping constants and the stateless `FeatureEngineer`** — everything else it needs (the fitted preprocessor, the classifier) travels through MLflow as one bundled artifact, so serving code never has to reconstruct training-time state by hand.

## Quick start

```bash
# 1. Train and deploy a model for every supported horizon (one data pull, shared cleaning/features)
cd 3_Pipeline
python main.py --all-horizons

# 2. Serve whatever got promoted to Production
cd ../4_Deploy_Online
docker build -t binance-classifier-api .
docker run -p 8000:8000 \
  -v $(pwd)/../3_Pipeline/mlflow_binance.db:/app/mlflow_binance.db \
  -v $(pwd)/../3_Pipeline/mlruns:/app/mlruns \
  binance-classifier-api

# 3. Ask it something
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"horizon": "30m"}'
```

## The problem:

## The suggested solution:

## System Components Explanation:

## Features Explanation:

| Feature            | What it measures                        | Simple meaning                                         |
| ------------------ | --------------------------------------- | ------------------------------------------------------ |
| `return_1`         | Price change over **1 candle**          | How much price changed over the last 15 min            |
| `return_3`         | Price change over **3 candles**         | How much price changed over the last 45 min            |
| `volatility_10`    | Standard deviation of recent returns    | How much price has been fluctuating recently           |
| `ema_20`           | 20-period exponential moving average    | Shorter-term trend                                     |
| `ema_50`           | 50-period exponential moving average    | Longer-term trend                                      |
| `dist_from_ema_20` | Current price relative to EMA-20        | Whether price is above/below its short-term trend      |
| `rsi_14`           | Relative Strength Index over 14 periods | Recent buying vs. selling momentum                     |
| `volume_sma_20`    | Average volume over 20 periods          | What "normal" recent volume looks like                 |
| `volume_ratio`     | Current volume ÷ average volume         | Whether current trading activity is unusually high/low |
