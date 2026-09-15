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
