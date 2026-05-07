
---

## 📘 README – Auto ARIMA Forex Signal App (`arima_auto.py`)

```markdown
# ⏱️ Auto ARIMA – Live Forex Signals (Auto)

## Overview

This app uses **Auto ARIMA (pmdarima)** to forecast the next price of 7 major forex pairs.  
Signals are generated based on the expected return and cycle alignment.  
The model automatically selects the best ARIMA order (p,d,q) and seasonality.

## Features

- ✅ **Auto‑retrain every 60 minutes** – ARIMA model refitted in the background.
- ✅ **Rolling validation** to measure out‑of‑sample MSE (overfitting monitor).
- ✅ **Sound notification** on new signals.
- ✅ **Only BUY/SELL** in confidence band 70–80%.
- ✅ **Dynamic TP/SL** based on ATR (3:1 risk‑reward).
- ✅ **DTW cycle detection** to augment the pure ARIMA forecast.

## Requirements

- Python 3.8+
- streamlit
- yfinance
- pmdarima
- dtaidistance
- scikit-learn
- plotly

## Installation

```bash
pip install streamlit yfinance pandas numpy pmdarima scikit-learn dtaidistance plotly