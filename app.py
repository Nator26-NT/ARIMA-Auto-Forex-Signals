import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dtaidistance import dtw
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ARIMA Auto Forex", layout="wide", page_icon="⏱️")
st.title("⏱️ ARIMA – Auto Forex Signals (Every 1 Hour)")
st.markdown("**Auto‑retrain every 1h** | **Sound notification** | **3:1 R:R** | **Only BUY/SELL 70‑80% conf**")

# Sound using st.audio (no JavaScript)
sound_path = "https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3"

with st.sidebar:
    st.header("⚙️ Settings")
    interval = st.selectbox("Data Interval", ["1h", "1d"], index=0)
    atr_period = st.slider("ATR Period", 7, 21, 14)
    risk_mult = st.slider("Risk in ATR units", 1.0, 3.0, 1.0, 0.5)
    reward_ratio = 3.0
    min_conf = st.slider("Min confidence (0.70‑0.80)", 0.70, 0.80, 0.70, 0.01)
    auto_update = st.checkbox("Auto‑update every hour", value=True)

FOREX_PAIRS = [
    ("EURUSD=X", "EUR/USD (Fiber)"),
    ("GBPUSD=X", "GBP/USD (Cable)"),
    ("USDJPY=X", "USD/JPY (Gopher)"),
    ("USDCHF=X", "USD/CHF (Swissie)"),
    ("AUDUSD=X", "AUD/USD (Aussie)"),
    ("USDCAD=X", "USD/CAD (Loonie)"),
    ("NZDUSD=X", "NZD/USD (Kiwi)")
]

if "prev_signals" not in st.session_state:
    st.session_state.prev_signals = {}
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now()
if "sound_played" not in st.session_state:
    st.session_state.sound_played = False

def get_date_ranges():
    now = datetime.now()
    cy = now.year
    return f"{cy-1}-01-01", f"{cy-1}-12-31", f"{cy}-01-01", now.strftime("%Y-%m-%d")

# Custom fetch with retry and delay to avoid rate limiting
@st.cache_data(ttl=600)
def fetch_data(pair, start, end, interval, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(pair, start=start, end=end, interval=interval, progress=False)
            if data.empty:
                data = yf.download(pair, start=start, end=end, interval='1d', progress=False)
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])
            data.columns = ['open','high','low','close','volume']
            return data
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait = (attempt + 1) * 2
                st.warning(f"Rate limit hit for {pair}. Waiting {wait} seconds...")
                time.sleep(wait)
            else:
                st.error(f"Error fetching {pair}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

def add_features(df, atr_period):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(20).std()
    df['tr'] = np.maximum(df['high'] - df['low'],
                np.maximum(abs(df['high'] - df['close'].shift(1)),
                           abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(atr_period).mean()
    return df.dropna()

def find_cycle_signal(recent_df, train_df, window=48):
    recent_prices = recent_df['close'].values[-window:]
    if len(recent_prices) < window:
        return 0.0, 0.0
    x = np.arange(window)
    slope, intercept = np.polyfit(x, recent_prices, 1)
    recent_detrend = recent_prices - (slope*x + intercept)
    scaler = StandardScaler()
    recent_norm = scaler.fit_transform(recent_detrend.reshape(-1,1)).flatten()
    best_dist = np.inf; best_idx = None
    for i in range(len(train_df)-window):
        hist_prices = train_df['close'].iloc[i:i+window].values
        hist_detrend = hist_prices - (np.polyfit(np.arange(window), hist_prices,1)[0]*np.arange(window) +
                                      np.polyfit(np.arange(window), hist_prices,1)[1])
        hist_norm = scaler.fit_transform(hist_detrend.reshape(-1,1)).flatten()
        dist = dtw.distance(recent_norm, hist_norm)
        if dist < best_dist:
            best_dist = dist; best_idx = i
    if best_idx is None:
        return 0.0, 0.0
    future = train_df['return'].iloc[best_idx+window:best_idx+window+5].mean()
    return future, 1.0/(1.0+best_dist)

@st.cache_resource(ttl=3600, show_spinner=False)
def get_arima_model(pair_ticker, interval):
    train_start, train_end, _, _ = get_date_ranges()
    train_df = fetch_data(pair_ticker, train_start, train_end, interval)
    if train_df.empty:
        return None, np.inf
    series = train_df['close']
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]
    model = auto_arima(train, start_p=1, max_p=3, start_q=1, max_q=3,
                       seasonal=True, m=24, stepwise=True, suppress_warnings=True, error_action='ignore')
    preds = []
    for i in range(len(test)):
        try:
            fc = model.predict(n_periods=1)
            preds.append(fc.iloc[0] if hasattr(fc, 'iloc') else fc[0])
            model.update(test.iloc[i:i+1])
        except:
            preds.append(np.nan)
    preds = np.array(preds)
    valid = ~np.isnan(preds) & ~np.isnan(test.values)
    mse = mean_squared_error(test.values[valid], preds[valid]) if valid.sum() > 2 else np.inf
    final_model = auto_arima(series, start_p=1, max_p=3, start_q=1, max_q=3,
                             seasonal=True, m=24, stepwise=True, suppress_warnings=True, error_action='ignore')
    return final_model, mse

def arima_signal(current_close, forecast, volatility, cycle_ret, cycle_conf):
    forecast_return = (forecast - current_close)/current_close
    model_sig = np.clip(forecast_return/(volatility+1e-6), -1, 1)
    cycle_sig = np.clip(cycle_ret/0.001, -1, 1)
    total = (0.6*model_sig + 0.4*cycle_conf*cycle_sig) / (0.6 + 0.4*cycle_conf + 1e-6)
    confidence = abs(total)
    signal = "BUY" if total > 0 else "SELL" if total < 0 else None
    return signal, confidence

def compute_tp_sl(price, atr, signal, risk_atr, reward_ratio):
    risk = atr * risk_atr
    reward = risk * reward_ratio
    if signal == "BUY":
        return price + reward, price - risk
    else:
        return price - reward, price + risk

def process_pair(pair_ticker, pair_name, interval, atr_period, risk_mult, reward_ratio, min_conf):
    model, val_mse = get_arima_model(pair_ticker, interval)
    if model is None:
        return None, f"{pair_name}: model failed"
    _, _, recent_start, recent_end = get_date_ranges()
    recent_df = fetch_data(pair_ticker, recent_start, recent_end, interval)
    if recent_df.empty:
        return None, f"{pair_name}: no recent data"
    recent_feat = add_features(recent_df, atr_period)
    if len(recent_feat) < 10:
        return None, f"{pair_name}: insufficient data"
    train_start, train_end, _, _ = get_date_ranges()
    train_df = fetch_data(pair_ticker, train_start, train_end, interval)
    train_feat = add_features(train_df, atr_period)
    latest_idx = len(recent_feat) - 1
    curr_point = recent_feat.iloc[:latest_idx]
    curr_row = recent_feat.iloc[latest_idx]
    cycle_ret, cycle_conf = find_cycle_signal(curr_point, train_feat)
    forecast_series = model.predict(n_periods=1)
    forecast = forecast_series.iloc[0] if hasattr(forecast_series, 'iloc') else forecast_series[0]
    signal, confidence = arima_signal(curr_row['close'], forecast, curr_row['volatility'], cycle_ret, cycle_conf)
    if signal is None or confidence < min_conf or confidence > 0.80:
        return None, f"{pair_name}: no valid signal"
    tp, sl = compute_tp_sl(curr_row['close'], curr_row['atr'], signal, risk_mult, reward_ratio)
    return {
        "pair": pair_name, "ticker": pair_ticker, "signal": signal,
        "confidence": confidence, "cycle_conf": cycle_conf, "price": curr_row['close'],
        "tp": tp, "sl": sl, "atr": curr_row['atr'], "forecast": forecast, "val_mse": val_mse
    }, None

def main():
    # Auto‑refresh every 1 hour
    if auto_update and (datetime.now() - st.session_state.last_update).total_seconds() > 3600:
        st.session_state.last_update = datetime.now()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.subheader("🔔 Live ARIMA Signals (Auto‑refresh every 1h)")
    results = []
    progress = st.progress(0)
    for i, (ticker, name) in enumerate(FOREX_PAIRS):
        with st.spinner(f"ARIMA on {name}..."):
            # Small delay between pairs to reduce rate limiting
            if i > 0:
                time.sleep(1)
            res, err = process_pair(ticker, name, interval, atr_period, risk_mult, reward_ratio, min_conf)
        if res:
            results.append(res)
        else:
            st.warning(err)
        progress.progress((i+1)/len(FOREX_PAIRS))
    progress.empty()
    
    # Play sound if new signals appear (using st.audio)
    current_keys = {f"{r['pair']}_{r['signal']}" for r in results}
    if current_keys and current_keys != st.session_state.prev_signals:
        st.audio(sound_path, format="audio/mpeg", autoplay=True)
        st.session_state.prev_signals = current_keys
        st.session_state.sound_played = True
    
    if not results:
        st.info("No BUY/SELL signals in 70-80% range. Waiting for next update.")
        return
    
    tabs = st.tabs([f"{r['pair']} – {r['signal']} (conf={r['confidence']:.2f})" for r in results])
    for tab, r in zip(tabs, results):
        with tab:
            col1, col2, col3 = st.columns(3)
            col1.metric("Direction", r['signal'], delta=f"{r['confidence']:.2f}")
            col2.metric("Cycle", f"{r['cycle_conf']:.2f}")
            col3.metric("Price", f"{r['price']:.5f}")
            st.code(f"{r['signal']} {r['ticker'].replace('=X','')} at {r['price']:.5f}\nTP: {r['tp']:.5f}\nSL: {r['sl']:.5f}\nForecast: {r['forecast']:.5f}\nVal MSE: {r['val_mse']:.6f}", language="text")
            st.success("✅ Signal ready")
    
    next_update = st.session_state.last_update + timedelta(hours=1)
    remaining = (next_update - datetime.now()).total_seconds() / 60
    st.caption(f"Next auto‑update in {remaining:.0f} minutes.")

if __name__ == "__main__":
    main()