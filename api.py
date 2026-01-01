from typing import Any, Dict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CUSTOM LAYERS (DEFINED BEFORE LOADING MODELS)
# =========================================================
class AttentionLayer(layers.Layer):
    """Custom attention mechanism for LSTM model"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config

class PositionalEncoding(layers.Layer):
    """Positional encoding for Transformer model"""
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config

# =========================================================
# LOAD ALL 6 MODELS + METADATA
# =========================================================
print("🔄 Loading improved ensemble models...")

scaler = joblib.load("scaler.pkl")
scaler_lstm = joblib.load("scaler_lstm.pkl")
rf_model = joblib.load("rf_model.pkl")
logreg_model = joblib.load("logreg_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")

# ✅ FIXED: Load models with custom_objects (no decorator needed)
custom_objects = {
    'AttentionLayer': AttentionLayer,
    'PositionalEncoding': PositionalEncoding
}

lstm_model = tf.keras.models.load_model(
    "lstm_attention_model.keras",
    custom_objects=custom_objects
)
transformer_model = tf.keras.models.load_model(
    "transformer_model.keras",
    custom_objects=custom_objects
)

FEATURES = joblib.load("features.pkl")
ensemble_weights = joblib.load("ensemble_weights.pkl")
metadata = joblib.load("model_metadata.pkl")

print(f"✅ Loaded 6 models with {len(FEATURES)} features")
print(f"📊 Ensemble weights: {ensemble_weights}")

LOOKBACK_DAYS = metadata['lookback_days']
HORIZON = metadata['horizon']

# =========================================================
# FEATURE COMPUTATION FUNCTIONS
# =========================================================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger(series, window=20, num_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-9)
    return upper, mid, lower, width

def compute_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["price"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_advanced_features(df):
    """Compute all 47 features from improved model"""
    # Momentum
    df['momentum_3d'] = df['price'].pct_change(3)
    df['momentum_10d'] = df['price'].pct_change(10)
    df['momentum_30d'] = df['price'].pct_change(30)
    df['accel_5d'] = df['price'].pct_change(5) - df['price'].pct_change(5).shift(5)
    
    # Volatility
    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_ratio'] = df['vol_5d'] / (df['vol_20d'] + 1e-9)
    
    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['hl_ma'] = df['hl_spread'].rolling(10).mean()
    
    # Volume features
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1e-9)
    df['volume_price_trend'] = df['Volume'] * df['ret_1d']
    df['volume_surge'] = (df['Volume'] > df['volume_ma_20'] * 2).astype(int)
    
    # Gap analysis
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    df['gap_ma'] = df['gap'].rolling(5).mean()
    
    # Range
    df['true_range'] = df[['High', 'Low']].apply(lambda x: x['High'] - x['Low'], axis=1)
    df['range_ma'] = df['true_range'].rolling(10).mean()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-9)
    
    # CCI (Commodity Channel Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-9)
    
    # MFI (Money Flow Index)
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(14).sum()
    negative_mf = negative_flow.rolling(14).sum()
    mfi_ratio = positive_mf / (negative_mf + 1e-9)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    # Distance from MAs
    df['ma_50'] = df['price'].rolling(50).mean()
    df['price_distance_ma20'] = (df['price'] - df['ma_20']) / (df['ma_20'] + 1e-9)
    df['price_distance_ma50'] = (df['price'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    df['price_trend_strength'] = df['ret_1d'].rolling(10).mean() / (df['vol_5d'] + 1e-9)
    
    # Volatility rank
    df['volatility_rank'] = df['vol_20d'].rolling(252, min_periods=20).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )
    
    # Statistical moments
    df['returns_skew'] = df['ret_1d'].rolling(20).skew()
    df['returns_kurt'] = df['ret_1d'].rolling(20).kurt()
    
    return df

def download_spy_vix():
    """Download SPY and VIX for market indicators"""
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=365*3)
        
        spy_data = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        spy_ret = spy_data['Close'].pct_change()
        
        vix_data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        vix_close = vix_data['Close']
        vix_change = vix_close.pct_change()
        
        return spy_ret, vix_close, vix_change
    except:
        return None, None, None

# =========================================================
# DATA PIPELINE
# =========================================================
def last_lookback_window(ticker: str):
    """Fetch and compute all 47 features"""
    end = datetime.utcnow()
    start = end - timedelta(days=730)
    
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["price"] = df["Close"]
    
    # Basic features
    df["ret_1d"] = df["price"].pct_change()
    df["ret_5d"] = df["price"].pct_change(5)
    df["ret_20d"] = df["price"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["ma_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)
    df["rsi"] = compute_rsi(df["price"])
    
    macd, macd_sig, macd_hist = compute_macd(df["price"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    
    bb_up, bb_mid, bb_low, bb_width = compute_bollinger(df["price"])
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low
    df["bb_width"] = bb_width
    
    df["atr_14"] = compute_atr(df)
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
    
    # Advanced features
    df = compute_advanced_features(df)
    
    # Market indicators
    spy_ret, vix_close, vix_change = download_spy_vix()
    if spy_ret is not None:
        df["spy_ret"] = spy_ret.reindex(df.index, method='ffill')
        df["spy_corr"] = df["ret_1d"].rolling(30).corr(df["spy_ret"])
        df["vix"] = vix_close.reindex(df.index, method='ffill')
        df["vix_change"] = vix_change.reindex(df.index, method='ffill')
    else:
        df["spy_ret"] = 0
        df["spy_corr"] = 0
        df["vix"] = 15
        df["vix_change"] = 0
    
    df = df.dropna()
    
    if len(df) < LOOKBACK_DAYS:
        return None
    
    return df.iloc[-LOOKBACK_DAYS:]

# =========================================================
# PREDICTION ENGINE
# =========================================================
def predict_ensemble(window):
    """Run all 6 models and combine predictions"""
    # Tabular models - convert to float explicitly
    X_tab = scaler.transform(window[FEATURES])
    proba_rf = float(rf_model.predict_proba(X_tab[-1:])[:, 1][0])
    proba_logreg = float(logreg_model.predict_proba(X_tab[-1:])[:, 1][0])
    proba_xgb = float(xgb_model.predict_proba(X_tab[-1:])[:, 1][0])
    proba_lgb = float(lgb_model.predict_proba(X_tab[-1:])[:, 1][0])
    
    # Sequential models (LSTM + Transformer)
    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]
    n_samples, n_timesteps, n_features = X_seq.shape
    X_seq_scaled = scaler_lstm.transform(X_seq.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
    
    proba_lstm = float(lstm_model.predict(X_seq_scaled, verbose=0).ravel()[0])
    proba_transformer = float(transformer_model.predict(X_seq_scaled, verbose=0).ravel()[0])
    
    # Weighted ensemble - ensure float type
    w = ensemble_weights
    proba_ens = float(
        w['lstm_weight'] * proba_lstm +
        w['transformer_weight'] * proba_transformer +
        w['rf_weight'] * proba_rf +
        w['logreg_weight'] * proba_logreg +
        w['xgb_weight'] * proba_xgb +
        w['lgb_weight'] * proba_lgb
    )
    
    return {
        'ensemble': proba_ens,
        'lstm': proba_lstm,
        'transformer': proba_transformer,
        'rf': proba_rf,
        'logreg': proba_logreg,
        'xgb': proba_xgb,
        'lgb': proba_lgb
    }

def build_explanation(row: pd.Series, probas: dict) -> str:
    """Generate explanation with model consensus"""
    msgs = []
    
    # Model consensus
    ensemble_proba = probas['ensemble']
    if ensemble_proba > 0.65:
        msgs.append(f"🎯 Strong BUY signal ({ensemble_proba:.1%} confidence)")
    elif ensemble_proba > 0.5:
        msgs.append(f"⚠️ Moderate BUY signal ({ensemble_proba:.1%} confidence)")
    elif ensemble_proba > 0.35:
        msgs.append(f"📊 Neutral ({ensemble_proba:.1%}) - HOLD recommended")
    else:
        msgs.append(f"🛑 SKIP signal ({ensemble_proba:.1%} confidence)")
    
    # RSI
    rsi = float(row.get("rsi", np.nan))
    if not np.isnan(rsi):
        if rsi > 70:
            msgs.append(f"RSI: {rsi:.1f} (Overbought)")
        elif rsi < 30:
            msgs.append(f"RSI: {rsi:.1f} (Oversold)")
        else:
            msgs.append(f"RSI: {rsi:.1f} (Neutral)")
    
    # MA trend
    ma10 = float(row.get("ma_10", np.nan))
    ma20 = float(row.get("ma_20", np.nan))
    if ma10 > ma20:
        msgs.append("📈 Bullish trend")
    else:
        msgs.append("📉 Bearish trend")
    
    return " | ".join(msgs)

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="Stock Signal API - 6-Model Ensemble", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SignalRequest(BaseModel):
    ticker: str

class SignalResponse(BaseModel):
    ticker: str
    date: str
    price: float
    proba: float
    signal: int
    action: str
    explanation: str

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "3.0",
        "models": metadata['models'],
        "features": len(FEATURES),
        "val_auc": metadata['val_auc'],
        "test_auc": metadata['test_auc_ensemble'],
        "ensemble_weights": ensemble_weights
    }

@app.post("/signal", response_model=SignalResponse)
def get_signal(req: SignalRequest):
    window = last_lookback_window(req.ticker)
    if window is None:
        return SignalResponse(
            ticker=req.ticker.upper(),
            date="N/A",
            price=0.0,
            proba=0.0,
            signal=0,
            action="NO_DATA",
            explanation="Insufficient data"
        )
    
    probas = predict_ensemble(window)
    proba_ens = probas['ensemble']
    signal = int(proba_ens >= 0.5)
    action = "BUY" if signal == 1 else "SKIP"
    
    last_row = window.iloc[-1]
    explanation = build_explanation(last_row, probas)
    
    return SignalResponse(
        ticker=req.ticker.upper(),
        date=str(last_row.name.date()),
        price=float(last_row["price"]),
        proba=float(proba_ens),
        signal=signal,
        action=action,
        explanation=explanation
    )

@app.get("/predict")
def get_signal_get(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    window = last_lookback_window(ticker)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "last_date": "N/A",
            "last_price": 0.0,
            "probability": 0.0,
            "signal": 0,
            "action": "NO_DATA",
            "explanation": "Insufficient data"
        }
    
    probas = predict_ensemble(window)
    proba_ens = probas['ensemble']
    signal = int(proba_ens >= 0.5)
    action = "BUY" if signal == 1 else "SKIP"
    
    last_row = window.iloc[-1]
    explanation = build_explanation(last_row, probas)
    
    return {
        "ticker": ticker.upper(),
        "last_date": str(last_row.name.date()),
        "last_price": float(last_row["price"]),
        "probability": float(proba_ens),
        "signal": signal,
        "action": action,
        "explanation": explanation,
        "model_probabilities": probas
    }

@app.get("/history")
def get_history(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    window = last_lookback_window(ticker)
    if window is None:
        return {"ticker": ticker.upper(), "history": []}
    
    history_data = []
    for i in range(len(window)):
        row = window.iloc[i]
        probas = predict_ensemble(window.iloc[max(0, i-59):i+1])
        
        history_data.append({
            "date": str(window.index[i].date()),
            "price": float(row["price"]),
            "probability": float(probas['ensemble']),
            "action": "BUY" if probas['ensemble'] >= 0.5 else "SKIP"
        })
    
    return {
        "ticker": ticker.upper(),
        "history": history_data,
        "dates": [str(idx.date()) for idx in window.index],
        "close": window["price"].round(2).tolist(),
        "ma10": window["ma_10"].round(2).tolist(),
        "ma20": window["ma_20"].round(2).tolist()
    }

@app.get("/metrics")
def get_metrics(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "model_info": {
            "ensemble_size": 6,
            "features": len(FEATURES),
            "val_auc": metadata['val_auc'],
            "test_auc": metadata['test_auc_ensemble'],
            "cagr": metadata['cagr'],
            "sharpe": metadata['sharpe']
        },
        "weights": ensemble_weights
    }
