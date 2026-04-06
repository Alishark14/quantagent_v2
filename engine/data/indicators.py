"""Technical indicators: RSI, MACD, ROC, Stoch, WillR, ATR, ADX, BB.

All functions are pure math operating on numpy arrays. No external data deps.
These values are the FACTUAL ground truth that grounds all LLM analysis.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Wilder's smoothing (used by RSI, ATR, ADX)
# ---------------------------------------------------------------------------

def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's exponential smoothing: EMA with alpha = 1/period."""
    result = np.empty_like(values)
    result[:period] = np.nan
    result[period - 1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, len(values)):
        result[i] = result[i - 1] * (1 - alpha) + values[i] * alpha
    return result


# ---------------------------------------------------------------------------
# EMA (used by MACD, Stochastic)
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with alpha = 2 / (period + 1)."""
    result = np.empty_like(values, dtype=float)
    result[:period] = np.nan
    result[period - 1] = np.mean(values[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, len(values)):
        result[i] = result[i - 1] * (1 - alpha) + values[i] * alpha
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """Standard RSI using Wilder's smoothing. Returns the latest value."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    latest_gain = avg_gain[-1]
    latest_loss = avg_loss[-1]

    if np.isnan(latest_gain) or np.isnan(latest_loss):
        return 50.0  # not enough data

    if latest_loss == 0:
        return 50.0 if latest_gain == 0 else 100.0
    rs = latest_gain / latest_loss
    return float(100.0 - 100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def compute_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """MACD with crossover detection."""
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema

    # Signal line: EMA of the MACD line (from where it becomes valid)
    valid_start = slow - 1
    macd_valid = macd_line[valid_start:]
    signal_line_partial = _ema(macd_valid, signal)

    # Reconstruct full-length signal array
    signal_line = np.full_like(macd_line, np.nan)
    signal_line[valid_start:] = signal_line_partial

    histogram = macd_line - signal_line

    macd_val = float(macd_line[-1])
    signal_val = float(signal_line[-1])
    hist_val = float(histogram[-1])

    # Histogram direction
    if len(histogram) >= 2 and not np.isnan(histogram[-2]):
        hist_dir = "rising" if hist_val > histogram[-2] else "falling"
    else:
        hist_dir = "rising" if hist_val >= 0 else "falling"

    # Crossover detection (current bar)
    cross = "none"
    if len(macd_line) >= 2 and not np.isnan(signal_line[-2]):
        prev_diff = macd_line[-2] - signal_line[-2]
        curr_diff = macd_val - signal_val
        if prev_diff <= 0 < curr_diff:
            cross = "bullish_cross"
        elif prev_diff >= 0 > curr_diff:
            cross = "bearish_cross"

    return {
        "macd": macd_val,
        "signal": signal_val,
        "histogram": hist_val,
        "histogram_direction": hist_dir,
        "cross": cross,
    }


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def compute_roc(close: np.ndarray, period: int = 10) -> float:
    """Rate of Change: percentage change over N periods."""
    if len(close) <= period:
        return 0.0
    prev = close[-period - 1]
    if prev == 0:
        return 0.0
    return float((close[-1] - prev) / prev * 100)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def compute_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> dict:
    """Slow Stochastic %K and %D with zone classification."""
    n = len(close)
    if n < k_period:
        return {"k": 50.0, "d": 50.0, "zone": "neutral"}

    # Raw %K
    raw_k = np.empty(n, dtype=float)
    raw_k[:] = np.nan
    for i in range(k_period - 1, n):
        hh = np.max(high[i - k_period + 1: i + 1])
        ll = np.min(low[i - k_period + 1: i + 1])
        if hh == ll:
            raw_k[i] = 50.0
        else:
            raw_k[i] = (close[i] - ll) / (hh - ll) * 100

    # Slow %K = SMA of raw %K
    valid = raw_k[~np.isnan(raw_k)]
    if len(valid) < smooth:
        return {"k": 50.0, "d": 50.0, "zone": "neutral"}

    slow_k = np.convolve(valid, np.ones(smooth) / smooth, mode="valid")

    # %D = SMA of slow %K
    if len(slow_k) < d_period:
        d_val = float(slow_k[-1])
    else:
        d_val = float(np.mean(slow_k[-d_period:]))

    k_val = float(slow_k[-1])

    if k_val > 80:
        zone = "overbought"
    elif k_val < 20:
        zone = "oversold"
    else:
        zone = "neutral"

    return {"k": k_val, "d": d_val, "zone": zone}


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------

def compute_williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> float:
    """Williams %R: oscillator from -100 to 0."""
    if len(close) < period:
        return -50.0
    hh = np.max(high[-period:])
    ll = np.min(low[-period:])
    if hh == ll:
        return -50.0
    return float((hh - close[-1]) / (hh - ll) * -100)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> float:
    """Average True Range using Wilder's smoothing. Returns latest value."""
    tr = np.empty(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_series = _wilder_smooth(tr, period)
    return float(atr_series[-1])


def compute_atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Full ATR series (needed for volatility percentile)."""
    tr = np.empty(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _wilder_smooth(tr, period)


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> dict:
    """ADX with +DI, -DI and trend classification."""
    n = len(high)
    if n < period + 1:
        return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0, "classification": "WEAK"}

    # Directional movement
    up_move = np.diff(high)
    down_move = -np.diff(low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True range
    tr = np.empty(n - 1)
    for i in range(n - 1):
        tr[i] = max(
            high[i + 1] - low[i + 1],
            abs(high[i + 1] - close[i]),
            abs(low[i + 1] - close[i]),
        )

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    # +DI / -DI
    plus_di = np.where(smoothed_tr > 0, smoothed_plus_dm / smoothed_tr * 100, 0.0)
    minus_di = np.where(smoothed_tr > 0, smoothed_minus_dm / smoothed_tr * 100, 0.0)

    # DX
    di_sum = plus_di + minus_di
    safe_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = np.where(di_sum > 0, np.abs(plus_di - minus_di) / safe_sum * 100, 0.0)

    adx_series = _wilder_smooth(dx, period)

    adx_val = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0.0
    plus_di_val = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0.0
    minus_di_val = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0.0

    if adx_val > 25:
        classification = "TRENDING"
    elif adx_val < 20:
        classification = "RANGING"
    else:
        classification = "WEAK"

    return {
        "adx": adx_val,
        "plus_di": plus_di_val,
        "minus_di": minus_di_val,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def compute_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict:
    """Bollinger Bands with width and width percentile."""
    if len(close) < period:
        mid = float(close[-1])
        return {
            "upper": mid,
            "middle": mid,
            "lower": mid,
            "width": 0.0,
            "width_percentile": 50.0,
        }

    sma = float(np.mean(close[-period:]))
    std = float(np.std(close[-period:], ddof=0))
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = upper - lower

    # Width percentile over last 100 periods (or available)
    lookback = min(len(close), 100)
    widths = []
    for i in range(lookback):
        idx = len(close) - lookback + i
        if idx >= period - 1:
            start = idx - period + 1
            s = float(np.std(close[start: idx + 1], ddof=0))
            widths.append(2 * std_dev * s)

    if len(widths) > 1:
        width_pct = float(np.searchsorted(np.sort(widths), width) / len(widths) * 100)
    else:
        width_pct = 50.0

    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "width": width,
        "width_percentile": width_pct,
    }


# ---------------------------------------------------------------------------
# Volume MA
# ---------------------------------------------------------------------------

def compute_volume_ma(volume: np.ndarray, period: int = 20) -> dict:
    """Volume moving average with spike detection."""
    current = float(volume[-1])
    if len(volume) < period:
        ma = float(np.mean(volume))
    else:
        ma = float(np.mean(volume[-period:]))

    ratio = current / ma if ma > 0 else 0.0
    spike = ratio > 3.0

    return {
        "ma": ma,
        "current": current,
        "ratio": ratio,
        "spike": spike,
    }


# ---------------------------------------------------------------------------
# Volatility percentile
# ---------------------------------------------------------------------------

def get_volatility_percentile(atr_series: np.ndarray) -> float:
    """Where does current ATR sit in the last 100 ATR values? Returns 0-100."""
    valid = atr_series[~np.isnan(atr_series)]
    if len(valid) < 2:
        return 50.0
    lookback = valid[-100:] if len(valid) > 100 else valid
    current = lookback[-1]
    return float(np.searchsorted(np.sort(lookback), current) / len(lookback) * 100)


# ---------------------------------------------------------------------------
# Unified computation
# ---------------------------------------------------------------------------

def compute_all_indicators(candles: list[dict]) -> dict:
    """Compute all indicators from OHLCV candle dicts. Injected into agent prompts."""
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)

    atr_s = compute_atr_series(high, low, close)

    return {
        "rsi": compute_rsi(close),
        "macd": compute_macd(close),
        "roc": compute_roc(close),
        "stochastic": compute_stochastic(high, low, close),
        "williams_r": compute_williams_r(high, low, close),
        "atr": compute_atr(high, low, close),
        "adx": compute_adx(high, low, close),
        "bollinger_bands": compute_bollinger_bands(close),
        "volume_ma": compute_volume_ma(volume),
        "volatility_percentile": get_volatility_percentile(atr_s),
    }
