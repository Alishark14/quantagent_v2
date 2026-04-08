"""Candlestick and trendline chart generation with matplotlib.

Dark trading theme. Returns PNG bytes (no file I/O).
Vision agents (PatternAgent, TrendAgent) consume these images.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
UP_COLOR = "#00d4aa"
DOWN_COLOR = "#ff6b6b"
VOLUME_UP_ALPHA = 0.4
VOLUME_DOWN_ALPHA = 0.3
TRENDLINE_FULL_COLOR = "#ffd700"     # gold — full OLS
TRENDLINE_SHORT_COLOR = "#00bfff"    # deep sky blue — last 20
BB_FILL_COLOR = "#9b59b6"           # purple — Bollinger band fill
BB_FILL_ALPHA = 0.12
SWING_HIGH_COLOR = "#ff6b6b"
SWING_LOW_COLOR = "#00d4aa"
DPI = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_arrays(candles: list[dict]) -> tuple[np.ndarray, ...]:
    """Extract OHLCV numpy arrays from candle dicts."""
    opens = np.array([c["open"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)
    return opens, highs, lows, closes, volumes


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply dark trading theme to figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def _draw_candlesticks(
    ax: plt.Axes,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    """Draw candlestick bodies and wicks on axes."""
    n = len(opens)
    x = np.arange(n)
    width = 0.6

    for i in range(n):
        color = UP_COLOR if closes[i] >= opens[i] else DOWN_COLOR
        # Wick
        ax.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.7)
        # Body
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_height = max(body_high - body_low, (highs[i] - lows[i]) * 0.005)
        rect = FancyBboxPatch(
            (x[i] - width / 2, body_low),
            width,
            body_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(rect)

    ax.set_xlim(-1, n)


def _draw_volume_bars(
    ax: plt.Axes,
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> None:
    """Draw volume bars on a secondary axes (bottom)."""
    n = len(volumes)
    x = np.arange(n)
    colors = [
        UP_COLOR if closes[i] >= opens[i] else DOWN_COLOR
        for i in range(n)
    ]
    alphas = [
        VOLUME_UP_ALPHA if closes[i] >= opens[i] else VOLUME_DOWN_ALPHA
        for i in range(n)
    ]
    for i in range(n):
        ax.bar(x[i], volumes[i], width=0.6, color=colors[i], alpha=alphas[i])

    ax.set_xlim(-1, n)
    ax.tick_params(colors=TEXT_COLOR, labelsize=6)
    ax.set_ylabel("Vol", fontsize=7, color=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _set_x_labels(ax: plt.Axes, candles: list[dict]) -> None:
    """Set readable x-axis labels from candle timestamps."""
    n = len(candles)
    step = max(1, n // 8)
    tick_positions = list(range(0, n, step))

    labels = []
    for i in tick_positions:
        ts = candles[i].get("timestamp", 0)
        if ts > 1e12:
            ts = ts / 1000  # ms -> s
        try:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            labels.append(dt.strftime("%b %d, %H:%M"))
        except (ValueError, OSError):
            labels.append(str(i))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=30, fontsize=7, color=TEXT_COLOR)


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Public: Candlestick chart
# ---------------------------------------------------------------------------

def generate_candlestick_chart(
    candles: list[dict],
    symbol: str,
    timeframe: str,
    swing_highs: list[float] | None = None,
    swing_lows: list[float] | None = None,
    width: int = 1024,
    height: int = 768,
) -> bytes:
    """Generate a dark-themed candlestick chart with volume bars.

    Optionally overlays swing high/low horizontal lines.
    Returns PNG image as bytes.
    """
    if not candles:
        return b""

    opens, highs, lows, closes, volumes = _extract_arrays(candles)

    fig_w = width / DPI
    fig_h = height / DPI
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)
    _apply_dark_theme(fig, ax_price)
    _apply_dark_theme(fig, ax_vol)

    ax_price.set_title(
        f"{symbol}  {timeframe}  —  Pattern Analysis",
        fontsize=11, fontweight="bold", color=TEXT_COLOR,
    )

    _draw_candlesticks(ax_price, opens, highs, lows, closes)
    _draw_volume_bars(ax_vol, opens, closes, volumes)
    _set_x_labels(ax_vol, candles)

    # Swing level lines
    n = len(candles)
    if swing_highs:
        for level in swing_highs:
            ax_price.axhline(
                y=level, color=SWING_HIGH_COLOR, linewidth=0.8,
                linestyle="--", alpha=0.6,
            )
    if swing_lows:
        for level in swing_lows:
            ax_price.axhline(
                y=level, color=SWING_LOW_COLOR, linewidth=0.8,
                linestyle="--", alpha=0.6,
            )

    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# Public: Trendline chart
# ---------------------------------------------------------------------------

def generate_trendline_chart(
    candles: list[dict],
    symbol: str,
    timeframe: str,
    width: int = 1024,
    height: int = 768,
) -> bytes:
    """Generate a dark-themed candlestick chart with OLS trendlines + Bollinger Bands.

    Overlays:
    - Full OLS trendline (regression through all candle closes)
    - Short OLS trendline (last 20 candles)
    - Bollinger Bands shaded region (20-period, 2 std dev)

    Returns PNG image as bytes.
    """
    if not candles:
        return b""

    opens, highs, lows, closes, volumes = _extract_arrays(candles)
    n = len(candles)

    fig_w = width / DPI
    fig_h = height / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _apply_dark_theme(fig, ax)

    ax.set_title(
        f"{symbol}  {timeframe}  —  Trend Analysis",
        fontsize=11, fontweight="bold", color=TEXT_COLOR,
    )

    _draw_candlesticks(ax, opens, highs, lows, closes)
    _set_x_labels(ax, candles)

    x_all = np.arange(n)

    # Full OLS trendline
    if n >= 2:
        coeffs_full = np.polyfit(x_all, closes, 1)
        trend_full = np.polyval(coeffs_full, x_all)
        ax.plot(
            x_all, trend_full,
            color=TRENDLINE_FULL_COLOR, linewidth=1.5, linestyle="--",
            label=f"OLS full (slope {coeffs_full[0]:.2f})",
            alpha=0.9,
        )

    # Short OLS trendline (last 20 candles)
    short_window = min(20, n)
    if short_window >= 2:
        x_short = np.arange(n - short_window, n)
        closes_short = closes[-short_window:]
        coeffs_short = np.polyfit(np.arange(short_window), closes_short, 1)
        trend_short = np.polyval(coeffs_short, np.arange(short_window))
        ax.plot(
            x_short, trend_short,
            color=TRENDLINE_SHORT_COLOR, linewidth=1.5, linestyle="-",
            label=f"OLS last {short_window} (slope {coeffs_short[0]:.2f})",
            alpha=0.9,
        )

    # Bollinger Bands (20-period, 2 std dev)
    bb_period = 20
    if n >= bb_period:
        bb_mid = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        for i in range(bb_period - 1, n):
            window = closes[i - bb_period + 1: i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=0)
            bb_mid[i] = sma
            bb_upper[i] = sma + 2.0 * std
            bb_lower[i] = sma - 2.0 * std

        valid = ~np.isnan(bb_mid)
        ax.fill_between(
            x_all[valid], bb_upper[valid], bb_lower[valid],
            color=BB_FILL_COLOR, alpha=BB_FILL_ALPHA,
            label="BB (20, 2σ)",
        )
        ax.plot(x_all[valid], bb_mid[valid], color=BB_FILL_COLOR, linewidth=0.6, alpha=0.5)

    ax.legend(
        loc="upper left", fontsize=7,
        facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
    )

    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# Public: Grounding context header
# ---------------------------------------------------------------------------

def generate_grounding_header(
    symbol: str,
    timeframe: str,
    indicators: dict[str, Any],
    flow: Any | None,
    parent_tf: Any | None,
    swing_highs: list[float],
    swing_lows: list[float],
    forecast_candles: int,
    forecast_description: str,
    num_candles: int,
    lookback_description: str,
    cost_info: dict | None = None,
) -> str:
    """Build the grounding context header injected into every LLM signal agent.

    This header contains FACTUAL data that anchors LLM analysis in
    mathematical reality. Indicator values are computed, not interpreted.
    """
    lines: list[str] = []
    lines.append("CONTEXT (do not override with visual impression):")
    lines.append(f"Symbol: {symbol} | Timeframe: {timeframe} | Lookback: {num_candles} candles ({lookback_description})")
    lines.append(f"Forecast horizon: {forecast_candles} candles ({forecast_description})")

    # Indicators
    if indicators:
        parts: list[str] = []

        # RSI
        rsi = indicators.get("rsi")
        if rsi is not None:
            zone = ""
            if rsi > 70:
                zone = " (overbought)"
            elif rsi < 30:
                zone = " (oversold)"
            parts.append(f"RSI: {rsi:.1f}{zone}")

        # MACD
        macd = indicators.get("macd")
        if isinstance(macd, dict):
            hist_dir = macd.get("histogram_direction", "")
            cross = macd.get("cross", "none")
            desc = f"histogram {hist_dir}"
            if cross != "none":
                desc += f", {cross.replace('_', ' ')}"
            parts.append(f"MACD: {desc}")

        # ROC
        roc = indicators.get("roc")
        if roc is not None:
            parts.append(f"ROC: {roc:.2f}%")

        # Stochastic
        stoch = indicators.get("stochastic")
        if isinstance(stoch, dict):
            k_val = stoch.get("k", 0)
            zone = stoch.get("zone", "neutral")
            parts.append(f"Stochastic: {k_val:.0f} ({zone})")

        # Williams %R
        willr = indicators.get("williams_r")
        if willr is not None:
            zone = ""
            if willr > -20:
                zone = " (overbought)"
            elif willr < -80:
                zone = " (oversold)"
            parts.append(f"Williams %R: {willr:.1f}{zone}")

        # ATR
        atr = indicators.get("atr")
        if atr is not None:
            parts.append(f"ATR: {atr:.2f}")

        # ADX
        adx = indicators.get("adx")
        if isinstance(adx, dict):
            adx_val = adx.get("adx", 0)
            classification = adx.get("classification", "")
            parts.append(f"ADX: {adx_val:.1f} ({classification})")

        # Bollinger Bands
        bb = indicators.get("bollinger_bands")
        if isinstance(bb, dict):
            width_pct = bb.get("width_percentile", 50)
            parts.append(f"BB width percentile: {width_pct:.0f}")

        # Volume
        vol = indicators.get("volume_ma")
        if isinstance(vol, dict):
            ratio = vol.get("ratio", 0)
            spike = vol.get("spike", False)
            vol_desc = f"Volume: {ratio:.1f}x avg"
            if spike:
                vol_desc += " (SPIKE)"
            parts.append(vol_desc)

        # Volatility percentile
        vol_pct = indicators.get("volatility_percentile")
        if vol_pct is not None:
            parts.append(f"Volatility percentile: {vol_pct:.0f}")

        lines.append(" | ".join(parts))

    # Flow data — only the funding rate stays in the grounding header.
    # OI deltas, GEX regime, and liquidation clusters used to be dumped
    # here but are now FlowSignalAgent's job. The other LLM agents
    # (Indicator / Pattern / Trend) should focus on their specialties;
    # FlowSignalAgent renders flow into a directional signal that lands
    # in ConvictionAgent's signals_block alongside the LLM voices.
    if flow is not None:
        if hasattr(flow, "funding_rate") and flow.funding_rate is not None:
            lines.append(
                f"Flow: Funding rate: {flow.funding_rate:+.4f}% ({flow.funding_signal})"
            )

    # Parent TF
    if parent_tf is not None:
        ptf_parts: list[str] = []
        if hasattr(parent_tf, "timeframe"):
            ptf_parts.append(f"Parent TF ({parent_tf.timeframe}): {parent_tf.trend_direction}")
            ptf_parts.append(f"price {parent_tf.ma_position.lower().replace('_', ' ')}")
            ptf_parts.append(f"ADX {parent_tf.adx_value:.0f} ({parent_tf.adx_classification})")
            ptf_parts.append(f"BB width pct {parent_tf.bb_width_percentile:.0f}")
        if ptf_parts:
            lines.append(" | ".join(ptf_parts))

    # Swing levels
    if swing_highs:
        levels_str = ", ".join(f"${h:,.2f}" for h in swing_highs[:3])
        lines.append(f"Nearest resistance: {levels_str}")
    if swing_lows:
        levels_str = ", ".join(f"${l:,.2f}" for l in swing_lows[:3])
        lines.append(f"Nearest support: {levels_str}")

    # Cost-aware R:R (if provided by ExecutionCostModel)
    if cost_info:
        raw_rr = cost_info.get("raw_rr", 0)
        net_rr = cost_info.get("net_rr", 0)
        sl_pct = cost_info.get("sl_pct", 0)
        tp_pct = cost_info.get("tp_pct", 0)
        drag = cost_info.get("fee_drag_pct", 0)
        severity = "LOW" if drag < 5 else "MEDIUM" if drag < 15 else "HIGH"
        lines.append(
            f"COST_AWARE_RR: SL {sl_pct:.1f}% / TP {tp_pct:.1f}% "
            f"-> raw RR {raw_rr:.2f} -> net RR {net_rr:.2f}. "
            f"Fee drag: {severity} ({drag:.0f}%)"
        )

    return "\n".join(lines)
