"""Binance adapter — DATA DOWNLOAD ONLY (Phase 5 Week 10 trading TBD).

This adapter exists today for one reason: pull historical OHLCV from
Binance USDT-M perpetual futures into our Parquet store via
``scripts/download_history.py``. Trading is intentionally NOT implemented
yet — every order / position / balance method raises
``NotImplementedError`` so the live engine cannot accidentally route
orders to Binance before the trading surface has been audited.

Symbol convention:

  * Internal format (everywhere in the codebase): ``BASE-USDC``
    (e.g. ``BTC-USDC``, ``ETH-USDC``, ``SOL-USDC``).
  * CCXT format for Binance perp futures: ``BASE/USDT:USDT``
    (Binance perpetuals settle in USDT, NOT USDC — there is no
    USDC-margined perp on Binance Futures).
  * On-disk Parquet path: ``data/parquet/binance/BASE-USDC/...``
    (we keep our internal symbol convention so the downloader / loader
    don't need to know that the underlying venue uses USDT).

The translation is contained inside ``_to_ccxt_symbol`` — consumers
NEVER see Binance's symbol shape. This is the same pattern as
``HyperliquidAdapter`` and matches the §rule "engine has zero CCXT
imports; all exchange logic lives in ``exchanges/*.py``".

Authentication: OHLCV is a public Binance endpoint, so no API key is
required for the data-download workflow. The constructor accepts a
``testnet`` flag for interface consistency with other adapters but
data-download mode does not switch venues — the public OHLCV endpoint
is the same on mainnet and testnet for historical klines.
"""

from __future__ import annotations

import logging

import ccxt

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from exchanges.factory import ExchangeFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------
#
# Programmatic rule: ``BASE-USDC`` → ``BASE/USDT:USDT`` for the symbols we
# care about today (BTC, ETH, SOL, etc.). The override dict is empty for
# now but kept as the documented escape hatch for any future symbols whose
# Binance ticker doesn't match the BASE prefix (e.g. wrapped variants).

SYMBOL_OVERRIDES: dict[str, str] = {}


_TRADING_NOT_IMPLEMENTED = (
    "BinanceAdapter is data-download only. Trading not yet implemented "
    "(Phase 5 Week 10)."
)


class BinanceAdapter(ExchangeAdapter):
    """Binance USDT-M perp futures adapter (data-download only)."""

    def __init__(self, testnet: bool = False) -> None:
        # No API keys: OHLCV is public. Even on testnet the historical
        # kline endpoint is unauthenticated.
        config: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        self._exchange = ccxt.binance(config)
        self._testnet = testnet
        if testnet:
            # set_sandbox_mode flips to https://testnet.binancefuture.com.
            # Historical OHLCV is identical between mainnet and testnet —
            # we expose the flag for interface symmetry only.
            try:
                self._exchange.set_sandbox_mode(True)
                logger.info("BinanceAdapter: TESTNET mode active")
            except Exception as e:  # pragma: no cover — defensive only
                logger.warning(f"BinanceAdapter: testnet toggle failed: {e}")

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "binance"

    def capabilities(self) -> AdapterCapabilities:
        # Reflects the long-term Binance feature set even though the
        # current implementation only exposes data download — capabilities
        # are read by config / sizing code that may run before trading is
        # enabled, and lying about the venue's properties would mislead
        # downstream cost models.
        return AdapterCapabilities(
            native_sl_tp=True,
            supports_short=True,
            market_hours=None,
            asset_types=["perpetual"],
            margin_type="cross",
            has_funding_rate=True,
            has_oi_data=True,
            max_leverage=125.0,
            order_types=["market", "limit", "stopMarket", "takeProfitMarket"],
            supports_partial_close=True,
        )

    # ------------------------------------------------------------------
    # Symbol conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        """Convert internal ``BTC-USDC`` → CCXT Binance perp ``BTC/USDT:USDT``.

        Already-CCXT-formatted strings (containing ``/``) pass through.
        ``-USDT`` is also accepted as a sibling form for callers that
        already know they want a USDT-quoted contract; both
        ``BTC-USDC`` and ``BTC-USDT`` map to ``BTC/USDT:USDT`` because
        Binance only ships a USDT-margined contract for these markets.
        """
        if "/" in symbol:
            return symbol
        if symbol in SYMBOL_OVERRIDES:
            return SYMBOL_OVERRIDES[symbol]
        if symbol.endswith("-USDC"):
            base = symbol[:-5]
        elif symbol.endswith("-USDT"):
            base = symbol[:-5]
        else:
            raise ValueError(
                f"BinanceAdapter: cannot convert symbol {symbol!r}; "
                "expected internal form BASE-USDC or BASE-USDT"
            )
        if not base:
            raise ValueError(
                f"BinanceAdapter: empty base in symbol {symbol!r}"
            )
        return f"{base}/USDT:USDT"

    @staticmethod
    def _from_ccxt_symbol(ccxt_symbol: str) -> str:
        """Convert CCXT ``BTC/USDT:USDT`` → internal ``BTC-USDC``.

        We canonicalise back to USDC to match the rest of the codebase
        (which speaks BASE-USDC end-to-end). The Parquet storage path
        uses the internal name, so consumers see ``BTC-USDC`` even
        though the underlying Binance contract is USDT-margined.
        """
        base = ccxt_symbol.split("/", 1)[0]
        return f"{base}-USDC"

    # ------------------------------------------------------------------
    # Data methods (the only ones actually implemented)
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: int | None = None,
    ) -> list[dict]:
        """Pull OHLCV from Binance USDT-M perp futures.

        Returns a list of dicts in the project's canonical shape:
        ``{timestamp, open, high, low, close, volume}`` — same as the
        Hyperliquid adapter so the downloader is venue-agnostic.

        Errors are caught and logged; an empty list is returned on
        failure so the downloader's pagination loop can decide whether
        to abort the month or move on.
        """
        ex_sym = self._to_ccxt_symbol(symbol)
        try:
            raw = self._exchange.fetch_ohlcv(
                ex_sym, timeframe, since=since, limit=limit
            )
            return [
                {
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                }
                for candle in raw
            ]
        except Exception as e:
            logger.error(f"BinanceAdapter.fetch_ohlcv failed for {symbol}: {e}")
            return []

    # ------------------------------------------------------------------
    # Trading surface — explicitly NOT implemented yet
    # ------------------------------------------------------------------
    #
    # Each method raises a clear error so a future caller that wires
    # Binance into the live executor will fail fast at the first call
    # site rather than silently no-op'ing or returning a stub.

    async def get_ticker(self, symbol: str) -> dict:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def get_balance(self) -> float:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def place_market_order(
        self, symbol: str, side: str, size: float
    ) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def place_limit_order(
        self, symbol: str, side: str, size: float, price: float
    ) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def place_sl_order(
        self, symbol: str, side: str, size: float, trigger_price: float
    ) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def place_tp_order(
        self, symbol: str, side: str, size: float, trigger_price: float
    ) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def cancel_all_orders(self, symbol: str) -> int:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def close_position(self, symbol: str) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        raise NotImplementedError(_TRADING_NOT_IMPLEMENTED)


# Register so ExchangeFactory.get_adapter("binance") works as soon as the
# module is imported. This matches the Hyperliquid registration pattern.
ExchangeFactory.register("binance", BinanceAdapter)
