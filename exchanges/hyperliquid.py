"""Hyperliquid adapter: native SL/TP, HIP-3, WalletConnect.

Primary exchange. Supports:
- Regular perpetuals (BTC, ETH, SOL, etc.)
- HIP-3 synthetic markets (commodities, indices, stocks, forex via XYZ deployer)
- Native stop-loss and take-profit orders
- Subaccounts via subAccountAddress
"""

from __future__ import annotations

import logging
import os

import ccxt

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from exchanges.factory import ExchangeFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol mapping — internal (BASE-USDC) ↔ CCXT (BASE/USDC:USDC)
# ---------------------------------------------------------------------------

SYMBOL_MAP: dict[str, str] = {
    # Regular perpetuals
    "BTC-USDC": "BTC/USDC:USDC",
    "ETH-USDC": "ETH/USDC:USDC",
    "SOL-USDC": "SOL/USDC:USDC",
    "DOGE-USDC": "DOGE/USDC:USDC",
    "AVAX-USDC": "AVAX/USDC:USDC",
    "LINK-USDC": "LINK/USDC:USDC",
    "HYPE-USDC": "HYPE/USDC:USDC",
    # HIP-3 Commodities
    "GOLD-USDC": "XYZ-GOLD/USDC:USDC",
    "SILVER-USDC": "XYZ-SILVER/USDC:USDC",
    "WTIOIL-USDC": "XYZ-CL/USDC:USDC",
    "BRENTOIL-USDC": "XYZ-BRENTOIL/USDC:USDC",
    "NATGAS-USDC": "XYZ-NATGAS/USDC:USDC",
    "COPPER-USDC": "XYZ-COPPER/USDC:USDC",
    "PLATINUM-USDC": "XYZ-PLATINUM/USDC:USDC",
    "PALLADIUM-USDC": "XYZ-PALLADIUM/USDC:USDC",
    "URANIUM-USDC": "XYZ-URANIUM/USDC:USDC",
    "WHEAT-USDC": "XYZ-WHEAT/USDC:USDC",
    "CORN-USDC": "XYZ-CORN/USDC:USDC",
    "ALUMINIUM-USDC": "XYZ-ALUMINIUM/USDC:USDC",
    # HIP-3 Indices
    "SP500-USDC": "XYZ-SP500/USDC:USDC",
    "JP225-USDC": "XYZ-JP225/USDC:USDC",
    "VIX-USDC": "XYZ-VIX/USDC:USDC",
    "DXY-USDC": "XYZ-DXY/USDC:USDC",
    # HIP-3 Stocks
    "TSLA-USDC": "XYZ-TSLA/USDC:USDC",
    "NVDA-USDC": "XYZ-NVDA/USDC:USDC",
    "AAPL-USDC": "XYZ-AAPL/USDC:USDC",
    "META-USDC": "XYZ-META/USDC:USDC",
    "MSFT-USDC": "XYZ-MSFT/USDC:USDC",
    "GOOGL-USDC": "XYZ-GOOGL/USDC:USDC",
    "AMZN-USDC": "XYZ-AMZN/USDC:USDC",
    "AMD-USDC": "XYZ-AMD/USDC:USDC",
    "NFLX-USDC": "XYZ-NFLX/USDC:USDC",
    "PLTR-USDC": "XYZ-PLTR/USDC:USDC",
    "COIN-USDC": "XYZ-COIN/USDC:USDC",
    "MSTR-USDC": "XYZ-MSTR/USDC:USDC",
    # HIP-3 Forex
    "EUR-USDC": "XYZ-EUR/USDC:USDC",
    "JPY-USDC": "XYZ-JPY/USDC:USDC",
}

# Reverse map for converting CCXT symbols back to internal format
_REVERSE_MAP: dict[str, str] = {v: k for k, v in SYMBOL_MAP.items()}

# CCXT symbols that require dex='xyz' param
HIP3_SYMBOLS: set[str] = {v for v in SYMBOL_MAP.values() if v.startswith("XYZ-")}


def _pos_size(pos: dict) -> float:
    """Extract position size from a CCXT position dict.

    CCXT normalizes into 'contracts', but HIP-3 may have it in info.szi.
    """
    contracts = pos.get("contracts")
    if contracts is not None:
        try:
            val = float(contracts)
            if val != 0:
                return abs(val)
        except (TypeError, ValueError):
            pass
    szi = pos.get("info", {}).get("szi")
    if szi is not None:
        try:
            return abs(float(szi))
        except (TypeError, ValueError):
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class HyperliquidAdapter(ExchangeAdapter):
    """Hyperliquid exchange adapter with native SL/TP and HIP-3 support."""

    def __init__(
        self,
        wallet_address: str | None = None,
        private_key: str | None = None,
        testnet: bool = False,
        subaccount_address: str | None = None,
    ) -> None:
        # Resolve testnet FIRST so credential lookup can prefer testnet env vars.
        if not testnet:
            testnet = os.environ.get("HYPERLIQUID_TESTNET", "").lower() in ("true", "1", "yes")

        # Read from env if not explicitly passed. In testnet mode, prefer the
        # dedicated `HYPERLIQUID_TESTNET_*` vars and fall back to the mainnet
        # vars for backward compatibility — operators can keep both pairs in
        # `.env` and the right pair gets picked based on `testnet`. This keeps
        # mainnet keys away from testnet runs and vice versa.
        if testnet:
            wallet_address = (
                wallet_address
                or os.environ.get("HYPERLIQUID_TESTNET_WALLET_ADDRESS")
                or os.environ.get("HYPERLIQUID_WALLET_ADDRESS")
            )
            private_key = (
                private_key
                or os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY")
                or os.environ.get("HYPERLIQUID_PRIVATE_KEY")
            )
        else:
            wallet_address = wallet_address or os.environ.get("HYPERLIQUID_WALLET_ADDRESS")
            private_key = private_key or os.environ.get("HYPERLIQUID_PRIVATE_KEY")

        config: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        if wallet_address:
            config["walletAddress"] = wallet_address
        if private_key:
            config["privateKey"] = private_key
        if subaccount_address:
            config["options"]["subAccountAddress"] = subaccount_address

        self._exchange = ccxt.hyperliquid(config)
        self._exchange.options["defaultSlippage"] = 0.05
        self._testnet = testnet

        if testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("HyperliquidAdapter: TESTNET mode active")

    def name(self) -> str:
        return "hyperliquid"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True,
            supports_short=True,
            market_hours=None,
            asset_types=["perpetual", "spot"],
            margin_type="cross",
            has_funding_rate=True,
            has_oi_data=True,
            max_leverage=50.0,
            order_types=["market", "limit", "stopMarket", "takeProfit"],
            supports_partial_close=True,
        )

    # ------------------------------------------------------------------
    # Symbol conversion
    # ------------------------------------------------------------------

    def _to_ccxt_symbol(self, symbol: str) -> str:
        """Convert internal symbol (BTC-USDC) to CCXT format (BTC/USDC:USDC)."""
        if "/" in symbol:
            return symbol
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]
        if symbol.endswith("-USDC"):
            base = symbol[:-5]
            candidate = f"{base}/USDC:USDC"
            return candidate
        raise ValueError(f"Cannot convert symbol: {symbol}")

    def _from_ccxt_symbol(self, ccxt_symbol: str) -> str:
        """Convert CCXT symbol back to internal format."""
        if ccxt_symbol in _REVERSE_MAP:
            return _REVERSE_MAP[ccxt_symbol]
        base_part = ccxt_symbol.split("/")[0]
        if "-" in base_part:
            _, asset = base_part.split("-", 1)
        else:
            asset = base_part
        return f"{asset}-USDC"

    def _hip3_params(self, ccxt_symbol: str) -> dict:
        """Return extra params for HIP-3 markets."""
        if ccxt_symbol in HIP3_SYMBOLS or ccxt_symbol.startswith("XYZ-"):
            return {"dex": "xyz"}
        return {}

    # ------------------------------------------------------------------
    # Data methods
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: int | None = None,
    ) -> list[dict]:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            raw = self._exchange.fetch_ohlcv(
                ex_sym, timeframe, since=since, limit=limit, params=extra
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
            logger.error(f"fetch_ohlcv failed for {symbol}: {e}")
            return []

    async def get_ticker(self, symbol: str) -> dict:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            ticker = self._exchange.fetch_ticker(ex_sym, extra)
            return {
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "last": ticker.get("last"),
                "volume": ticker.get("baseVolume"),
            }
        except Exception as e:
            logger.error(f"get_ticker failed for {symbol}: {e}")
            return {}

    async def get_balance(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            total = balance.get("total", {})
            usdc = total.get("USDC", 0)
            if usdc:
                return float(usdc)
            return float(total.get("USDT0", 0) or 0)
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return 0.0

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        result: list[Position] = []
        try:
            if symbol:
                ex_sym = self._to_ccxt_symbol(symbol)
                extra = self._hip3_params(ex_sym)
                raw_positions = self._exchange.fetch_positions([ex_sym], extra) or []
            else:
                raw_positions = self._exchange.fetch_positions() or []
        except Exception as e:
            logger.error(f"get_positions (perp) failed: {e}")
            raw_positions = []

        for pos in raw_positions:
            size = _pos_size(pos)
            if size > 0:
                result.append(Position(
                    symbol=self._from_ccxt_symbol(pos.get("symbol", "")),
                    direction=pos.get("side", "long"),
                    size=size,
                    entry_price=float(pos.get("entryPrice") or pos.get("info", {}).get("entryPx") or 0),
                    unrealized_pnl=float(pos.get("unrealizedPnl") or 0),
                    leverage=float(pos.get("leverage") or 0) or None,
                ))

        # Also fetch HIP-3 positions if no specific symbol
        if not symbol:
            try:
                hip3 = self._exchange.fetch_positions(params={"dex": "xyz"}) or []
                for pos in hip3:
                    size = _pos_size(pos)
                    if size > 0:
                        result.append(Position(
                            symbol=self._from_ccxt_symbol(pos.get("symbol", "")),
                            direction=pos.get("side", "long"),
                            size=size,
                            entry_price=float(pos.get("entryPrice") or pos.get("info", {}).get("entryPx") or 0),
                            unrealized_pnl=float(pos.get("unrealizedPnl") or 0),
                            leverage=float(pos.get("leverage") or 0) or None,
                        ))
            except Exception as e:
                logger.warning(f"get_positions (HIP-3) failed (non-fatal): {e}")

        return result

    # ------------------------------------------------------------------
    # Order methods
    # ------------------------------------------------------------------

    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            ticker = self._exchange.fetch_ticker(ex_sym, extra)
            price = float(ticker.get("last", 0))
            order = self._exchange.create_order(ex_sym, "market", side, size, price, extra)
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                fill_price=float(order.get("average") or order.get("price") or price),
                fill_size=float(order.get("filled") or size),
                error=None,
            )
        except Exception as e:
            logger.error(f"place_market_order failed for {symbol}: {e}")
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error=str(e))

    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            order = self._exchange.create_order(ex_sym, "limit", side, size, price, extra)
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                fill_price=float(order.get("price") or price),
                fill_size=float(order.get("filled") or 0),
                error=None,
            )
        except Exception as e:
            logger.error(f"place_limit_order failed for {symbol}: {e}")
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error=str(e))

    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            order = self._exchange.create_order(
                ex_sym, "stop", side, size, trigger_price,
                {"stopPrice": trigger_price, "triggerPrice": trigger_price, "reduceOnly": True, **extra},
            )
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                fill_price=None,
                fill_size=None,
                error=None,
            )
        except Exception as e:
            logger.error(f"place_sl_order failed for {symbol}: {e}")
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error=str(e))

    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            order = self._exchange.create_order(
                ex_sym, "take_profit", side, size, trigger_price,
                {"takeProfitPrice": trigger_price, "triggerPrice": trigger_price, "reduceOnly": True, **extra},
            )
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                fill_price=None,
                fill_size=None,
                error=None,
            )
        except Exception as e:
            logger.error(f"place_tp_order failed for {symbol}: {e}")
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error=str(e))

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            self._exchange.cancel_order(order_id, ex_sym, extra)
            return True
        except Exception as e:
            logger.error(f"cancel_order failed for {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        cancelled = 0
        try:
            orders = self._exchange.fetch_open_orders(ex_sym, params=extra)
            for o in orders:
                try:
                    self._exchange.cancel_order(o["id"], ex_sym, extra)
                    cancelled += 1
                except Exception as e:
                    logger.warning(f"Failed to cancel order {o['id']}: {e}")
        except Exception as e:
            logger.error(f"cancel_all_orders failed for {symbol}: {e}")
        return cancelled

    async def close_position(self, symbol: str) -> OrderResult:
        positions = await self.get_positions(symbol)
        if not positions:
            return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)
        pos = positions[0]
        close_side = "sell" if pos.direction == "long" else "buy"
        return await self.place_market_order(symbol, close_side, pos.size)

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        # Cancel existing SL orders, then place new one
        try:
            orders = self._exchange.fetch_open_orders(ex_sym, params=extra)
            for o in orders:
                if o.get("type") in ("stop", "stopMarket") or o.get("info", {}).get("orderType") in ("Stop Market",):
                    try:
                        self._exchange.cancel_order(o["id"], ex_sym, extra)
                    except Exception as e:
                        logger.warning(f"Failed to cancel old SL {o['id']}: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch orders for SL modify: {e}")

        positions = await self.get_positions(symbol)
        if not positions:
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="No position found")
        pos = positions[0]
        sl_side = "sell" if pos.direction == "long" else "buy"
        return await self.place_sl_order(symbol, sl_side, pos.size, new_price)

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        ex_sym = self._to_ccxt_symbol(symbol)
        extra = self._hip3_params(ex_sym)
        try:
            orders = self._exchange.fetch_open_orders(ex_sym, params=extra)
            for o in orders:
                if o.get("type") in ("take_profit", "takeProfit") or o.get("info", {}).get("orderType") in ("Take Profit Market",):
                    try:
                        self._exchange.cancel_order(o["id"], ex_sym, extra)
                    except Exception as e:
                        logger.warning(f"Failed to cancel old TP {o['id']}: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch orders for TP modify: {e}")

        positions = await self.get_positions(symbol)
        if not positions:
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="No position found")
        pos = positions[0]
        tp_side = "sell" if pos.direction == "long" else "buy"
        return await self.place_tp_order(symbol, tp_side, pos.size, new_price)

    # ------------------------------------------------------------------
    # Optional flow data
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float | None:
        ex_sym = self._to_ccxt_symbol(symbol)
        try:
            rate = self._exchange.fetch_funding_rate(ex_sym)
            return float(rate.get("fundingRate", 0))
        except Exception as e:
            logger.warning(f"get_funding_rate failed for {symbol}: {e}")
            return None

    async def get_open_interest(self, symbol: str) -> float | None:
        ex_sym = self._to_ccxt_symbol(symbol)
        try:
            oi = self._exchange.fetch_open_interest(ex_sym)
            return float(oi.get("openInterestAmount", 0))
        except Exception as e:
            logger.warning(f"get_open_interest failed for {symbol}: {e}")
            return None

    async def fetch_meta(self) -> list[dict]:
        """Fetch asset metadata from Hyperliquid info endpoint."""
        try:
            markets = self._exchange.load_markets()
            result = []
            for sym, market in markets.items():
                info = market.get("info", {})
                is_hip3 = sym.startswith("XYZ-") or "-XYZ" in sym
                result.append({
                    "symbol": market.get("id", sym),
                    "deployer_fee_scale": float(info.get("deployerFeeScale", 0) or 0),
                    "growth_mode": bool(info.get("growthMode", False)),
                    "is_hip3": is_hip3,
                })
            return result
        except Exception as e:
            logger.warning(f"fetch_meta failed: {e}")
            return []

    async def fetch_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """Fetch L2 orderbook snapshot."""
        ex_sym = self._to_ccxt_symbol(symbol)
        try:
            book = self._exchange.fetch_order_book(ex_sym, limit=limit)
            return {
                "bids": [[float(p), float(q)] for p, q in book.get("bids", [])[:limit]],
                "asks": [[float(p), float(q)] for p, q in book.get("asks", [])[:limit]],
            }
        except Exception as e:
            logger.warning(f"fetch_orderbook failed for {symbol}: {e}")
            return {"bids": [], "asks": []}

    async def fetch_user_fees(self) -> dict:
        """Fetch user's fee tier and discounts.

        Hyperliquid CCXT doesn't expose this directly, so we return
        tier 0 defaults. Override via cost model set_* methods for testing.
        """
        return {"tier": 0, "staking_discount": 0.0, "referral_discount": 0.0}


# Register with factory
ExchangeFactory.register("hyperliquid", HyperliquidAdapter)
