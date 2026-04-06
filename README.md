![CI](https://github.com/Alishark14/quantshark_v2/actions/workflows/ci.yml/badge.svg)

# QuantAgent v2

Modular AI-powered trading engine. Event-driven architecture with pluggable signal agents (LLM + ML), exchange adapters, and a conviction-based execution pipeline.

## Architecture

```
SENTINEL (persistent, code-only, per symbol)
  │ monitors price, computes readiness, manages SL/TP
  │
  └── ANALYSIS PIPELINE (spawns TraderBot)
        ├── 1. DATA LAYER      (OHLCV, indicators, flow, parent TF)
        ├── 2. SIGNAL LAYER    (3 LLM + N ML agents via SignalProducer)
        ├── 3. CONVICTION      (meta-evaluator, scores 0-1)
        ├── 4. EXECUTION       (action selection + mechanical safety)
        └── 5. REFLECTION      (async, post-trade rule distillation)
```

## Setup

```bash
# Clone
git clone https://github.com/Alishark14/quantshark_v2.git
cd quantshark_v2

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env  # fill in API keys

# Run tests
pytest tests/
```

## Commands

```bash
# Start the full server (BotRunner + API)
python -m quantagent run

# Run database migrations
python -m quantagent migrate

# Seed dev database
python -m quantagent seed
```

## API

The API runs on port 8000 by default. Auth via `X-API-Key` header.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/bots` | POST | Create bot config |
| `/v1/bots` | GET | List user's bots |
| `/v1/bots/{id}` | GET | Bot detail + last cycle |
| `/v1/bots/{id}` | DELETE | Stop a bot |
| `/v1/bots/{id}/analyze` | POST | Trigger manual analysis |
| `/v1/trades` | GET | Recent trades (filters: bot_id, symbol, limit) |
| `/v1/trades/{id}` | GET | Trade detail |
| `/v1/positions` | GET | Open positions |
| `/v1/rules` | GET | Active reflection rules |
| `/v1/health` | GET | System health (no auth) |

## Tech Stack

- **Python 3.12+**, async/await throughout
- **Claude API** (Anthropic) for signal + conviction agents
- **FastAPI** + uvicorn for the API layer
- **PostgreSQL** (asyncpg) / SQLite (dev fallback)
- **Alembic** for database migrations
- **CCXT** for exchange connectivity (Hyperliquid primary)

## Project Structure

```
engine/          Pure library — zero web/DB imports
sentinel/        Persistent market monitors
exchanges/       One adapter per exchange
llm/             LLM provider abstraction
storage/         Repository pattern (PostgreSQL / SQLite)
tracking/        Observability + data moat capture
api/             FastAPI web layer
quantagent/      CLI + BotRunner
```

## License

Proprietary. All rights reserved.
