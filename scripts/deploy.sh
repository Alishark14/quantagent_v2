#!/usr/bin/env bash
# QuantAgent v2 — One-command production deploy
#
# Usage:
#   ./scripts/deploy.sh                    # deploy to default server
#   ./scripts/deploy.sh user@host          # deploy to specific server
#   ./scripts/deploy.sh user@host branch   # deploy specific branch
#
# What it does:
#   1. SSH to server
#   2. Pull latest code
#   3. Install/update dependencies
#   4. Run database migrations
#   5. Restart the service
#   6. Wait for health check

set -euo pipefail

SERVER="${1:-quantagent@your-server.com}"
BRANCH="${2:-main}"
DEPLOY_DIR="/opt/quantagent"
SERVICE="quantagent"

echo "=== QuantAgent Deploy ==="
echo "Server:  $SERVER"
echo "Branch:  $BRANCH"
echo "Dir:     $DEPLOY_DIR"
echo ""

ssh "$SERVER" bash -s "$BRANCH" "$DEPLOY_DIR" "$SERVICE" << 'REMOTE_SCRIPT'
set -euo pipefail

BRANCH="$1"
DEPLOY_DIR="$2"
SERVICE="$3"

cd "$DEPLOY_DIR"

echo "[1/5] Pulling latest code..."
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

echo "[2/5] Installing dependencies..."
.venv/bin/pip install -e ".[dev]" --quiet

echo "[3/5] Running database migrations..."
.venv/bin/python -m quantagent migrate

echo "[4/5] Restarting service..."
sudo systemctl restart "$SERVICE"

echo "[5/5] Waiting for health check..."
sleep 3

for i in 1 2 3 4 5; do
    STATUS=$(curl -sf http://127.0.0.1:8000/v1/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unreachable")
    if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "degraded" ]; then
        echo ""
        echo "=== Deploy complete! Status: $STATUS ==="
        curl -s http://127.0.0.1:8000/v1/health | python3 -m json.tool
        exit 0
    fi
    echo "  Attempt $i: $STATUS — retrying in 3s..."
    sleep 3
done

echo "ERROR: Health check failed after 5 attempts"
echo "Check logs: sudo journalctl -u $SERVICE -n 50 --no-pager"
exit 1
REMOTE_SCRIPT

echo ""
echo "Deploy finished."
