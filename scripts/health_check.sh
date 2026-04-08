#!/usr/bin/env bash
# QuantAgent v2 — Health check
#
# Usage:
#   ./scripts/health_check.sh              # check localhost
#   ./scripts/health_check.sh host:port    # check remote

set -euo pipefail

HOST="${1:-127.0.0.1:8000}"
URL="http://$HOST/v1/health"

echo "Checking $URL ..."
echo ""

RESPONSE=$(curl -sf "$URL" 2>/dev/null) || {
    echo "FAILED: Could not reach $URL"
    exit 1
}

echo "$RESPONSE" | python3 -m json.tool

STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
VERSION=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('engine_version','unknown'))")

echo ""
echo "Status:  $STATUS"
echo "Version: $VERSION"

if [ "$STATUS" = "healthy" ]; then
    exit 0
else
    exit 1
fi
