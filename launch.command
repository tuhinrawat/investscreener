#!/bin/bash
# ─────────────────────────────────────────────
#  NSE Swing Screener — Mac launcher
#  Double-click this file in Finder to start.
# ─────────────────────────────────────────────

# Always cd to the folder this script lives in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "═══════════════════════════════════"
echo "   NSE Swing Screener"
echo "═══════════════════════════════════"

# Kill any stale instance on port 8501
if lsof -ti:8501 > /dev/null 2>&1; then
    echo "→ Stopping previous instance..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Activate project venv
source venv/bin/activate

# Start Streamlit in background, log to /tmp
echo "→ Starting Streamlit..."
nohup streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > /tmp/screener_streamlit.log 2>&1 &

SPID=$!
echo "  PID: $SPID  (logs: /tmp/screener_streamlit.log)"

# Wait until the server responds (up to 15 s)
echo "→ Waiting for server to be ready..."
for i in {1..15}; do
    if curl -s -o /dev/null http://localhost:8501; then
        echo "  Ready after ${i}s"
        break
    fi
    sleep 1
done

# Open browser
echo "→ Opening browser..."
open http://localhost:8501

echo ""
echo "Screener is running at http://localhost:8501"
echo "Close this window to keep it running in the background."
echo "To stop it: lsof -ti:8501 | xargs kill -9"
