#!/usr/bin/env bash
# Streamlit belgesel dashboard'unu başlatır.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sanal ortamı etkinleştir
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# .env yükle (varsa)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "🎬 Küratör Dashboard başlatılıyor..."
echo "   URL: http://localhost:8501"
echo "   Durdurmak için Ctrl+C"
echo ""

exec streamlit run dashboard.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#FF4B4B"
