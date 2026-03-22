#!/bin/bash
cd /home/ubuntu/curator-documentary
source venv/bin/activate
streamlit run dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
