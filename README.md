# Treasury / Retirement Portfolio Manager

An educational Streamlit tool for designing, monitoring, and stress-testing a
diversified ETF portfolio. Two operating modes: **Corporate Treasury** (excess
USD cash, horizon-based) and **Personal Retirement**. Includes an optional
**Bank-Compliant Mode** that excludes all crypto assets per bank policy.

## Features
Risk-level targets built on an 8-fund ETF core (VTI, VXUS, AVUV, AVDV, VNQ,
GLDM, BND, VTIP) · satellite positions with per-name, crypto, and budget caps ·
5/25 drift monitoring with tax-aware order suggestions · look-through
concentration analysis · diversification metrics and correlation heatmap ·
glide-path simulator · scenario stress tests · Monte Carlo projections ·
risk-level comparison · rebalancing cost estimator · asset-location guide ·
historical backtest (yfinance) · portfolio health score · CSV / IPS exports ·
save & load configurations · Conservative / Base / Optimistic return
assumption sets.

## Run locally
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run portfolio_manager.py
```

## Deploy (shareable link)
Push `portfolio_manager.py` + `requirements.txt` to a GitHub repo, then create
an app at share.streamlit.io pointing at it. No secrets or API keys required.
Consider making the app private before sharing configurations with real data.

## Important disclaimer
**Educational tool only — not investment advice.** Nothing in this application
constitutes investment, tax, or legal advice, a recommendation, or an offer or
solicitation to buy or sell any security. All figures are model estimates based
on user-adjustable assumptions; they are not forecasts. Past performance does
not guarantee future results. Investing involves risk, including loss of
principal. Compliance with any institution's policies and applicable law is the
user's responsibility. Consult licensed financial, tax, and legal professionals
before acting.
