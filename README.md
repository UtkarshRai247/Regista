# The Architect Framework

**Quantifying the Hidden Value of the Regista**  
*Case Study: Granit Xhaka — Bayer Leverkusen 2023/24*

## Research Question

> "How can we measure the full creative influence of midfielders whose primary contribution to attack occurs before the final pass — and do existing analytics frameworks systematically undervalue these players?"

## Key Findings

- **Xhaka's total chain value (6.09) nearly equals Wirtz's (6.50)** — but 80% of Xhaka's value comes from deep buildup (position 6+), while Wirtz concentrates near the final action
- **Decision Surplus statistically separates Xhaka from Andrich** (p = 4.34e-06) — Xhaka consistently finds passes more valuable than the alternatives available to him
- **DS is completely independent of progressive passing** (r = -0.040), proving it captures a fundamentally different dimension of creative value
- **Leverkusen created 2.14 xG/match with Xhaka vs 1.70 without** — a 26% reduction in the one match he missed
- **Andrich drops from rank 8 (traditional) to rank 15 (Architect Score)** — same position as Xhaka, fundamentally different role

## Quick Start

```bash
pip install statsbombpy pandas numpy scikit-learn xgboost pyarrow torch matplotlib mplsoccer seaborn networkx streamlit
python -m src.main
streamlit run app.py
```

## Status — Complete

- [x] Data pipeline (137K events, 118K freeze frames)
- [x] Transformer possession value model (61% improvement over baseline)
- [x] Decision Surplus (32,360 passes, p < 0.001 discrimination)
- [x] Defensive Topology Disruption (55,692 actions)
- [x] Full Architect Score (6 components)
- [x] Validation, visualizations, dashboard, report
