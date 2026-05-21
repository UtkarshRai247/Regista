"""
Main Pipeline — The Architect Framework
Orchestrates the full analysis pipeline from data ingestion through
Architect Score assembly and validation.

Usage:
    python -m src.main                    # Run full pipeline
    python -m src.main --skip-ingestion   # Skip data pull (use cached)
    python -m src.main --quick            # Skip DS and DTD (event-based score only)
"""

import argparse
import time
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from src.data_ingestion import run_full_ingestion
from src.baseline_metrics import (
    compute_player_baselines, compute_chain_positions,
    compute_percentile_rankings, print_comparison
)
from src.novel_metrics import (
    build_pass_difficulty_model, compute_decision_surplus,
    compute_defensive_disruption, compute_prv, compute_cir,
    compute_tvi, assemble_architect_score
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def run_pipeline(skip_ingestion=False, quick=False):
    """Run the full Architect Framework pipeline."""
    start = time.time()

    print("=" * 70)
    print("THE ARCHITECT FRAMEWORK")
    print("Quantifying the Hidden Value of the Regista")
    print("Case Study: Granit Xhaka — Bayer Leverkusen 2023/24")
    print("=" * 70)

    # ─── STAGE 1: DATA INGESTION ───
    print(f"\n{'─'*70}")
    print("STAGE 1: Data Ingestion")
    print(f"{'─'*70}")

    if skip_ingestion and (RAW_DIR / "leverkusen_2324_events.parquet").exists():
        print("Loading cached data...")
        events = pd.read_parquet(RAW_DIR / "leverkusen_2324_events.parquet")
        passes = pd.read_parquet(PROCESSED_DIR / "passes_enriched.parquet")
        chains = pd.read_parquet(PROCESSED_DIR / "possession_chains.parquet")
        with open(RAW_DIR / "frame_lookup.pkl", 'rb') as f:
            frame_lookup = pickle.load(f)
    else:
        events, passes, chains, frame_lookup = run_full_ingestion()

    # ─── STAGE 2: BASELINES ───
    print(f"\n{'─'*70}")
    print("STAGE 2: Baseline Metrics")
    print(f"{'─'*70}")

    baselines = compute_player_baselines(events, passes)
    chain_profiles = compute_chain_positions(chains)

    # Print Xhaka's traditional rankings
    rankings = compute_percentile_rankings(baselines)
    print(f"\n  Xhaka Traditional Metric Rankings:")
    for metric, (val, pct) in rankings.items():
        print(f"    {metric:>25}: {val:>8.1f}  ({pct:.0f}th percentile)")

    # Print chain position comparison
    print_comparison(chain_profiles)

    # ─── STAGE 3: EVENT-BASED NOVEL METRICS ───
    print(f"\n{'─'*70}")
    print("STAGE 3: Event-Based Novel Metrics (PRV, CIR, TVI)")
    print(f"{'─'*70}")

    prv = compute_prv(passes)
    cir = compute_cir(chains)
    tvi = compute_tvi(events)

    # Print Xhaka's event-based metrics
    _print_metric_ranks(prv, 'mean_prv', 'PRV (mean pass value under pressure)')
    _print_metric_ranks(cir, 'cir', 'CIR (chain initiation rate)')
    _print_metric_ranks(tvi, 'tvi', 'TVI (tempo variance index)')

    # ─── STAGE 4: FREEZE-FRAME METRICS (DS, DTD) ───
    ds_df = None
    dtd_df = None

    if not quick:
        print(f"\n{'─'*70}")
        print("STAGE 4: Freeze-Frame Metrics (Decision Surplus, Defensive Disruption)")
        print(f"{'─'*70}")

        # 4a: Pass difficulty model
        diff_model, feat_cols = build_pass_difficulty_model(passes, frame_lookup)

        # 4b: Decision Surplus
        ds_df = compute_decision_surplus(passes, frame_lookup, diff_model, feat_cols)
        _print_ds_summary(ds_df)

        # 4c: Defensive Topology Disruption
        dtd_df = compute_defensive_disruption(events, frame_lookup, distance_threshold=15.0)
        _print_dtd_summary(dtd_df)
    else:
        print("\n  [QUICK MODE] Skipping DS and DTD computation")

    # ─── STAGE 5: ARCHITECT SCORE ASSEMBLY ───
    print(f"\n{'─'*70}")
    print("STAGE 5: Architect Score Assembly")
    print(f"{'─'*70}")

    scores = assemble_architect_score(baselines, ds_df, dtd_df, prv, cir, tvi, chains)
    _print_final_rankings(scores, quick=quick)

    # ─── STAGE 6: XHAKA vs WIRTZ vs ANDRICH ───
    print(f"\n{'─'*70}")
    print("STAGE 6: Internal Comparison — Xhaka vs Wirtz vs Andrich")
    print(f"{'─'*70}")
    _print_trio_comparison(scores, quick=quick)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    return scores


def _print_metric_ranks(df, col, label, n=10):
    """Print top N players for a metric."""
    top = df.nlargest(n, col)
    print(f"\n  Top {n} by {label}:")
    for _, row in top.iterrows():
        marker = " ◄" if 'Xhaka' in str(row['player']) else ""
        print(f"    {row['player']:<35} {row[col]:.4f}{marker}")


def _print_ds_summary(ds_df):
    """Print Decision Surplus summary for key players."""
    print(f"\n  Decision Surplus Summary:")
    for name in ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']:
        player = ds_df[ds_df['player'] == name]
        if len(player) > 0:
            print(f"    {name:<25} median DS: {player['decision_surplus'].median():.4f}"
                  f"  mean: {player['decision_surplus'].mean():.4f}"
                  f"  n={len(player)}")


def _print_dtd_summary(dtd_df):
    """Print Defensive Disruption summary for key players."""
    print(f"\n  Defensive Topology Disruption Summary:")
    for name in ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']:
        player = dtd_df[dtd_df['player'] == name]
        if len(player) > 0:
            print(f"    {name:<25} mean DTD: {player['dtd_raw'].mean():.4f}"
                  f"  n={len(player)}")


def _print_final_rankings(scores, quick=False):
    """Print final Architect Score rankings."""
    score_col = 'architect_score_event' if quick else 'architect_score_full'
    
    # Filter to players with enough data
    ranked = scores[scores['matches'] >= 5].copy()
    ranked = ranked.sort_values(score_col, ascending=False)

    print(f"\n  Top 15 by Architect Score ({'Event-Based' if quick else 'Full'}):")
    print(f"  {'Player':<35} {'Team':<20} {'AS':>6} {'PACV':>6} {'PRV':>6} {'CIR':>6} {'TVI':>6}", end="")
    if not quick:
        print(f" {'DS':>6} {'DTD':>6}", end="")
    print()
    print("  " + "-" * (95 if not quick else 83))

    for _, row in ranked.head(15).iterrows():
        marker = " ◄" if 'Xhaka' in str(row['player']) else ""
        print(f"  {row['player']:<35} {row['team']:<20} "
              f"{row[score_col]:>6.2f} {row['pacv_z']:>6.2f} {row['prv_z']:>6.2f} "
              f"{row['cir_z']:>6.2f} {row['tvi_z']:>6.2f}", end="")
        if not quick:
            print(f" {row['ds_z']:>6.2f} {row['dtd_z']:>6.2f}", end="")
        print(marker)


def _print_trio_comparison(scores, quick=False):
    """Print the Xhaka vs Wirtz vs Andrich comparison."""
    trio = ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']
    cols = ['pacv', 'prv', 'cir', 'tvi']
    z_cols = [f'{c}_z' for c in cols]
    
    if not quick:
        cols += ['ds', 'dtd']
        z_cols += ['ds_z', 'dtd_z']

    for name in trio:
        row = scores[scores['player'] == name]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        score_col = 'architect_score_event' if quick else 'architect_score_full'
        print(f"\n  {name}:")
        print(f"    Architect Score: {row[score_col]:.3f}")
        for c, z in zip(cols, z_cols):
            print(f"    {c.upper():<6} raw: {row[c]:>8.4f}  z-score: {row[z]:>6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Architect Framework")
    parser.add_argument('--skip-ingestion', action='store_true',
                        help='Skip data pull, use cached files')
    parser.add_argument('--quick', action='store_true',
                        help='Skip DS and DTD (event-based score only)')
    args = parser.parse_args()

    run_pipeline(skip_ingestion=args.skip_ingestion, quick=args.quick)
