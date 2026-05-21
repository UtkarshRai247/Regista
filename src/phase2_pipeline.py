"""
Phase 2 Pipeline — The Architect Framework
Trains the pass difficulty model on combined tournament data,
then runs all Phase 2 analytics on Euro 2024 + Euro 2020.

Produces per-player metric files for score assembly.
"""

import sys
import os
import shutil
import time
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature columns used by the optimised Decision Surplus module ─────────────
FEAT_COLS = [
    'start_x', 'start_y', 'end_x', 'end_y',
    'pass_length', 'pass_angle', 'dist_toward_goal', 'lateral_dist',
    'defenders_on_line', 'defenders_near_target',
    'closest_defender', 'is_under_pressure', 'is_ground_pass',
    'n_visible', 'n_teammates', 'n_opponents',
]

TURNOVER_PENALTY = -0.05


# ─── Positional value ─────────────────────────────────────────────────────────

def pos_value(x, y):
    """Positional value proxy — consistent with decision_surplus.py."""
    x_val = (x / 120.0) ** 1.5
    y_center = abs(y - 40) / 40
    y_penalty = 1 - 0.3 * y_center
    return x_val * y_penalty


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_tournament_data(tournament_name):
    """
    Load events + freeze frames for a tournament.
    Returns (events_df, frame_lookup).
    """
    events_path = RAW_DIR / f"{tournament_name}_events.parquet"
    frames_path = RAW_DIR / f"{tournament_name}_frames.pkl"

    if not events_path.exists():
        raise FileNotFoundError(f"Events not found: {events_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames not found: {frames_path}")

    print(f"  Loading {tournament_name} events...")
    events_df = pd.read_parquet(events_path)

    print(f"  Loading {tournament_name} frames (this may take a moment)...")
    with open(frames_path, "rb") as f:
        frame_lookup = pickle.load(f)

    print(f"  {tournament_name}: {len(events_df):,} events, {len(frame_lookup):,} freeze frames")
    return events_df, frame_lookup


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Pass Enrichment
# ══════════════════════════════════════════════════════════════════════════════

def enrich_passes(events_df):
    """
    Enrich pass events with derived columns.
    Same logic as data_ingestion.enrich_passes(), adapted for multiple tournaments.
    """
    passes = events_df[events_df['type'] == 'Pass'].copy()

    # ── Coordinates ───────────────────────────────────────────────────────────
    def _x(loc):
        try:
            return float(loc[0])
        except Exception:
            return np.nan

    def _y(loc):
        try:
            return float(loc[1])
        except Exception:
            return np.nan

    passes['start_x'] = passes['location'].apply(_x)
    passes['start_y'] = passes['location'].apply(_y)
    passes['end_x'] = passes['pass_end_location'].apply(_x)
    passes['end_y'] = passes['pass_end_location'].apply(_y)

    # ── Derived geometry ──────────────────────────────────────────────────────
    passes['dist_toward_goal'] = passes['end_x'] - passes['start_x']
    passes['is_progressive'] = passes['dist_toward_goal'] >= 10
    passes['is_completed'] = passes['pass_outcome'].isna()
    passes['lateral_dist'] = (passes['end_y'] - passes['start_y']).abs()

    # pass_length & pass_angle — use existing columns if present, else compute
    if 'pass_length' not in passes.columns or passes['pass_length'].isna().all():
        dx = passes['end_x'] - passes['start_x']
        dy = passes['end_y'] - passes['start_y']
        passes['pass_length'] = np.sqrt(dx ** 2 + dy ** 2)
    if 'pass_angle' not in passes.columns or passes['pass_angle'].isna().all():
        dx = passes['end_x'] - passes['start_x']
        dy = passes['end_y'] - passes['start_y']
        passes['pass_angle'] = np.arctan2(dy, dx)

    # Fill NaN pass_length/pass_angle from coordinates
    mask_nan_len = passes['pass_length'].isna()
    mask_nan_ang = passes['pass_angle'].isna()
    if mask_nan_len.any():
        dx = passes.loc[mask_nan_len, 'end_x'] - passes.loc[mask_nan_len, 'start_x']
        dy = passes.loc[mask_nan_len, 'end_y'] - passes.loc[mask_nan_len, 'start_y']
        passes.loc[mask_nan_len, 'pass_length'] = np.sqrt(dx ** 2 + dy ** 2)
    if mask_nan_ang.any():
        dx = passes.loc[mask_nan_ang, 'end_x'] - passes.loc[mask_nan_ang, 'start_x']
        dy = passes.loc[mask_nan_ang, 'end_y'] - passes.loc[mask_nan_ang, 'start_y']
        passes.loc[mask_nan_ang, 'pass_angle'] = np.arctan2(dy, dx)

    # ── Boolean flags ─────────────────────────────────────────────────────────
    passes['is_switch'] = passes['pass_switch'].fillna(False).astype(bool) if 'pass_switch' in passes.columns else False
    passes['is_through_ball'] = passes['pass_through_ball'].fillna(False).astype(bool) if 'pass_through_ball' in passes.columns else False
    passes['is_cross'] = passes['pass_cross'].fillna(False).astype(bool) if 'pass_cross' in passes.columns else False
    passes['is_under_pressure'] = passes['under_pressure'].fillna(False).astype(bool)

    out_path = PROCESSED_DIR / "phase2_passes_enriched.parquet"
    passes.to_parquet(out_path, index=False)
    print(f"  Saved {len(passes):,} enriched passes to {out_path}")
    return passes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Pass Difficulty Model Training
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_for_training(passes_df, frame_lookup):
    """
    Build training rows from passes that have freeze frames.
    Returns (X_df, y_array).
    """
    from src.decision_surplus import compute_spatial_features_batch, preprocess_freeze_frames

    pass_ids = passes_df['id'].values
    print(f"    Pre-processing {len(pass_ids):,} pass IDs against {len(frame_lookup):,} frames...")
    ff_data = preprocess_freeze_frames(frame_lookup, pass_ids)
    print(f"    Got {len(ff_data):,} usable freeze frames")

    valid = passes_df[passes_df['id'].isin(ff_data.keys())].copy()
    rows = []
    targets = []

    for _, p in valid.iterrows():
        ff = ff_data[p['id']]
        sx, sy = p['start_x'], p['start_y']
        ex, ey = p['end_x'], p['end_y']
        if np.isnan(sx) or np.isnan(ex):
            continue

        spatial = compute_spatial_features_batch(
            [sx, sy],
            np.array([[ex, ey]]),
            ff['opponents'],
            FEAT_COLS,
        )

        rows.append({
            'start_x': sx,
            'start_y': sy,
            'end_x': ex,
            'end_y': ey,
            'pass_length': float(p['pass_length']) if not pd.isna(p['pass_length']) else spatial['pass_length'][0],
            'pass_angle': float(p['pass_angle']) if not pd.isna(p['pass_angle']) else spatial['pass_angle'][0],
            'dist_toward_goal': spatial['dist_toward_goal'][0],
            'lateral_dist': spatial['lateral_dist'][0],
            'defenders_on_line': spatial['defenders_on_line'][0],
            'defenders_near_target': spatial['defenders_near_target'][0],
            'closest_defender': spatial['closest_defender'][0],
            'is_under_pressure': int(p['is_under_pressure']),
            'is_ground_pass': int(float(p['pass_length']) <= 35 if not pd.isna(p['pass_length']) else spatial['pass_length'][0] <= 35),
            'n_visible': ff['n_total'],
            'n_teammates': ff['n_teammates'],
            'n_opponents': ff['n_opponents'],
        })
        targets.append(int(p['is_completed']))

    X = pd.DataFrame(rows, columns=FEAT_COLS)
    y = np.array(targets)
    return X, y


def train_pass_difficulty_model(passes_df, frame_lookup, extra_frames=None):
    """
    Train 16-feature GBM predicting pass completion probability.
    Combines passes from all available datasets that have freeze frames.
    Saves model + feature list to models/.
    Returns (model, feat_cols).
    """
    print("\n" + "=" * 70)
    print("TRAINING PASS DIFFICULTY MODEL")
    print("=" * 70)

    all_X = []
    all_y = []

    # Primary tournament passes
    print("\n  Building training features from tournament passes...")
    X_t, y_t = _preprocess_for_training(passes_df, frame_lookup)
    all_X.append(X_t)
    all_y.append(y_t)
    print(f"  Tournament passes: {len(X_t):,} rows ({y_t.mean()*100:.1f}% completed)")

    # Leverkusen passes (if available)
    lev_passes_path = PROCESSED_DIR / "passes_enriched.parquet"
    lev_frames_path = RAW_DIR / "frame_lookup.pkl"
    if lev_passes_path.exists() and lev_frames_path.exists():
        print("\n  Adding Leverkusen passes for richer training data...")
        lev_passes = pd.read_parquet(lev_passes_path)
        with open(lev_frames_path, "rb") as f:
            lev_frames = pickle.load(f)
        X_l, y_l = _preprocess_for_training(lev_passes, lev_frames)
        all_X.append(X_l)
        all_y.append(y_l)
        print(f"  Leverkusen passes: {len(X_l):,} rows ({y_l.mean()*100:.1f}% completed)")

    # Extra frames (e.g., another tournament)
    if extra_frames is not None:
        for name, (ep, ef) in extra_frames.items():
            print(f"\n  Adding {name} passes...")
            X_e, y_e = _preprocess_for_training(ep, ef)
            all_X.append(X_e)
            all_y.append(y_e)
            print(f"  {name}: {len(X_e):,} rows")

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)

    print(f"\n  Total training set: {len(X):,} passes ({y.mean()*100:.1f}% completed)")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )

    t0 = time.time()
    print("  Fitting GBM (n_estimators=200)...")
    model.fit(X[FEAT_COLS], y)
    elapsed = time.time() - t0

    preds = model.predict_proba(X[FEAT_COLS])[:, 1]
    auc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)
    print(f"  Training complete in {elapsed:.0f}s")
    print(f"  Training AUC:    {auc:.4f}")
    print(f"  Brier score:     {brier:.4f}")

    joblib.dump(model, MODELS_DIR / "pass_difficulty_model.pkl")
    joblib.dump(FEAT_COLS, MODELS_DIR / "pass_difficulty_features.pkl")
    print(f"  Saved model to {MODELS_DIR / 'pass_difficulty_model.pkl'}")
    print(f"  Saved feat_cols to {MODELS_DIR / 'pass_difficulty_features.pkl'}")

    return model, FEAT_COLS


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Possession Chain Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_chains(events_df):
    """
    Extract possession chains from combined tournament events.
    Same logic as data_ingestion.extract_possession_chains().
    """
    print("\nExtracting possession chains...")

    action_types = [
        'Pass', 'Carry', 'Shot', 'Dribble', 'Ball Receipt*',
        'Clearance', 'Miscontrol', 'Dispossessed', 'Interception',
        'Ball Recovery', 'Foul Won', 'Goal Keeper'
    ]

    chains = []
    for (match_id, poss_num), poss_events in events_df.groupby(['match_id', 'possession']):
        poss_events = poss_events.sort_values('index')
        poss_team = poss_events['possession_team'].iloc[0]

        team_actions = poss_events[
            (poss_events['team'] == poss_team) &
            (poss_events['type'].isin(action_types))
        ].copy()

        if len(team_actions) < 2:
            continue

        shots = poss_events[poss_events['type'] == 'Shot']
        if len(shots) > 0:
            last_shot = shots.iloc[-1]
            terminal_xg = last_shot.get('shot_statsbomb_xg', 0)
            terminal_xg = 0 if pd.isna(terminal_xg) else float(terminal_xg)
            ended_in_shot = True
            shot_outcome = last_shot.get('shot_outcome', 'Unknown')
        else:
            terminal_xg = 0.0
            ended_in_shot = False
            shot_outcome = None

        # tournament column if present
        tournament = poss_events['tournament'].iloc[0] if 'tournament' in poss_events.columns else 'unknown'

        chains.append({
            'match_id': match_id,
            'possession_num': poss_num,
            'team': poss_team,
            'tournament': tournament,
            'n_actions': len(team_actions),
            'terminal_xg': terminal_xg,
            'ended_in_shot': ended_in_shot,
            'shot_outcome': shot_outcome,
            'play_pattern': poss_events['play_pattern'].iloc[0],
            'start_minute': team_actions['minute'].iloc[0],
            'action_ids': team_actions['id'].tolist(),
            'action_types': team_actions['type'].tolist(),
            'players': team_actions['player'].tolist(),
            'player_ids': team_actions['player_id'].tolist(),
            'locations': team_actions['location'].tolist(),
            'timestamps': team_actions['timestamp'].tolist(),
            'durations': team_actions['duration'].tolist(),
        })

    chain_df = pd.DataFrame(chains)
    out_path = PROCESSED_DIR / "phase2_possession_chains.parquet"
    chain_df.to_parquet(out_path, index=False)
    n_shot = chain_df['ended_in_shot'].sum()
    print(f"  Saved {len(chain_df):,} chains ({n_shot:,} ending in shots) to {out_path}")
    return chain_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Baseline Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_baselines(events_df, passes_df):
    """
    Compute per-player baseline metrics: xa, progressive_passes, sca, matches.
    Filters to players with >= 50 passes.
    """
    print("\nComputing baseline metrics...")

    stats = []
    for player_id, pp in passes_df.groupby('player_id'):
        if pd.isna(player_id):
            continue
        if len(pp) < 50:
            continue

        player_name = pp['player'].iloc[0]
        team = pp['team'].mode().iloc[0]
        matches = pp['match_id'].nunique()
        n_passes = len(pp)
        completed = pp['is_completed'].sum()
        progressive = pp['is_progressive'].sum()

        # actual_assists: passes with pass_goal_assist flag
        # (named actual_assists not xa — pass_xa/xA column not available in this dataset)
        actual_assists = float(pp['pass_goal_assist'].fillna(False).astype(bool).sum())

        # SCA: passes with pass_shot_assist or pass_goal_assist
        sca = int(
            (pp['pass_shot_assist'].notna() | pp['pass_goal_assist'].notna()).sum()
        )

        stats.append({
            'player_id': player_id,
            'player': player_name,
            'team': team,
            'matches': matches,
            'total_passes': n_passes,
            'pass_completion_pct': completed / n_passes * 100 if n_passes > 0 else 0,
            'progressive_passes': int(progressive),
            'pressured_passes': int(pp['is_under_pressure'].sum()),
            'through_balls': int(pp['is_through_ball'].sum()),
            'switches': int(pp['is_switch'].sum()),
            'crosses': int(pp['is_cross'].sum()),
            'sca': sca,
            'actual_assists': actual_assists,
        })

    result = pd.DataFrame(stats)
    out_path = PROCESSED_DIR / "phase2_baselines.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved baselines for {len(result):,} players (>= 50 passes)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Chain Position Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_chain_positions(chains_df, events_df):
    """
    For shot-ending chains, compute where each player contributes.
    Aggregates pre_assist_ratio, buildup_ratio, total_xg_involved, chains_involved per player.
    No team filter (all tournaments).
    """
    print("\nComputing chain position profiles (all teams)...")

    shot_chains = chains_df[chains_df['ended_in_shot']].copy()
    print(f"  {len(shot_chains):,} shot-ending chains")

    # Build player_id lookup from events
    player_id_lookup = {}
    if events_df is not None and 'player_id' in events_df.columns:
        player_id_lookup = events_df.groupby('player')['player_id'].first().to_dict()

    profiles = {}
    for _, chain in shot_chains.iterrows():
        n = chain['n_actions']
        xg = chain['terminal_xg']
        chain_player_ids = chain.get('player_ids', [])

        for i, (player, action_type) in enumerate(zip(chain['players'], chain['action_types'])):
            if pd.isna(player):
                continue
            pos_from_end = n - 1 - i

            if player not in profiles:
                pid = chain_player_ids[i] if i < len(chain_player_ids) else player_id_lookup.get(player, None)
                profiles[player] = {
                    'player_id': pid,
                    'shot_pos': 0, 'assist_pos': 0,
                    'pre_assist_2': 0, 'pre_assist_3': 0,
                    'pre_assist_4_5': 0, 'buildup_6_plus': 0,
                    'total_chain_actions': 0,
                    'total_xg_involved': 0.0,
                    'chains_involved': set(),
                    'xg_at_shot_pos': 0.0, 'xg_at_assist_pos': 0.0,
                    'xg_at_pre_assist': 0.0, 'xg_at_buildup': 0.0,
                    'team': chain['team'],
                }

            p = profiles[player]
            p['total_chain_actions'] += 1
            p['total_xg_involved'] += xg
            p['chains_involved'].add((chain['match_id'], chain['possession_num']))

            if pos_from_end == 0:
                p['shot_pos'] += 1
                p['xg_at_shot_pos'] += xg
            elif pos_from_end == 1:
                p['assist_pos'] += 1
                p['xg_at_assist_pos'] += xg
            elif pos_from_end <= 3:
                p['pre_assist_2' if pos_from_end == 2 else 'pre_assist_3'] += 1
                p['xg_at_pre_assist'] += xg
            elif pos_from_end <= 5:
                p['pre_assist_4_5'] += 1
                p['xg_at_pre_assist'] += xg
            else:
                p['buildup_6_plus'] += 1
                p['xg_at_buildup'] += xg

    rows = []
    for player, p in profiles.items():
        n_c = len(p['chains_involved'])
        p['player'] = player
        p['chains_involved'] = n_c
        p['pre_assist_actions'] = p['pre_assist_2'] + p['pre_assist_3'] + p['pre_assist_4_5']
        p['final_actions'] = p['shot_pos'] + p['assist_pos']
        p['pre_assist_ratio'] = p['pre_assist_actions'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        p['buildup_ratio'] = p['buildup_6_plus'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        p['final_ratio'] = p['final_actions'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        rows.append(p)

    result = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "phase2_chain_positions.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved chain profiles for {len(result):,} players")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Decision Surplus
# ══════════════════════════════════════════════════════════════════════════════

def run_decision_surplus(passes_df, frame_lookup, model, feat_cols):
    """
    Wrapper around existing decision_surplus module.
    Runs on all tournament passes, saves to phase2 path.

    compute_decision_surplus() hardcodes a write to decision_surplus.parquet —
    we back up the Phase 1 file first, then move the wrongly-written file to
    the phase2 path after the call.
    """
    print("\n" + "=" * 70)
    print("COMPUTING DECISION SURPLUS")
    print("=" * 70)
    t_start = time.time()

    phase1_backup = PROCESSED_DIR / "phase1_decision_surplus_backup.parquet"
    wrong_path = PROCESSED_DIR / "decision_surplus.parquet"
    phase2_path = PROCESSED_DIR / "phase2_decision_surplus.parquet"

    # Backup Phase 1 file before it gets overwritten
    if wrong_path.exists() and not phase1_backup.exists():
        shutil.copy2(str(wrong_path), str(phase1_backup))
        print("  Backed up Phase 1 decision_surplus.parquet")

    from src.decision_surplus import compute_decision_surplus
    ds_df = compute_decision_surplus(passes_df, frame_lookup, model, feat_cols)

    # Move the wrongly-written file to the phase2 path
    if wrong_path.exists():
        shutil.move(str(wrong_path), str(phase2_path))
        print(f"  Saved to {phase2_path}")
    else:
        # Fallback: write directly if compute_decision_surplus didn't write to wrong_path
        ds_df.to_parquet(phase2_path, index=False)
        print(f"  Saved to {phase2_path}")

    elapsed = time.time() - t_start
    print(f"  DS complete: {len(ds_df):,} rows in {elapsed:.0f}s")
    return ds_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Defensive Topology Disruption
# ══════════════════════════════════════════════════════════════════════════════

def run_defensive_disruption(events_df, frame_lookup):
    """
    Compute Defensive Topology Disruption using an O(1) event index for lookups.

    Note: wrapping compute_dtd from defensive_topology.py causes O(n²) lookups
    due to events_df[events_df['id'] == prev_id] inside the loop.
    This optimized version pre-builds an event dict for O(1) access.
    """
    import networkx as nx

    print("\n" + "=" * 70)
    print("COMPUTING DEFENSIVE TOPOLOGY DISRUPTION (optimized)")
    print("=" * 70)
    t_start = time.time()

    # Build O(1) event index
    print("  Building event index for O(1) lookups...")
    event_index = {row['id']: row for _, row in events_df.iterrows()}
    print(f"  Event index built in {time.time()-t_start:.0f}s")

    def _build_graph(freeze_frame, threshold=15.0):
        defenders = []
        for p in freeze_frame:
            if not p['teammate'] and not p.get('keeper', False):
                defenders.append(p['location'])
        if len(defenders) < 3:
            return None, np.array(defenders) if defenders else np.empty((0, 2))
        positions = np.array(defenders)
        G = nx.Graph()
        for i in range(len(positions)):
            G.add_node(i, pos=positions[i])
        for i in range(len(positions)):
            diffs = positions[i + 1:] - positions[i]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            for j_offset, d in enumerate(dists):
                if d <= threshold:
                    G.add_edge(i, i + 1 + j_offset, weight=d)
        return G, positions

    def _metrics(G):
        if G is None or len(G.nodes) < 3:
            return None
        return {
            'n_edges': G.number_of_edges(),
            'n_nodes': G.number_of_nodes(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'n_components': nx.number_connected_components(G),
        }

    distance_threshold = 15.0
    action_types = ['Pass', 'Carry', 'Dribble', 'Shot']
    sorted_events = events_df.sort_values(['match_id', 'index'])
    relevant = sorted_events[sorted_events['type'].isin(action_types)]

    print(f"  Relevant events: {len(relevant):,}")
    results = []
    prev_id = None
    prev_match = None
    prev_metrics = None
    n_results = 0
    t_dtd = time.time()

    for _, event in relevant.iterrows():
        eid = event['id']

        if event['match_id'] != prev_match:
            prev_match = event['match_id']
            prev_id = eid
            prev_metrics = None
            continue

        if prev_id not in frame_lookup or eid not in frame_lookup:
            prev_id = eid
            prev_metrics = None
            continue

        ff_pre = frame_lookup[prev_id]['freeze_frame']
        ff_post = frame_lookup[eid]['freeze_frame']

        if len(ff_pre) < 10 or len(ff_post) < 10:
            prev_id = eid
            prev_metrics = None
            continue

        if prev_metrics is None:
            G_pre, _ = _build_graph(ff_pre, distance_threshold)
            m_pre = _metrics(G_pre)
        else:
            m_pre = prev_metrics

        G_post, _ = _build_graph(ff_post, distance_threshold)
        m_post = _metrics(G_post)

        if m_pre is None or m_post is None:
            prev_id = eid
            prev_metrics = m_post
            continue

        edge_change = (m_pre['n_edges'] - m_post['n_edges']) / max(m_pre['n_edges'], 1)
        cluster_change = m_pre['avg_clustering'] - m_post['avg_clustering']
        component_change = m_post['n_components'] - m_pre['n_components']

        # O(1) lookup using pre-built index
        prev_ev = event_index[prev_id]
        loc = prev_ev['location']
        start_x = float(loc[0]) if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2 else np.nan

        results.append({
            'event_id': prev_id,
            'player': prev_ev['player'],
            'player_id': prev_ev['player_id'],
            'team': prev_ev['team'],
            'match_id': prev_ev['match_id'],
            'action_type': prev_ev['type'],
            'minute': prev_ev['minute'],
            'start_x': start_x,
            'edges_pre': m_pre['n_edges'],
            'edges_post': m_post['n_edges'],
            'edge_change_pct': edge_change,
            'cluster_change': cluster_change,
            'component_change': component_change,
            'density_pre': m_pre['density'],
            'density_post': m_post['density'],
            'dtd_raw': 0.5 * edge_change + 0.3 * cluster_change + 0.2 * component_change,
        })

        prev_id = eid
        prev_metrics = m_post
        n_results += 1

        if n_results % 10000 == 0:
            elapsed_dtd = time.time() - t_dtd
            rate = n_results / elapsed_dtd
            print(f"    {n_results:,} results in {elapsed_dtd:.0f}s ({rate:.0f}/s)")

    dtd_df = pd.DataFrame(results)
    elapsed = time.time() - t_start
    out_path = PROCESSED_DIR / "phase2_defensive_disruption.parquet"
    dtd_df.to_parquet(out_path, index=False)
    print(f"  DTD done: {len(dtd_df):,} rows in {elapsed:.0f}s — saved to {out_path}")
    return dtd_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRV, CIR, TVI
# ══════════════════════════════════════════════════════════════════════════════

def compute_prv(passes_df):
    """
    PRV: mean positional value of completed pressured passes per player.
    Uses the same pos_value() as decision_surplus.py.
    """
    print("\nComputing Press Resistance Value (PRV)...")

    pressured = passes_df[passes_df['is_under_pressure']].copy()
    pressured['pass_value'] = pressured.apply(
        lambda p: pos_value(p['end_x'], p['end_y']) if p['is_completed'] else TURNOVER_PENALTY,
        axis=1,
    )

    prv = pressured.groupby(['player', 'player_id', 'team']).agg(
        n_pressured_passes=('pass_value', 'count'),
        total_prv=('pass_value', 'sum'),
        mean_prv=('pass_value', 'mean'),
        median_prv=('pass_value', 'median'),
        pressured_completion=('is_completed', 'mean'),
    ).reset_index()

    out_path = PROCESSED_DIR / "phase2_prv.parquet"
    prv.to_parquet(out_path, index=False)
    print(f"  Saved PRV for {len(prv):,} players to {out_path}")
    return prv


def compute_cir(chains_df, events_df=None):
    """
    CIR: fraction of each team's shot chains initiated by the player.
    Computed per-team so that Spain's 5/30 = 0.167, not 5/all_teams_total.
    """
    print("\nComputing Chain Initiation Rate (CIR)...")

    shot_chains = chains_df[chains_df['ended_in_shot']].copy()

    # Build player_id lookup from events if available
    player_id_lookup = {}
    if events_df is not None and 'player_id' in events_df.columns:
        player_id_lookup = events_df.groupby('player')['player_id'].first().to_dict()

    rows = []
    for team, team_chains in shot_chains.groupby('team'):
        total = len(team_chains)
        initiator_counts = {}
        initiator_player_ids = {}
        for _, chain in team_chains.iterrows():
            players = chain['players']
            player_ids = chain.get('player_ids', [])
            if len(players) > 0 and not pd.isna(players[0]):
                init = players[0]
                initiator_counts[init] = initiator_counts.get(init, 0) + 1
                if init not in initiator_player_ids:
                    pid = player_ids[0] if len(player_ids) > 0 else None
                    initiator_player_ids[init] = pid

        for player, cnt in initiator_counts.items():
            pid = initiator_player_ids.get(player) or player_id_lookup.get(player, None)
            rows.append({
                'player': player,
                'player_id': pid,
                'team': team,
                'chains_initiated': cnt,
                'team_shot_chains': total,
                'cir': cnt / total if total > 0 else 0,
            })

    cir_df = pd.DataFrame(rows).sort_values('chains_initiated', ascending=False)
    out_path = PROCESSED_DIR / "phase2_cir.parquet"
    cir_df.to_parquet(out_path, index=False)
    print(f"  Saved CIR for {len(cir_df):,} players (per-team) to {out_path}")
    return cir_df


def compute_tvi(events_df, min_passes=50):
    """
    TVI: CV(hold_duration) * mean(|direction_change_in_degrees|).

    Matches Phase 1 formula from novel_metrics.py:
      TVI = CV(hold_duration) * mean(|direction_change|)
    where direction_change is the absolute angle difference (in degrees)
    between consecutive passes for the same player, sorted by time.

    - hold_time CV: computed from pass durations grouped by player
    - mean_dir_change: mean of absolute direction changes between consecutive passes
    """
    print("\nComputing Tempo Variance Index (TVI)...")

    passes = events_df[events_df['type'] == 'Pass'].copy()

    # Extract start/end coords vectorised
    def _x(loc):
        try:
            return float(loc[0])
        except Exception:
            return np.nan

    def _y(loc):
        try:
            return float(loc[1])
        except Exception:
            return np.nan

    if 'start_x' not in passes.columns:
        passes['start_x'] = passes['location'].apply(_x)
        passes['start_y'] = passes['location'].apply(_y)
    if 'end_x' not in passes.columns:
        end_locs = passes['pass_end_location']
        passes['end_x'] = end_locs.apply(_x)
        passes['end_y'] = end_locs.apply(_y)

    # Compute outgoing pass angle (vectorised)
    if 'pass_angle' in passes.columns and passes['pass_angle'].notna().any():
        passes['_out_angle'] = passes['pass_angle']
    else:
        passes['_out_angle'] = np.arctan2(
            passes['end_y'] - passes['start_y'],
            passes['end_x'] - passes['start_x'],
        )

    results = []
    for player, grp in passes.groupby('player'):
        if len(grp) < min_passes:
            continue

        # Hold time: use duration column
        hold_times = grp['duration'].dropna().astype(float).values
        if len(hold_times) < 10:
            continue

        # Direction changes: absolute angle difference between consecutive passes (degrees)
        # Sort by match_id then index to get temporal order
        grp_sorted = grp.sort_values(['match_id', 'index'] if 'index' in grp.columns else ['match_id'])
        angles = grp_sorted['_out_angle'].dropna().values
        if len(angles) < 2:
            continue

        # Compute direction changes between consecutive passes
        angle_diffs = np.diff(angles)
        # Wrap to [-pi, pi] then take absolute value in degrees
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        direction_changes = np.abs(np.degrees(angle_diffs))

        if len(direction_changes) < 1:
            continue

        hold_arr = hold_times
        hold_cv = hold_arr.std() / hold_arr.mean() if hold_arr.mean() > 0 else 0

        # Mean direction change (Phase 1 formula)
        mean_dir_change = float(np.mean(direction_changes))

        # TVI = CV(hold_duration) * mean(|direction_change_in_degrees|)
        tvi = hold_cv * mean_dir_change

        team = grp['team'].mode().iloc[0]
        results.append({
            'player': player,
            'team': team,
            'n_passes': len(grp),
            'hold_time_mean': float(hold_arr.mean()),
            'hold_time_std': float(hold_arr.std()),
            'hold_time_cv': float(hold_cv),
            'mean_dir_change': mean_dir_change,
            'tvi': float(tvi),
        })

    result_df = pd.DataFrame(results).sort_values('tvi', ascending=False)
    out_path = PROCESSED_DIR / "phase2_tvi.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"  Saved TVI for {len(result_df):,} players to {out_path}")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Transformer PACV
# ══════════════════════════════════════════════════════════════════════════════

def compute_pacv(chains_df):
    """
    Compute PACV using the same approach as Phase 1's assemble_architect_score():
    PACV = total_xg_involved * pre_assist_ratio (from chain positions).

    Also runs Transformer to extract action values and saves to
    data/processed/phase2_action_values.parquet.

    extract_action_values() may hardcode a write to action_values.parquet —
    we back up the Phase 1 file first, then move the wrongly-written file to
    the phase2 path after the call.
    """
    print("\nComputing PACV (Transformer action values)...")

    import torch
    from src.transformer_model import (
        PossessionTransformer, PossessionDataset, extract_action_values,
    )

    # Load model
    model = PossessionTransformer(input_dim=17, d_model=64, nhead=4, num_layers=4, d_ff=128)
    model_path = MODELS_DIR / "possession_transformer.pt"
    model.load_state_dict(torch.load(str(model_path), weights_only=True))
    model.eval()
    print(f"  Loaded Transformer from {model_path}")

    # Build dataset
    dataset = PossessionDataset(chains_df)
    print(f"  Dataset: {len(dataset):,} chains")

    phase1_backup = PROCESSED_DIR / "phase1_action_values_backup.parquet"
    wrong_path = PROCESSED_DIR / "action_values.parquet"
    phase2_path = PROCESSED_DIR / "phase2_action_values.parquet"

    # Backup Phase 1 file before it gets overwritten
    if wrong_path.exists() and not phase1_backup.exists():
        shutil.copy2(str(wrong_path), str(phase1_backup))
        print("  Backed up Phase 1 action_values.parquet")

    # Extract action values (attention-weighted)
    values_df = extract_action_values(model, dataset, chains_df)

    # Move the wrongly-written file to the phase2 path
    if wrong_path.exists():
        shutil.move(str(wrong_path), str(phase2_path))
        print(f"  Saved to {phase2_path}")
    else:
        # Fallback: write directly if extract_action_values didn't write to wrong_path
        values_df.to_parquet(phase2_path, index=False)
        print(f"  Saved to {phase2_path}")

    print(f"  Saved {len(values_df):,} action values to {phase2_path}")

    return values_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2_pipeline(tournaments=('euro2024', 'euro2020'), skip_model=False):
    """
    Orchestrate the full Phase 2 pipeline.

    1.  Load all tournament events + frames
    2.  Enrich passes
    3.  Train pass difficulty model (unless skip_model=True and model exists)
    4.  Extract possession chains
    5.  Compute baselines
    6.  Compute chain positions
    7.  Compute Decision Surplus
    8.  Compute Defensive Topology Disruption
    9.  Compute PRV, CIR, TVI
    10. Compute PACV (Transformer action values)
    11. Print summary
    """
    pipeline_start = time.time()

    print("\n" + "=" * 70)
    print("PHASE 2 PIPELINE — THE ARCHITECT FRAMEWORK")
    print("=" * 70)
    print(f"Tournaments: {list(tournaments)}")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print("\n[1/10] Loading tournament data...")
    all_events = []
    combined_frames = {}

    for t in tournaments:
        ev, fl = load_tournament_data(t)
        ev = ev.copy()
        ev['tournament'] = t
        all_events.append(ev)
        combined_frames.update(fl)

    events_df = pd.concat(all_events, ignore_index=True)
    print(f"\n  Combined: {len(events_df):,} events, {len(combined_frames):,} freeze frames")
    print(f"  Tournaments: {events_df['tournament'].value_counts().to_dict()}")

    # ── Step 2: Enrich passes ──────────────────────────────────────────────────
    print("\n[2/10] Enriching passes...")
    passes_df = enrich_passes(events_df)

    # ── Step 3: Train pass difficulty model ────────────────────────────────────
    model_path = MODELS_DIR / "pass_difficulty_model.pkl"
    feat_path = MODELS_DIR / "pass_difficulty_features.pkl"

    if skip_model and model_path.exists() and feat_path.exists():
        print("\n[3/10] Loading existing pass difficulty model (skip_model=True)...")
        model = joblib.load(model_path)
        feat_cols = joblib.load(feat_path)
        print(f"  Loaded model from {model_path}")
    else:
        print("\n[3/10] Training pass difficulty model...")
        model, feat_cols = train_pass_difficulty_model(passes_df, combined_frames)

    # ── Step 4: Extract possession chains ─────────────────────────────────────
    print("\n[4/10] Extracting possession chains...")
    chains_df = extract_chains(events_df)

    # ── Step 5: Baseline metrics ───────────────────────────────────────────────
    print("\n[5/10] Computing baseline metrics...")
    baselines_df = compute_baselines(events_df, passes_df)

    # ── Step 6: Chain positions ────────────────────────────────────────────────
    print("\n[6/10] Computing chain positions...")
    chain_pos_df = compute_chain_positions(chains_df, events_df)

    # ── Step 7: Decision Surplus ───────────────────────────────────────────────
    print("\n[7/10] Computing Decision Surplus (this takes ~15-25 min)...")
    ds_df = run_decision_surplus(passes_df, combined_frames, model, feat_cols)

    # ── Step 8: Defensive Topology Disruption ─────────────────────────────────
    print("\n[8/10] Computing Defensive Topology Disruption...")
    dtd_df = run_defensive_disruption(events_df, combined_frames)

    # ── Step 9: PRV, CIR, TVI ─────────────────────────────────────────────────
    print("\n[9/10] Computing PRV, CIR, TVI...")
    prv_df = compute_prv(passes_df)
    cir_df = compute_cir(chains_df, events_df)
    tvi_df = compute_tvi(events_df)

    # ── Step 10: PACV ──────────────────────────────────────────────────────────
    print("\n[10/10] Computing PACV (Transformer action values)...")
    action_values_df = compute_pacv(chains_df)

    # ── Step 11: Summary ───────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("PHASE 2 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total wall time: {total_elapsed/60:.1f} min")

    # Per-tournament stats
    for t in tournaments:
        t_events = events_df[events_df['tournament'] == t]
        t_passes = passes_df[passes_df['tournament'] == t] if 'tournament' in passes_df.columns else passes_df
        print(f"\n  [{t.upper()}]")
        print(f"    Events:  {len(t_events):,}")
        print(f"    Passes:  {len(t_passes):,}")

    # File summary
    output_files = [
        "phase2_passes_enriched.parquet",
        "phase2_possession_chains.parquet",
        "phase2_decision_surplus.parquet",
        "phase2_defensive_disruption.parquet",
        "phase2_chain_positions.parquet",
        "phase2_baselines.parquet",
        "phase2_prv.parquet",
        "phase2_cir.parquet",
        "phase2_tvi.parquet",
        "phase2_action_values.parquet",
    ]
    print("\n  Output files:")
    for fname in output_files:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            try:
                df_tmp = pd.read_parquet(fpath)
                print(f"    [OK] {fname:<45} {len(df_tmp):>7,} rows  ({size_kb:,.0f} KB)")
            except Exception:
                print(f"    [OK] {fname:<45} ({size_kb:,.0f} KB)")
        else:
            print(f"    [MISSING] {fname}")

    print("\n  Models:")
    for mname in ["pass_difficulty_model.pkl", "pass_difficulty_features.pkl"]:
        mp = MODELS_DIR / mname
        status = "OK" if mp.exists() else "MISSING"
        print(f"    [{status}] {mname}")

    # Spot-check target players in DS
    target_players = ['Granit Xhaka', 'Toni Kroos', 'Rodri', 'Jorginho']
    print("\n  Target player DS spot-check:")
    for name in target_players:
        matches = ds_df[ds_df['player'].str.contains(name.split()[-1], na=False, case=False)]
        if len(matches) > 0:
            med = matches['decision_surplus'].median()
            mn = matches['decision_surplus'].mean()
            print(f"    {name:<30} n={len(matches):>4}  median DS={med:>7.4f}  mean DS={mn:>7.4f}")
        else:
            print(f"    {name:<30} NOT FOUND in DS output")

    return {
        'events_df': events_df,
        'passes_df': passes_df,
        'chains_df': chains_df,
        'baselines_df': baselines_df,
        'chain_pos_df': chain_pos_df,
        'ds_df': ds_df,
        'dtd_df': dtd_df,
        'prv_df': prv_df,
        'cir_df': cir_df,
        'tvi_df': tvi_df,
        'action_values_df': action_values_df,
        'model': model,
        'feat_cols': feat_cols,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Analytics Pipeline")
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip model training if pass_difficulty_model.pkl already exists"
    )
    parser.add_argument(
        "--tournaments", nargs="+", default=["euro2024", "euro2020"],
        help="Tournament names to process (default: euro2024 euro2020)"
    )
    args = parser.parse_args()

    results = run_phase2_pipeline(
        tournaments=tuple(args.tournaments),
        skip_model=args.skip_model,
    )
