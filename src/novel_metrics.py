"""
Novel Metrics Module — The Architect Framework
Computes the six sub-components of the Architect Score:
  1. Pre-Assist Chain Value (PACV) — from possession value model
  2. Decision Surplus (DS) — counterfactual pass evaluation
  3. Defensive Topology Disruption (DTD) — graph-based defensive analysis
  4. Press Resistance Value (PRV) — value of passes under pressure
  5. Chain Initiation Rate (CIR) — how often player starts shot chains
  6. Tempo Variance Index (TVI) — rhythm and directional variance
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from scipy.spatial import ConvexHull

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ============================================================
# PASS DIFFICULTY MODEL (prerequisite for Decision Surplus)
# ============================================================

def _get_defenders_near_line(passer_loc, target_loc, freeze_frame, threshold=3.0):
    """Count defenders within `threshold` meters of the pass line."""
    px, py = passer_loc
    tx, ty = target_loc
    line_vec = np.array([tx - px, ty - py])
    line_len = np.linalg.norm(line_vec)
    if line_len < 0.1:
        return 0

    line_unit = line_vec / line_len
    count = 0
    for player in freeze_frame:
        if player['teammate']:
            continue
        dx = player['location'][0] - px
        dy = player['location'][1] - py
        # Project onto line
        proj = dx * line_unit[0] + dy * line_unit[1]
        if proj < 0 or proj > line_len:
            continue
        # Perpendicular distance
        perp = abs(dx * line_unit[1] - dy * line_unit[0])
        if perp < threshold:
            count += 1
    return count


def _get_defenders_near_point(point, freeze_frame, threshold=5.0):
    """Count defenders within `threshold` meters of a point."""
    count = 0
    for player in freeze_frame:
        if player['teammate']:
            continue
        dist = np.sqrt(
            (player['location'][0] - point[0])**2 +
            (player['location'][1] - point[1])**2
        )
        if dist < threshold:
            count += 1
    return count


def build_pass_difficulty_model(passes_df, frame_lookup):
    """
    Train a model predicting P(pass completion | spatial context).
    Used for counterfactual decision surplus computation.
    """
    print("Building pass difficulty model...")

    features = []
    targets = []

    for _, p in passes_df.iterrows():
        if p['id'] not in frame_lookup:
            continue

        ff = frame_lookup[p['id']]['freeze_frame']
        if len(ff) < 10:  # too few players visible
            continue

        start = [p['start_x'], p['start_y']]
        end = [p['end_x'], p['end_y']]
        if np.isnan(start[0]) or np.isnan(end[0]):
            continue

        defenders_on_line = _get_defenders_near_line(start, end, ff, threshold=3.0)
        defenders_near_target = _get_defenders_near_point(end, ff, threshold=5.0)

        features.append({
            'start_x': start[0],
            'start_y': start[1],
            'end_x': end[0],
            'end_y': end[1],
            'pass_length': p['pass_length'],
            'pass_angle': p['pass_angle'],
            'dist_toward_goal': p['dist_toward_goal'],
            'defenders_on_line': defenders_on_line,
            'defenders_near_target': defenders_near_target,
            'is_under_pressure': int(p['is_under_pressure']),
            'is_ground_pass': int(p.get('pass_height') == 'Ground Pass'),
            'is_right_foot': int(p.get('pass_body_part') == 'Right Foot'),
            'n_visible_players': len(ff),
        })
        targets.append(int(p['is_completed']))

    X = pd.DataFrame(features)
    y = np.array(targets)
    print(f"  Training on {len(X)} passes ({y.mean()*100:.1f}% completed)")

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)

    # Quick accuracy check
    preds = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score, brier_score_loss
    print(f"  Training AUC: {roc_auc_score(y, preds):.3f}")
    print(f"  Brier score: {brier_score_loss(y, preds):.4f}")

    return model, X.columns.tolist()


# ============================================================
# DECISION SURPLUS
# ============================================================

def _generate_alternatives(passer_loc, freeze_frame, max_ground=35, max_loft=50):
    """Generate plausible alternative pass targets from freeze frame data."""
    alternatives = []
    for player in freeze_frame:
        if not player['teammate'] or player.get('actor', False):
            continue  # skip opponents and the passer
        if player.get('keeper', False):
            continue  # skip goalkeeper

        target = player['location']
        dist = np.sqrt(
            (target[0] - passer_loc[0])**2 +
            (target[1] - passer_loc[1])**2
        )

        if dist < 2:
            continue  # too close
        if dist > max_loft:
            continue  # too far for any pass

        alternatives.append({
            'target_loc': target,
            'distance': dist,
            'is_ground_feasible': dist <= max_ground,
        })

    return alternatives


def compute_decision_surplus(passes_df, frame_lookup, difficulty_model, feature_cols,
                             possession_value_fn=None):
    """
    For each pass with a freeze frame, generate alternatives,
    evaluate each, and compute the surplus of the chosen pass.
    
    If possession_value_fn is None, uses a simple positional value
    proxy (distance toward goal / 120).
    """
    print("Computing Decision Surplus...")

    if possession_value_fn is None:
        # Simple proxy: value = how close to goal (normalized 0-1)
        def possession_value_fn(x, y):
            return x / 120.0

    results = []
    for idx, p in passes_df.iterrows():
        if p['id'] not in frame_lookup:
            continue

        ff_data = frame_lookup[p['id']]
        ff = ff_data['freeze_frame']
        if len(ff) < 14:
            continue

        passer_loc = [p['start_x'], p['start_y']]
        actual_end = [p['end_x'], p['end_y']]
        if np.isnan(passer_loc[0]) or np.isnan(actual_end[0]):
            continue

        # Actual pass value
        actual_completed = p['is_completed']
        if actual_completed:
            actual_value = possession_value_fn(actual_end[0], actual_end[1])
        else:
            actual_value = -0.05  # turnover penalty

        # Generate and evaluate alternatives
        alternatives = _generate_alternatives(passer_loc, ff)
        if len(alternatives) < 2:
            continue

        alt_values = []
        for alt in alternatives:
            target = alt['target_loc']
            defenders_on_line = _get_defenders_near_line(passer_loc, target, ff, 3.0)
            defenders_near_target = _get_defenders_near_point(target, ff, 5.0)
            
            feat = pd.DataFrame([{
                'start_x': passer_loc[0],
                'start_y': passer_loc[1],
                'end_x': target[0],
                'end_y': target[1],
                'pass_length': alt['distance'],
                'pass_angle': np.arctan2(target[1] - passer_loc[1], target[0] - passer_loc[0]),
                'dist_toward_goal': target[0] - passer_loc[0],
                'defenders_on_line': defenders_on_line,
                'defenders_near_target': defenders_near_target,
                'is_under_pressure': int(p['is_under_pressure']),
                'is_ground_pass': 1 if alt['is_ground_feasible'] else 0,
                'is_right_foot': int(p.get('pass_body_part') == 'Right Foot'),
                'n_visible_players': len(ff),
            }])

            p_complete = difficulty_model.predict_proba(feat[feature_cols])[0][1]
            dest_value = possession_value_fn(target[0], target[1])
            expected_value = p_complete * dest_value + (1 - p_complete) * (-0.05)
            alt_values.append(expected_value)

        mean_alt = np.mean(alt_values)
        max_alt = np.max(alt_values)

        results.append({
            'event_id': p['id'],
            'player': p['player'],
            'player_id': p['player_id'],
            'match_id': p['match_id'],
            'team': p['team'],
            'minute': p['minute'],
            'start_x': passer_loc[0],
            'start_y': passer_loc[1],
            'end_x': actual_end[0],
            'end_y': actual_end[1],
            'actual_value': actual_value,
            'mean_alt_value': mean_alt,
            'max_alt_value': max_alt,
            'n_alternatives': len(alternatives),
            'decision_surplus': actual_value - mean_alt,
            'decision_surplus_vs_max': actual_value - max_alt,
            'is_completed': actual_completed,
            'is_under_pressure': p['is_under_pressure'],
        })

        if len(results) % 5000 == 0:
            print(f"  Processed {len(results)} passes...")

    result_df = pd.DataFrame(results)
    result_df.to_parquet(PROCESSED_DIR / "decision_surplus.parquet", index=False)
    print(f"  Computed DS for {len(result_df)} passes")

    return result_df


# ============================================================
# DEFENSIVE TOPOLOGY DISRUPTION
# ============================================================

def _build_defensive_graph(freeze_frame, distance_threshold=15.0):
    """
    Build a graph of the defending team.
    Nodes = defenders, edges = coverage relationships (within threshold distance).
    """
    defenders = [p for p in freeze_frame if not p['teammate'] and not p.get('keeper', False)]
    if len(defenders) < 3:
        return None, defenders

    G = nx.Graph()
    for i, d in enumerate(defenders):
        G.add_node(i, pos=d['location'])

    for i in range(len(defenders)):
        for j in range(i + 1, len(defenders)):
            dist = np.sqrt(
                (defenders[i]['location'][0] - defenders[j]['location'][0])**2 +
                (defenders[i]['location'][1] - defenders[j]['location'][1])**2
            )
            if dist <= distance_threshold:
                G.add_edge(i, j, weight=dist)

    return G, defenders


def _graph_metrics(G):
    """Extract topology metrics from a defensive graph."""
    if G is None or len(G.nodes) < 3:
        return None

    return {
        'n_edges': G.number_of_edges(),
        'n_nodes': G.number_of_nodes(),
        'edge_density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'n_components': nx.number_connected_components(G),
    }


def compute_defensive_disruption(events_df, frame_lookup, distance_threshold=15.0):
    """
    For consecutive events with freeze frames, measure how the defensive
    graph changes after each action. High disruption = the action broke
    defensive connectivity.
    """
    print(f"Computing Defensive Topology Disruption (threshold={distance_threshold}m)...")

    # Sort events by match and index to get consecutive pairs
    sorted_events = events_df.sort_values(['match_id', 'index'])

    # Only look at on-ball actions that could disrupt defense
    action_types = ['Pass', 'Carry', 'Dribble', 'Shot']
    relevant = sorted_events[sorted_events['type'].isin(action_types)].copy()

    results = []
    prev_event = None
    prev_match = None

    for _, event in relevant.iterrows():
        if event['match_id'] != prev_match:
            prev_event = event
            prev_match = event['match_id']
            continue

        # We need freeze frames for both the current and previous event
        if prev_event['id'] not in frame_lookup or event['id'] not in frame_lookup:
            prev_event = event
            continue

        ff_pre = frame_lookup[prev_event['id']]['freeze_frame']
        ff_post = frame_lookup[event['id']]['freeze_frame']

        if len(ff_pre) < 10 or len(ff_post) < 10:
            prev_event = event
            continue

        # Build defensive graphs
        G_pre, _ = _build_defensive_graph(ff_pre, distance_threshold)
        G_post, _ = _build_defensive_graph(ff_post, distance_threshold)

        metrics_pre = _graph_metrics(G_pre)
        metrics_post = _graph_metrics(G_post)

        if metrics_pre is None or metrics_post is None:
            prev_event = event
            continue

        # Compute disruption
        edge_change = (metrics_pre['n_edges'] - metrics_post['n_edges']) / max(metrics_pre['n_edges'], 1)
        cluster_change = metrics_pre['avg_clustering'] - metrics_post['avg_clustering']
        component_change = metrics_post['n_components'] - metrics_pre['n_components']

        results.append({
            'event_id': prev_event['id'],
            'player': prev_event['player'],
            'player_id': prev_event['player_id'],
            'team': prev_event['team'],
            'match_id': prev_event['match_id'],
            'action_type': prev_event['type'],
            'minute': prev_event['minute'],
            'start_x': float(prev_event['location'][0]) if isinstance(prev_event['location'], (list, np.ndarray)) else np.nan,
            'edges_pre': metrics_pre['n_edges'],
            'edges_post': metrics_post['n_edges'],
            'edge_change_pct': edge_change,
            'cluster_change': cluster_change,
            'component_change': component_change,
            'density_pre': metrics_pre['edge_density'],
            'density_post': metrics_post['edge_density'],
            'dtd_raw': 0.5 * edge_change + 0.3 * cluster_change + 0.2 * component_change,
        })

        prev_event = event

    result_df = pd.DataFrame(results)
    result_df.to_parquet(PROCESSED_DIR / "defensive_disruption.parquet", index=False)
    print(f"  Computed DTD for {len(result_df)} actions")

    return result_df


# ============================================================
# PRESS RESISTANCE VALUE (PRV)
# ============================================================

def compute_prv(passes_df, possession_value_fn=None):
    """
    Compute Press Resistance Value: the value of passes under pressure.
    """
    print("Computing Press Resistance Value...")

    if possession_value_fn is None:
        def possession_value_fn(x, y):
            return x / 120.0

    pressured = passes_df[passes_df['is_under_pressure']].copy()
    pressured['pass_value'] = pressured.apply(
        lambda p: possession_value_fn(p['end_x'], p['end_y']) if p['is_completed']
        else -0.05,
        axis=1
    )

    prv = pressured.groupby(['player', 'player_id', 'team']).agg(
        n_pressured_passes=('pass_value', 'count'),
        total_prv=('pass_value', 'sum'),
        mean_prv=('pass_value', 'mean'),
        median_prv=('pass_value', 'median'),
        pressured_completion=('is_completed', 'mean'),
    ).reset_index()

    prv.to_parquet(PROCESSED_DIR / "prv.parquet", index=False)
    print(f"  Computed PRV for {len(prv)} players")
    return prv


# ============================================================
# CHAIN INITIATION RATE (CIR)
# ============================================================

def compute_cir(chains_df, team='Bayer Leverkusen'):
    """
    Compute Chain Initiation Rate: how often a player's action begins
    a possession chain that ends in a shot.
    """
    print("Computing Chain Initiation Rate...")

    shot_chains = chains_df[(chains_df['team'] == team) & (chains_df['ended_in_shot'])]
    total_shot_chains = len(shot_chains)

    initiator_counts = {}
    for _, chain in shot_chains.iterrows():
        players = chain['players']
        if len(players) > 0 and not pd.isna(players[0]):
            initiator = players[0]
            initiator_counts[initiator] = initiator_counts.get(initiator, 0) + 1

    cir_df = pd.DataFrame([
        {'player': p, 'chains_initiated': c, 'cir': c / total_shot_chains}
        for p, c in initiator_counts.items()
    ]).sort_values('chains_initiated', ascending=False)

    cir_df.to_parquet(PROCESSED_DIR / "cir.parquet", index=False)
    print(f"  {total_shot_chains} shot chains, {len(cir_df)} unique initiators")
    return cir_df


# ============================================================
# TEMPO VARIANCE INDEX (TVI)
# ============================================================

def compute_tvi(events_df, min_passes=50):
    """
    Compute Tempo Variance Index: how much a player varies passing rhythm
    and direction.
    
    TVI = CV(hold_time) × mean(direction_change)
    """
    print("Computing Tempo Variance Index...")

    # Get all passes with timing info
    passes = events_df[events_df['type'] == 'Pass'].copy()
    receipts = events_df[events_df['type'] == 'Ball Receipt*'].copy()

    results = []
    for player, player_passes in passes.groupby('player'):
        if len(player_passes) < min_passes:
            continue

        hold_times = []
        direction_changes = []

        for _, p in player_passes.iterrows():
            # Hold time: duration of the pass event (includes time on ball before release)
            if pd.notna(p.get('duration')):
                hold_times.append(float(p['duration']))

            # Direction change: angle between incoming and outgoing pass
            if isinstance(p['location'], (list, np.ndarray)) and isinstance(p['pass_end_location'], (list, np.ndarray)):
                out_angle = np.arctan2(
                    p['pass_end_location'][1] - p['location'][1],
                    p['pass_end_location'][0] - p['location'][0]
                )
                # Find the previous event to this player (ball receipt)
                prev = events_df[
                    (events_df['match_id'] == p['match_id']) &
                    (events_df['index'] < p['index']) &
                    (events_df['player'] == player) &
                    (events_df['type'].isin(['Ball Receipt*', 'Carry']))
                ]
                if len(prev) > 0:
                    last_prev = prev.iloc[-1]
                    if isinstance(last_prev['location'], (list, np.ndarray)):
                        in_angle = np.arctan2(
                            p['location'][1] - last_prev['location'][1],
                            p['location'][0] - last_prev['location'][0]
                        )
                        angle_diff = abs(out_angle - in_angle)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        direction_changes.append(np.degrees(angle_diff))

        if len(hold_times) < 10 or len(direction_changes) < 10:
            continue

        hold_times = np.array(hold_times)
        hold_cv = hold_times.std() / hold_times.mean() if hold_times.mean() > 0 else 0
        mean_dir_change = np.mean(direction_changes)

        team = player_passes['team'].mode().iloc[0]
        results.append({
            'player': player,
            'team': team,
            'n_passes': len(player_passes),
            'hold_time_mean': hold_times.mean(),
            'hold_time_std': hold_times.std(),
            'hold_time_cv': hold_cv,
            'mean_direction_change': mean_dir_change,
            'tvi': hold_cv * mean_dir_change,
        })

    result_df = pd.DataFrame(results).sort_values('tvi', ascending=False)
    result_df.to_parquet(PROCESSED_DIR / "tvi.parquet", index=False)
    print(f"  Computed TVI for {len(result_df)} players")
    return result_df


# ============================================================
# ARCHITECT SCORE ASSEMBLY
# ============================================================

def assemble_architect_score(baselines, ds_df, dtd_df, prv_df, cir_df, tvi_df, chains_df):
    """
    Assemble the composite Architect Score from all sub-components.
    Two versions:
      - AS-E (Event-Based): PACV + PRV + CIR + TVI (no freeze frames needed)
      - AS-F (Full): AS-E + DS + DTD
    """
    print("Assembling Architect Score...")

    # Start with baseline player list (Leverkusen + opponents)
    players = baselines[['player', 'player_id', 'team', 'position', 'matches']].copy()

    # --- PACV (from chain positions) ---
    chain_pos = pd.read_parquet(PROCESSED_DIR / "chain_positions.parquet")
    players = players.merge(
        chain_pos[['player', 'pre_assist_ratio', 'buildup_ratio', 'total_xg_involved', 'chains_involved']],
        on='player', how='left'
    )
    # PACV proxy: xG involved weighted by pre-assist ratio
    players['pacv'] = players['total_xg_involved'].fillna(0) * players['pre_assist_ratio'].fillna(0)

    # --- PRV ---
    players = players.merge(prv_df[['player', 'mean_prv', 'n_pressured_passes']], on='player', how='left')
    players['prv'] = players['mean_prv'].fillna(0)

    # --- CIR ---
    players = players.merge(cir_df[['player', 'cir']], on='player', how='left')
    players['cir'] = players['cir'].fillna(0)

    # --- TVI ---
    players = players.merge(tvi_df[['player', 'tvi']], on='player', how='left')
    players['tvi'] = players['tvi'].fillna(0)

    # --- DS (median per player) ---
    if ds_df is not None and len(ds_df) > 0:
        ds_player = ds_df.groupby('player').agg(
            median_ds=('decision_surplus', 'median'),
            mean_ds=('decision_surplus', 'mean'),
            n_ds_passes=('decision_surplus', 'count'),
        ).reset_index()
        players = players.merge(ds_player, on='player', how='left')
        players['ds'] = players['median_ds'].fillna(0)
    else:
        players['ds'] = 0
        players['n_ds_passes'] = 0

    # --- DTD (mean per player) ---
    if dtd_df is not None and len(dtd_df) > 0:
        dtd_player = dtd_df.groupby('player').agg(
            mean_dtd=('dtd_raw', 'mean'),
            n_dtd_actions=('dtd_raw', 'count'),
        ).reset_index()
        players = players.merge(dtd_player, on='player', how='left')
        players['dtd'] = players['mean_dtd'].fillna(0)
    else:
        players['dtd'] = 0

    # --- Z-score normalization ---
    score_cols = ['pacv', 'prv', 'cir', 'tvi', 'ds', 'dtd']
    for col in score_cols:
        mean = players[col].mean()
        std = players[col].std()
        if std > 0:
            players[f'{col}_z'] = (players[col] - mean) / std
        else:
            players[f'{col}_z'] = 0

    # --- Composite scores ---
    event_cols = ['pacv_z', 'prv_z', 'cir_z', 'tvi_z']
    full_cols = event_cols + ['ds_z', 'dtd_z']

    players['architect_score_event'] = players[event_cols].mean(axis=1)
    players['architect_score_full'] = players[full_cols].mean(axis=1)

    players.to_parquet(PROCESSED_DIR / "architect_scores.parquet", index=False)
    print(f"  Assembled scores for {len(players)} players")

    return players
