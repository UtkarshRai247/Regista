"""
Decision Surplus — Optimized Computation
For each pass with a freeze frame, generates alternative pass targets,
evaluates each through the pass difficulty model, and computes the
surplus of the chosen pass over the average alternative.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import time
from pathlib import Path


def pos_value(x, y):
    """Positional value proxy. Will be replaced by Transformer learned values."""
    x_val = (x / 120.0) ** 1.5
    y_center = abs(y - 40) / 40
    y_penalty = 1 - 0.3 * y_center
    return x_val * y_penalty


TURNOVER_PENALTY = -0.05


def preprocess_freeze_frames(frame_lookup, pass_ids):
    """Convert freeze frames from list-of-dicts to numpy arrays for speed."""
    processed = {}
    for pid in pass_ids:
        if pid not in frame_lookup:
            continue
        ff = frame_lookup[pid]['freeze_frame']
        if len(ff) < 10:
            continue

        teammates = []
        opponents = []
        for p in ff:
            loc = p['location']
            if p['teammate']:
                if not p.get('actor', False) and not p.get('keeper', False):
                    teammates.append(loc)
            else:
                if not p.get('keeper', False):
                    opponents.append(loc)

        if len(teammates) < 2:
            continue

        processed[pid] = {
            'teammates': np.array(teammates),
            'opponents': np.array(opponents) if opponents else np.empty((0, 2)),
            'n_total': len(ff),
            'n_teammates': len(teammates),
            'n_opponents': len(opponents),
        }
    return processed


def compute_spatial_features_batch(passer_loc, targets, opponents, feat_cols):
    """Compute features for multiple alternative passes at once."""
    n = len(targets)
    sx, sy = passer_loc

    dists = np.sqrt((targets[:, 0] - sx)**2 + (targets[:, 1] - sy)**2)
    angles = np.arctan2(targets[:, 1] - sy, targets[:, 0] - sx)
    dtg = targets[:, 0] - sx
    lat = np.abs(targets[:, 1] - sy)

    # Defenders near pass line and target for each alternative
    def_on_line = np.zeros(n)
    def_near_target = np.zeros(n)

    if len(opponents) > 0:
        for i in range(n):
            tx, ty = targets[i]
            line_vec = np.array([tx - sx, ty - sy])
            line_len = np.linalg.norm(line_vec)
            if line_len < 0.5:
                continue
            line_unit = line_vec / line_len

            # Vectorized defender-on-line check
            dx = opponents[:, 0] - sx
            dy = opponents[:, 1] - sy
            proj = dx * line_unit[0] + dy * line_unit[1]
            perp = np.abs(dx * line_unit[1] - dy * line_unit[0])
            mask = (proj >= 0) & (proj <= line_len) & (perp < 3.0)
            def_on_line[i] = mask.sum()

            # Defenders near target
            tdist = np.sqrt((opponents[:, 0] - tx)**2 + (opponents[:, 1] - ty)**2)
            def_near_target[i] = (tdist < 5.0).sum()

    # Closest defender to passer
    if len(opponents) > 0:
        passer_dists = np.sqrt((opponents[:, 0] - sx)**2 + (opponents[:, 1] - sy)**2)
        closest = passer_dists.min()
    else:
        closest = 999.0

    return {
        'pass_length': dists,
        'pass_angle': angles,
        'dist_toward_goal': dtg,
        'lateral_dist': lat,
        'defenders_on_line': def_on_line,
        'defenders_near_target': def_near_target,
        'closest_defender': np.full(n, closest),
    }


def compute_decision_surplus(passes_df, frame_lookup, model, feat_cols,
                              batch_size=500, max_passes=None):
    """Compute Decision Surplus for all passes with freeze frames."""
    start = time.time()

    pass_ids = passes_df['id'].values
    ff_data = preprocess_freeze_frames(frame_lookup, pass_ids)
    print(f"  Preprocessed {len(ff_data)} freeze frames in {time.time()-start:.0f}s")

    valid_passes = passes_df[passes_df['id'].isin(ff_data.keys())].copy()
    if max_passes:
        valid_passes = valid_passes.head(max_passes)
    print(f"  Processing {len(valid_passes)} passes...")

    results = []
    t0 = time.time()

    for idx, (_, p) in enumerate(valid_passes.iterrows()):
        ff = ff_data[p['id']]
        sx, sy = p['start_x'], p['start_y']
        ex, ey = p['end_x'], p['end_y']

        if np.isnan(sx) or np.isnan(ex):
            continue

        # Actual pass value
        actual_value = pos_value(ex, ey) if p['is_completed'] else TURNOVER_PENALTY

        # Filter teammates as potential targets
        teammates = ff['teammates']
        dists = np.sqrt((teammates[:, 0] - sx)**2 + (teammates[:, 1] - sy)**2)
        mask = (dists >= 2) & (dists <= 50)
        targets = teammates[mask]

        if len(targets) < 2:
            continue

        # Batch compute features for all alternatives
        spatial = compute_spatial_features_batch(
            [sx, sy], targets, ff['opponents'], feat_cols
        )

        # Build feature DataFrame for batch prediction
        feat_df = pd.DataFrame({
            'start_x': sx,
            'start_y': sy,
            'end_x': targets[:, 0],
            'end_y': targets[:, 1],
            'pass_length': spatial['pass_length'],
            'pass_angle': spatial['pass_angle'],
            'dist_toward_goal': spatial['dist_toward_goal'],
            'lateral_dist': spatial['lateral_dist'],
            'defenders_on_line': spatial['defenders_on_line'],
            'defenders_near_target': spatial['defenders_near_target'],
            'closest_defender': spatial['closest_defender'],
            'is_under_pressure': int(p['is_under_pressure']),
            'is_ground_pass': (spatial['pass_length'] <= 35).astype(int),
            'n_visible': ff['n_total'],
            'n_teammates': ff['n_teammates'],
            'n_opponents': ff['n_opponents'],
        })

        # Batch prediction
        p_complete = model.predict_proba(feat_df[feat_cols])[:, 1]
        dest_values = np.array([pos_value(t[0], t[1]) for t in targets])
        alt_expected = p_complete * dest_values + (1 - p_complete) * TURNOVER_PENALTY

        mean_alt = alt_expected.mean()
        max_alt = alt_expected.max()

        results.append({
            'event_id': p['id'],
            'player': p['player'],
            'player_id': p['player_id'],
            'match_id': p['match_id'],
            'team': p['team'],
            'minute': p['minute'],
            'start_x': sx, 'start_y': sy,
            'end_x': ex, 'end_y': ey,
            'actual_value': actual_value,
            'mean_alt_value': mean_alt,
            'max_alt_value': max_alt,
            'n_alternatives': len(targets),
            'decision_surplus': actual_value - mean_alt,
            'ds_vs_max': actual_value - max_alt,
            'is_completed': p['is_completed'],
            'is_under_pressure': p['is_under_pressure'],
            'is_progressive': p['is_progressive'],
        })

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    {idx+1}/{len(valid_passes)} ({rate:.0f} passes/s)")

    ds_df = pd.DataFrame(results)
    ds_df.to_parquet("data/processed/decision_surplus.parquet", index=False)

    elapsed = time.time() - start
    print(f"  Done: {len(ds_df)} passes in {elapsed:.0f}s ({len(ds_df)/elapsed:.0f}/s)")

    return ds_df


def print_ds_analysis(ds_df):
    """Print full Decision Surplus analysis."""
    print(f"\n{'='*70}")
    print("DECISION SURPLUS RESULTS")
    print(f"{'='*70}")

    player_ds = ds_df.groupby(['player', 'team']).agg(
        n=('decision_surplus', 'count'),
        median=('decision_surplus', 'median'),
        mean=('decision_surplus', 'mean'),
        std=('decision_surplus', 'std'),
        pct_pos=('decision_surplus', lambda x: (x > 0).mean() * 100),
    ).reset_index()

    print("\nLeverkusen players (100+ passes) by median DS:")
    lev = player_ds[(player_ds['team'] == 'Bayer Leverkusen') & (player_ds['n'] >= 100)]
    lev = lev.sort_values('median', ascending=False)
    for _, r in lev.iterrows():
        m = " ◄" if 'Xhaka' in r['player'] else ""
        print(f"  {r['player']:<35} med: {r['median']:>7.4f}  "
              f"mean: {r['mean']:>7.4f}  %pos: {r['pct_pos']:>5.1f}%  n={r['n']:>4}{m}")

    print(f"\n{'='*70}")
    print("XHAKA vs WIRTZ vs ANDRICH — Decision Surplus")
    print(f"{'='*70}")
    for name in ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']:
        p = ds_df[ds_df['player'] == name]
        if len(p) == 0:
            continue
        pres = p[p['is_under_pressure']]
        prog = p[p['is_progressive']]
        print(f"\n  {name}:")
        print(f"    Passes evaluated:   {len(p)}")
        print(f"    Median DS:          {p['decision_surplus'].median():.4f}")
        print(f"    Mean DS:            {p['decision_surplus'].mean():.4f}")
        print(f"    % positive DS:      {(p['decision_surplus'] > 0).mean()*100:.1f}%")
        print(f"    DS under pressure:  {pres['decision_surplus'].median():.4f} (n={len(pres)})")
        print(f"    DS on progressive:  {prog['decision_surplus'].median():.4f} (n={len(prog)})")

    # Split-half reliability
    print(f"\n{'='*70}")
    print("SPLIT-HALF RELIABILITY — Xhaka DS")
    print(f"{'='*70}")
    xhaka = ds_df[ds_df['player'].str.contains('Xhaka')]
    matches = sorted(xhaka['match_id'].unique())
    per_match = xhaka.groupby('match_id')['decision_surplus'].median()
    odd = per_match.iloc[::2]
    even = per_match.iloc[1::2]
    print(f"  Odd matches median DS:  {odd.mean():.4f} (n={len(odd)})")
    print(f"  Even matches median DS: {even.mean():.4f} (n={len(even)})")
    print(f"  Match-level DS std:     {per_match.std():.4f}")
    print(f"  Match-level DS range:   [{per_match.min():.4f}, {per_match.max():.4f}]")

    # Correlation with progressive passing
    player_ds2 = ds_df.groupby('player').agg(
        median_ds=('decision_surplus', 'median'),
        prog_pct=('is_progressive', 'mean'),
    ).reset_index()
    corr = player_ds2['median_ds'].corr(player_ds2['prog_pct'])
    print(f"\n  Correlation(DS, progressive%): r = {corr:.3f} (target: < 0.4)")


if __name__ == "__main__":
    passes = pd.read_parquet("data/processed/passes_enriched.parquet")
    with open("data/raw/frame_lookup.pkl", "rb") as f:
        frame_lookup = pickle.load(f)
    model = joblib.load("models/pass_difficulty_model.pkl")
    feat_cols = joblib.load("models/pass_difficulty_features.pkl")

    ds_df = compute_decision_surplus(passes, frame_lookup, model, feat_cols)
    print_ds_analysis(ds_df)
