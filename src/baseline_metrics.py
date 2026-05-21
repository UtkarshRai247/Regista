"""
Baseline Metrics Module — The Architect Framework
Computes traditional per-player stats (xA, progressive passes, SCA, etc.)
and the chain position analysis that shows WHERE each player contributes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def compute_player_baselines(events_df, passes_df):
    """
    Compute traditional per-player stats for all players in the dataset.
    These serve as the benchmark against which Architect Score is compared.
    """
    print("Computing player baseline stats...")

    stats = []
    for player_id, player_events in events_df.groupby('player_id'):
        if pd.isna(player_id):
            continue

        player_name = player_events['player'].iloc[0]
        team = player_events['team'].mode().iloc[0]
        position = player_events['position'].mode().iloc[0] if len(player_events['position'].mode()) > 0 else 'Unknown'
        matches_played = player_events['match_id'].nunique()

        # Get this player's passes from the enriched pass data
        p_passes = passes_df[passes_df['player_id'] == player_id]
        n_passes = len(p_passes)
        if n_passes < 30:
            continue

        completed = p_passes['is_completed'].sum()

        # xA: sum of xG for shots this player assisted
        xa = 0
        assisted_shots = p_passes[p_passes['pass_assisted_shot_id'].notna()]
        for _, p in assisted_shots.iterrows():
            shot = events_df[
                (events_df['id'] == p['pass_assisted_shot_id']) &
                (events_df['type'] == 'Shot')
            ]
            if len(shot) > 0 and pd.notna(shot.iloc[0].get('shot_statsbomb_xg')):
                xa += shot.iloc[0]['shot_statsbomb_xg']

        stats.append({
            'player_id': player_id,
            'player': player_name,
            'team': team,
            'position': position,
            'matches': matches_played,
            'total_passes': n_passes,
            'pass_completion_pct': completed / n_passes * 100,
            'progressive_passes': p_passes['is_progressive'].sum(),
            'completed_progressive': (p_passes['is_progressive'] & p_passes['is_completed']).sum(),
            'pressured_passes': p_passes['is_under_pressure'].sum(),
            'pressured_completion_pct': (
                p_passes[p_passes['is_under_pressure']]['is_completed'].mean() * 100
                if p_passes['is_under_pressure'].sum() > 0 else 0
            ),
            'through_balls': p_passes['is_through_ball'].sum(),
            'switches': p_passes['is_switch'].sum(),
            'crosses': p_passes['is_cross'].sum(),
            'sca': len(p_passes[p_passes['pass_shot_assist'].notna() | p_passes['pass_goal_assist'].notna()]),
            'xa': xa,
            'shots': len(player_events[player_events['type'] == 'Shot']),
            'avg_pass_x': p_passes['start_x'].mean(),
            'avg_pass_y': p_passes['start_y'].mean(),
            'avg_pass_length': p_passes['pass_length'].mean(),
        })

    result = pd.DataFrame(stats)
    result.to_parquet(PROCESSED_DIR / "player_baselines.parquet", index=False)
    print(f"  Saved baselines for {len(result)} players")
    return result


def compute_chain_positions(chains_df, team='Bayer Leverkusen'):
    """
    For shot-ending chains, compute WHERE each player contributes.
    Position 0 = shot, 1 = assist position, 2+ = pre-assist and buildup.
    This is the foundation for the Pre-Assist Chain Value (PACV) metric.
    """
    print(f"Computing chain position profiles for {team}...")

    shot_chains = chains_df[(chains_df['team'] == team) & (chains_df['ended_in_shot'])]
    print(f"  {len(shot_chains)} shot-ending chains")

    profiles = {}
    for _, chain in shot_chains.iterrows():
        n = chain['n_actions']
        xg = chain['terminal_xg']

        for i, (player, action_type) in enumerate(zip(chain['players'], chain['action_types'])):
            if pd.isna(player):
                continue
            pos_from_end = n - 1 - i

            if player not in profiles:
                profiles[player] = {
                    'shot_pos': 0, 'assist_pos': 0,
                    'pre_assist_2': 0, 'pre_assist_3': 0,
                    'pre_assist_4_5': 0, 'buildup_6_plus': 0,
                    'total_chain_actions': 0,
                    'total_xg_involved': 0,
                    'chains_involved': set(),
                    'xg_at_shot_pos': 0, 'xg_at_assist_pos': 0,
                    'xg_at_pre_assist': 0, 'xg_at_buildup': 0,
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

    # Convert to DataFrame
    rows = []
    for player, p in profiles.items():
        p['player'] = player
        p['chains_involved'] = len(p['chains_involved'])
        p['pre_assist_actions'] = p['pre_assist_2'] + p['pre_assist_3'] + p['pre_assist_4_5']
        p['final_actions'] = p['shot_pos'] + p['assist_pos']
        p['pre_assist_ratio'] = p['pre_assist_actions'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        p['buildup_ratio'] = p['buildup_6_plus'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        p['final_ratio'] = p['final_actions'] / p['total_chain_actions'] if p['total_chain_actions'] > 0 else 0
        rows.append(p)

    result = pd.DataFrame(rows)
    result = result[result['chains_involved'] >= 10].sort_values('chains_involved', ascending=False)

    result.to_parquet(PROCESSED_DIR / "chain_positions.parquet", index=False)
    print(f"  Saved chain profiles for {len(result)} players")
    return result


def compute_percentile_rankings(baselines_df, target_player='Granit Xhaka'):
    """
    Compute percentile rankings for a target player against all midfielders.
    Returns a dict of metric -> (value, percentile).
    """
    midfield_positions = [
        'Center Midfield', 'Left Center Midfield', 'Right Center Midfield',
        'Center Defensive Midfield', 'Left Defensive Midfield',
        'Right Defensive Midfield', 'Center Attacking Midfield'
    ]
    midfielders = baselines_df[baselines_df['position'].isin(midfield_positions)]
    target = baselines_df[baselines_df['player'] == target_player]

    if len(target) == 0:
        print(f"  Player '{target_player}' not found")
        return {}

    target = target.iloc[0]
    metrics = ['total_passes', 'pass_completion_pct', 'progressive_passes',
               'pressured_passes', 'through_balls', 'switches', 'sca', 'xa']

    rankings = {}
    for m in metrics:
        val = target[m]
        pct = (midfielders[m] < val).mean() * 100
        rankings[m] = (val, pct)

    return rankings


def print_comparison(chain_profiles, players=None):
    """Print a formatted comparison of chain position profiles."""
    if players is None:
        players = ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']

    print(f"\n{'='*70}")
    print("CHAIN POSITION PROFILES — Where do players contribute in shot chains?")
    print(f"{'='*70}")

    for name in players:
        row = chain_profiles[chain_profiles['player'] == name]
        if len(row) == 0:
            print(f"\n  {name}: NOT FOUND")
            continue
        row = row.iloc[0]

        print(f"\n  {name}:")
        print(f"    Chains involved:    {row['chains_involved']}")
        print(f"    Shot position:      {row['shot_pos']:>3}  ({row['xg_at_shot_pos']:.2f} xG)")
        print(f"    Assist position:    {row['assist_pos']:>3}  ({row['xg_at_assist_pos']:.2f} xG)")
        print(f"    Pre-assist (2-5):   {row['pre_assist_actions']:>3}  ({row['xg_at_pre_assist']:.2f} xG)")
        print(f"    Deep buildup (6+):  {row['buildup_6_plus']:>3}  ({row['xg_at_buildup']:.2f} xG)")
        print(f"    Final ratio:        {row['final_ratio']:.1%}")
        print(f"    Pre-assist ratio:   {row['pre_assist_ratio']:.1%}")
        print(f"    Buildup ratio:      {row['buildup_ratio']:.1%}")
        print(f"    Total xG involved:  {row['total_xg_involved']:.2f}")


if __name__ == "__main__":
    events = pd.read_parquet(PROCESSED_DIR.parent / "raw" / "leverkusen_2324_events.parquet")
    passes = pd.read_parquet(PROCESSED_DIR / "passes_enriched.parquet")
    chains = pd.read_parquet(PROCESSED_DIR / "possession_chains.parquet")

    baselines = compute_player_baselines(events, passes)
    chain_profiles = compute_chain_positions(chains)

    rankings = compute_percentile_rankings(baselines)
    print(f"\nXhaka Percentile Rankings (among midfielders):")
    for metric, (val, pct) in rankings.items():
        print(f"  {metric}: {val:.1f} ({pct:.0f}th percentile)")

    print_comparison(chain_profiles)
