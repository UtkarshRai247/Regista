"""
Defensive Topology Disruption (DTD)
Models the defending team's structure as a graph and measures
how each attacking action disrupts its connectivity.
"""

import pandas as pd
import numpy as np
import pickle
import time
import networkx as nx
from pathlib import Path


def build_defensive_graph(freeze_frame, threshold=15.0):
    """Build graph: nodes=defenders, edges=coverage within threshold."""
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

    # Vectorized distance computation
    for i in range(len(positions)):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        for j_offset, d in enumerate(dists):
            if d <= threshold:
                G.add_edge(i, i + 1 + j_offset, weight=d)

    return G, positions


def graph_metrics(G):
    """Extract topology metrics from a defensive graph."""
    if G is None or len(G.nodes) < 3:
        return None
    return {
        'n_edges': G.number_of_edges(),
        'n_nodes': G.number_of_nodes(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'n_components': nx.number_connected_components(G),
    }


def compute_dtd(events_df, frame_lookup, distance_threshold=15.0):
    """Compute DTD for all on-ball actions with consecutive freeze frames."""
    start = time.time()
    print(f"Computing DTD (threshold={distance_threshold}m)...")

    action_types = ['Pass', 'Carry', 'Dribble', 'Shot']
    sorted_events = events_df.sort_values(['match_id', 'index'])
    relevant = sorted_events[sorted_events['type'].isin(action_types)].copy()

    results = []
    prev_id = None
    prev_match = None
    prev_metrics = None

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

        # Use cached pre-metrics if available
        if prev_metrics is None:
            G_pre, _ = build_defensive_graph(ff_pre, distance_threshold)
            m_pre = graph_metrics(G_pre)
        else:
            m_pre = prev_metrics

        G_post, _ = build_defensive_graph(ff_post, distance_threshold)
        m_post = graph_metrics(G_post)

        if m_pre is None or m_post is None:
            prev_id = eid
            prev_metrics = m_post
            continue

        edge_change = (m_pre['n_edges'] - m_post['n_edges']) / max(m_pre['n_edges'], 1)
        cluster_change = m_pre['avg_clustering'] - m_post['avg_clustering']
        component_change = m_post['n_components'] - m_pre['n_components']

        prev_event = events_df[events_df['id'] == prev_id].iloc[0]
        loc = prev_event['location']
        start_x = float(loc[0]) if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2 else np.nan

        results.append({
            'event_id': prev_id,
            'player': prev_event['player'],
            'player_id': prev_event['player_id'],
            'team': prev_event['team'],
            'match_id': prev_event['match_id'],
            'action_type': prev_event['type'],
            'minute': prev_event['minute'],
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

    dtd_df = pd.DataFrame(results)
    dtd_df.to_parquet("data/processed/defensive_disruption.parquet", index=False)
    elapsed = time.time() - start
    print(f"  Done: {len(dtd_df)} actions in {elapsed:.0f}s")
    return dtd_df


def print_dtd_analysis(dtd_df, passes_df=None):
    """Print DTD analysis and comparisons."""
    print(f"\n{'='*70}")
    print("DEFENSIVE TOPOLOGY DISRUPTION RESULTS")
    print(f"{'='*70}")

    player_dtd = dtd_df.groupby(['player', 'team']).agg(
        n=('dtd_raw', 'count'),
        mean_dtd=('dtd_raw', 'mean'),
        median_dtd=('dtd_raw', 'median'),
        std_dtd=('dtd_raw', 'std'),
        pct_positive=('dtd_raw', lambda x: (x > 0).mean() * 100),
    ).reset_index()

    print("\nLeverkusen players (50+ actions) by mean DTD:")
    lev = player_dtd[(player_dtd['team'] == 'Bayer Leverkusen') & (player_dtd['n'] >= 50)]
    lev = lev.sort_values('mean_dtd', ascending=False)
    for _, r in lev.iterrows():
        m = " ◄" if 'Xhaka' in r['player'] else ""
        print(f"  {r['player']:<35} mean: {r['mean_dtd']:>8.5f}  "
              f"med: {r['median_dtd']:>8.5f}  %pos: {r['pct_positive']:>5.1f}%  n={r['n']:>4}{m}")

    # Trio comparison
    print(f"\n{'='*70}")
    print("XHAKA vs WIRTZ vs ANDRICH — Defensive Disruption")
    print(f"{'='*70}")
    for name in ['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich']:
        p = dtd_df[dtd_df['player'] == name]
        if len(p) == 0:
            continue
        passes_only = p[p['action_type'] == 'Pass']
        print(f"\n  {name}:")
        print(f"    Actions evaluated:  {len(p)}")
        print(f"    Mean DTD (all):     {p['dtd_raw'].mean():.5f}")
        print(f"    Mean DTD (passes):  {passes_only['dtd_raw'].mean():.5f}" if len(passes_only) > 0 else "")
        print(f"    % positive DTD:     {(p['dtd_raw'] > 0).mean()*100:.1f}%")

    # Correlation with progressive distance
    if passes_df is not None:
        dtd_passes = dtd_df[dtd_df['action_type'] == 'Pass'].copy()
        enriched = passes_df[['id', 'dist_toward_goal', 'is_progressive']].rename(columns={'id': 'event_id'})
        merged = dtd_passes.merge(enriched, on='event_id', how='inner')
        corr = merged['dtd_raw'].corr(merged['dist_toward_goal'])
        print(f"\n  Correlation(DTD, progressive distance): r = {corr:.3f} (target: < 0.3)")

    # Top 10 highest DTD actions
    print(f"\n  Top 10 highest-DTD individual actions:")
    top = dtd_df.nlargest(10, 'dtd_raw')
    for _, r in top.iterrows():
        print(f"    {r['player']:<25} {r['action_type']:<8} min {r['minute']:>3.0f}'  "
              f"DTD={r['dtd_raw']:.4f}  edges: {r['edges_pre']:.0f}→{r['edges_post']:.0f}")


if __name__ == "__main__":
    events = pd.read_parquet("data/raw/leverkusen_2324_events.parquet")
    passes = pd.read_parquet("data/processed/passes_enriched.parquet")
    with open("data/raw/frame_lookup.pkl", "rb") as f:
        frame_lookup = pickle.load(f)

    dtd_df = compute_dtd(events, frame_lookup, distance_threshold=15.0)
    print_dtd_analysis(dtd_df, passes)
