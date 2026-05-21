"""
Data Ingestion Module — The Architect Framework
Pulls StatsBomb open data for Leverkusen 2023/24 (events + 360 freeze frames)
and builds the foundational datasets for all downstream analysis.
"""

import pandas as pd
import numpy as np
import requests
import json
import pickle
import os
from pathlib import Path
from statsbombpy import sb

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# StatsBomb competition/season IDs
LEVERKUSEN_COMP = 9    # Bundesliga
LEVERKUSEN_SEASON = 281  # 2023/24


def pull_leverkusen_events():
    """Pull all event data for Leverkusen's 34 Bundesliga matches."""
    print("Pulling Leverkusen 2023/24 match list...")
    matches = sb.matches(competition_id=LEVERKUSEN_COMP, season_id=LEVERKUSEN_SEASON)
    lev_matches = matches[
        (matches['home_team'] == 'Bayer Leverkusen') | 
        (matches['away_team'] == 'Bayer Leverkusen')
    ]
    print(f"  Found {len(lev_matches)} Leverkusen matches")

    all_events = []
    for i, (_, match) in enumerate(lev_matches.iterrows()):
        mid = match['match_id']
        events = sb.events(match_id=mid)
        events['match_id'] = mid
        events['home_team_name'] = match['home_team']
        events['away_team_name'] = match['away_team']
        events['home_score'] = match['home_score']
        events['away_score'] = match['away_score']
        events['match_date'] = match['match_date']
        all_events.append(events)
        if (i + 1) % 10 == 0:
            print(f"  Pulled {i + 1}/{len(lev_matches)} matches...")

    df = pd.concat(all_events, ignore_index=True)
    
    out_path = RAW_DIR / "leverkusen_2324_events.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} events to {out_path}")
    
    # Also save match metadata
    lev_matches.to_parquet(RAW_DIR / "leverkusen_2324_matches.parquet", index=False)
    
    return df


def pull_360_frames(events_df):
    """Pull 360 freeze frame data for all Leverkusen matches from GitHub."""
    match_ids = events_df['match_id'].unique()
    print(f"Pulling 360 data for {len(match_ids)} matches...")

    frame_lookup = {}
    total_frames = 0

    for i, mid in enumerate(match_ids):
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{int(mid)}.json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for frame in data:
                    frame_lookup[frame['event_uuid']] = {
                        'freeze_frame': frame['freeze_frame'],
                        'visible_area': frame['visible_area']
                    }
                total_frames += len(data)
        except Exception as e:
            print(f"  Error on match {mid}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(match_ids)} matches done ({total_frames} frames)")

    out_path = RAW_DIR / "frame_lookup.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(frame_lookup, f)
    print(f"  Saved {len(frame_lookup)} freeze frames to {out_path}")

    return frame_lookup


def enrich_passes(events_df):
    """Add derived columns to pass events: coordinates, progressive flag, etc."""
    passes = events_df[events_df['type'] == 'Pass'].copy()

    # Extract coordinates from numpy arrays
    passes['start_x'] = passes['location'].apply(
        lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else np.nan
    )
    passes['start_y'] = passes['location'].apply(
        lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else np.nan
    )
    passes['end_x'] = passes['pass_end_location'].apply(
        lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else np.nan
    )
    passes['end_y'] = passes['pass_end_location'].apply(
        lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else np.nan
    )

    # Derived features
    passes['dist_toward_goal'] = passes['end_x'] - passes['start_x']
    passes['is_progressive'] = passes['dist_toward_goal'] >= 10  # 10 yards forward
    passes['is_completed'] = passes['pass_outcome'].isna()  # NaN = completed in StatsBomb
    passes['lateral_dist'] = (passes['end_y'] - passes['start_y']).abs()
    passes['is_switch'] = passes['pass_switch'].fillna(False).astype(bool)
    passes['is_through_ball'] = passes['pass_through_ball'].fillna(False).astype(bool)
    passes['is_cross'] = passes['pass_cross'].fillna(False).astype(bool)
    passes['is_under_pressure'] = passes['under_pressure'].fillna(False).astype(bool)

    out_path = PROCESSED_DIR / "passes_enriched.parquet"
    passes.to_parquet(out_path, index=False)
    print(f"  Saved {len(passes)} enriched passes to {out_path}")

    return passes


def extract_possession_chains(events_df):
    """
    Extract possession chains from StatsBomb event data.
    Each chain is a sequence of on-ball actions by one team,
    tagged with the terminal outcome.
    """
    print("Extracting possession chains...")
    
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

        # Terminal outcome
        shots = poss_events[poss_events['type'] == 'Shot']
        if len(shots) > 0:
            last_shot = shots.iloc[-1]
            terminal_xg = last_shot.get('shot_statsbomb_xg', 0)
            terminal_xg = 0 if pd.isna(terminal_xg) else terminal_xg
            ended_in_shot = True
            shot_outcome = last_shot.get('shot_outcome', 'Unknown')
        else:
            terminal_xg = 0
            ended_in_shot = False
            shot_outcome = None

        chains.append({
            'match_id': match_id,
            'possession_num': poss_num,
            'team': poss_team,
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
    
    out_path = PROCESSED_DIR / "possession_chains.parquet"
    chain_df.to_parquet(out_path, index=False)
    print(f"  Saved {len(chain_df)} chains ({chain_df['ended_in_shot'].sum()} ending in shots)")

    return chain_df


def run_full_ingestion():
    """Run the complete data ingestion pipeline."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Pull events
    events_path = RAW_DIR / "leverkusen_2324_events.parquet"
    if events_path.exists():
        print("Loading cached events...")
        events = pd.read_parquet(events_path)
    else:
        events = pull_leverkusen_events()

    # Step 2: Pull 360 frames
    frames_path = RAW_DIR / "frame_lookup.pkl"
    if frames_path.exists():
        print("Loading cached freeze frames...")
        with open(frames_path, 'rb') as f:
            frame_lookup = pickle.load(f)
    else:
        frame_lookup = pull_360_frames(events)

    # Step 3: Enrich passes
    passes = enrich_passes(events)

    # Step 4: Extract possession chains
    chains = extract_possession_chains(events)

    # Print summary
    xhaka_events = events[events['player'].str.contains('Xhaka', na=False)]
    xhaka_passes = passes[passes['player'].str.contains('Xhaka', na=False)]
    xhaka_with_frames = xhaka_passes[xhaka_passes['id'].isin(frame_lookup.keys())]

    print(f"\n{'='*60}")
    print("DATA INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total events: {len(events)}")
    print(f"  Total passes: {len(passes)}")
    print(f"  Total freeze frames: {len(frame_lookup)}")
    print(f"  Total possession chains: {len(chains)}")
    print(f"  Chains ending in shots: {chains['ended_in_shot'].sum()}")
    print(f"  Xhaka events: {len(xhaka_events)}")
    print(f"  Xhaka passes: {len(xhaka_passes)}")
    print(f"  Xhaka passes with freeze frames: {len(xhaka_with_frames)} ({len(xhaka_with_frames)/len(xhaka_passes)*100:.1f}%)")

    return events, passes, chains, frame_lookup


if __name__ == "__main__":
    run_full_ingestion()
