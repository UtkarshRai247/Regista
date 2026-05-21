"""
Phase 2 Ingestion Module — The Architect Framework
Pulls StatsBomb open data for Euro 2024, Euro 2020, and Leverkusen 2023/24 360 frames.
Identifies target players and logs data coverage.
"""

import pandas as pd
import numpy as np
import requests
import pickle
import os
from pathlib import Path
from statsbombpy import sb

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "phase2"

# Competition/Season IDs
EURO_2024_COMP = 55
EURO_2024_SEASON = 282

EURO_2020_COMP = 55
EURO_2020_SEASON = 43

LEVERKUSEN_COMP = 9
LEVERKUSEN_SEASON = 281

# Target player substrings
# Note: StatsBomb stores full legal names; short nicknames don't match directly.
#   Pedri  → "Pedro González López"    (search: 'González López')
#   Vitinha → "Vitor Machado Ferreira" (search: 'Vitor Machado')
#   Jorginho → "Jorge Luiz Frello Filho" (search: 'Frello')
EURO_2024_PLAYERS = [
    'González López',   # Pedri
    'Hernández Cascante',  # Rodri
    'Zubimendi',
    'Vitor Machado',    # Vitinha
    'Kroos',
    'Xhaka',
    'Fabián Ruiz',      # Fabian Ruiz Peña (not Fabian Schär)
    'Kanté',
    'Bellingham',
]

EURO_2020_PLAYERS = [
    'González López',   # Pedri
    'Frenkie',          # Frenkie de Jong
    'Frello',           # Jorginho (Jorge Luiz Frello Filho)
    'Verratti',
    'Busquets',
    'Phillips',
]


def pull_tournament_events(competition_id: int, season_id: int, name: str) -> pd.DataFrame:
    """
    Pull all match events for a competition/season.

    Adds match metadata columns and saves to data/raw/{name}_events.parquet.
    Also saves match list to data/raw/{name}_matches.parquet.

    Returns the full events DataFrame.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_events = RAW_DIR / f"{name}_events.parquet"
    out_matches = RAW_DIR / f"{name}_matches.parquet"

    if out_events.exists():
        print(f"[{name}] Events file already exists — loading from cache.")
        return pd.read_parquet(out_events)

    print(f"[{name}] Pulling match list (competition_id={competition_id}, season_id={season_id})...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"[{name}]   Found {len(matches)} matches.")

    matches.to_parquet(out_matches, index=False)
    print(f"[{name}]   Saved match list to {out_matches}")

    all_events = []
    for i, (_, match) in enumerate(matches.iterrows()):
        mid = match['match_id']
        try:
            events = sb.events(match_id=mid)
        except Exception as e:
            print(f"[{name}]   ERROR pulling match {mid}: {e}")
            continue

        events['match_id'] = mid
        events['home_team_name'] = match['home_team']
        events['away_team_name'] = match['away_team']
        events['home_score'] = match['home_score']
        events['away_score'] = match['away_score']
        events['match_date'] = match['match_date']
        all_events.append(events)

        if (i + 1) % 10 == 0:
            print(f"[{name}]   {i + 1}/{len(matches)} matches pulled...")

    if not all_events:
        raise RuntimeError(f"[{name}] No events retrieved for any match. Check API connectivity.")
    df = pd.concat(all_events, ignore_index=True)
    df.to_parquet(out_events, index=False)
    print(f"[{name}]   Saved {len(df):,} events to {out_events}")

    return df


def pull_360_frames(match_ids, name: str, output_path=None) -> dict:
    """
    Pull 360 freeze frame data for given match IDs from StatsBomb open-data GitHub.

    Returns frame_lookup dict: {event_uuid: {'freeze_frame': [...], 'visible_area': [...]}}.
    Skips matches that return HTTP 404 (no 360 data available).
    Saves to data/raw/{name}_frames.pkl.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_path if output_path is not None else (RAW_DIR / f"{name}_frames.pkl")

    if out_path.exists():
        print(f"[{name}] Frames file already exists — loading from cache.")
        with open(out_path, 'rb') as f:
            return pickle.load(f)

    print(f"[{name}] Pulling 360 data for {len(match_ids)} matches...")

    frame_lookup = {}
    total_frames = 0
    skipped = 0

    for i, mid in enumerate(match_ids):
        url = (
            f"https://raw.githubusercontent.com/statsbomb/open-data/master"
            f"/data/three-sixty/{int(mid)}.json"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for frame in data:
                    frame_lookup[frame['event_uuid']] = {
                        'freeze_frame': frame['freeze_frame'],
                        'visible_area': frame['visible_area'],
                    }
                total_frames += len(data)
            elif resp.status_code == 404:
                skipped += 1
            else:
                print(f"[{name}]   Unexpected HTTP {resp.status_code} for match {mid}")
        except requests.exceptions.Timeout:
            # Retry once with a longer timeout
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    for frame in data:
                        frame_lookup[frame['event_uuid']] = {
                            'freeze_frame': frame['freeze_frame'],
                            'visible_area': frame['visible_area'],
                        }
                    total_frames += len(data)
            except Exception as e2:
                print(f"[{name}]   Retry failed for match {mid}: {e2}")
        except Exception as e:
            print(f"[{name}]   Error on match {mid}: {e}")

        if (i + 1) % 10 == 0:
            print(
                f"[{name}]   {i + 1}/{len(match_ids)} done "
                f"({total_frames:,} frames, {skipped} without 360 data)"
            )

    with open(out_path, 'wb') as f:
        pickle.dump(frame_lookup, f)

    print(
        f"[{name}]   Saved {len(frame_lookup):,} freeze frames to {out_path} "
        f"({skipped} matches had no 360 data)"
    )
    return frame_lookup


def identify_target_players(
    events_df: pd.DataFrame,
    frame_lookup: dict,
    player_substrings: list,
    tournament_name: str = "",
) -> pd.DataFrame:
    """
    For each player substring, find coverage stats in events_df.

    Returns a DataFrame with columns:
        search_term, full_name, team, matches_played,
        total_passes, passes_with_ff, sufficient_data
    """
    passes = events_df[events_df['type'] == 'Pass'].copy()

    rows = []
    for substr in player_substrings:
        mask = passes['player'].str.contains(substr, na=False, case=False)
        player_passes = passes[mask]

        if len(player_passes) == 0:
            # Try alternative encoding (e.g. accented characters)
            mask_alt = events_df['player'].str.contains(substr, na=False, case=False)
            if mask_alt.sum() == 0:
                print(f"  WARNING: '{substr}' not found in {tournament_name}")
                rows.append({
                    'search_term': substr,
                    'full_name': None,
                    'team': None,
                    'tournament': tournament_name,
                    'matches_played': 0,
                    'total_passes': 0,
                    'passes_with_ff': 0,
                    'sufficient_data': False,
                })
                continue
            # Player found in events but not in passes (unlikely but safe)
            full_name = events_df.loc[mask_alt, 'player'].value_counts().index[0]
            team = events_df.loc[mask_alt, 'team'].value_counts().index[0]
            rows.append({
                'search_term': substr,
                'full_name': full_name,
                'team': team,
                'tournament': tournament_name,
                'matches_played': 0,
                'total_passes': 0,
                'passes_with_ff': 0,
                'sufficient_data': False,
            })
            continue

        # Most common name variant (handles accents/abbreviations)
        full_name = player_passes['player'].value_counts().index[0]
        team = player_passes['team'].value_counts().index[0]
        matches_played = player_passes['match_id'].nunique()
        total_passes = len(player_passes)
        passes_with_ff = player_passes['id'].isin(frame_lookup.keys()).sum()
        sufficient = (passes_with_ff >= 50) or (total_passes >= 100)

        rows.append({
            'search_term': substr,
            'full_name': full_name,
            'team': team,
            'tournament': tournament_name,
            'matches_played': matches_played,
            'total_passes': total_passes,
            'passes_with_ff': int(passes_with_ff),
            'sufficient_data': sufficient,
        })

    return pd.DataFrame(rows)


def run_tournament_ingestion():
    """
    Orchestrate Phase 2 data ingestion:
      1. Euro 2024 events + 360 frames
      2. Euro 2020 events + 360 frames
      3. Leverkusen 2023/24 360 frames (events already exist)
      4. Player coverage report for both tournaments
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Euro 2024 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Euro 2024 events")
    print("=" * 60)
    euro2024_events = pull_tournament_events(EURO_2024_COMP, EURO_2024_SEASON, "euro2024")

    print("\n" + "=" * 60)
    print("STEP 2: Euro 2024 360 frames")
    print("=" * 60)
    euro2024_match_ids = euro2024_events['match_id'].unique()
    euro2024_frames = pull_360_frames(euro2024_match_ids, "euro2024")

    # ── Euro 2020 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Euro 2020 events")
    print("=" * 60)
    euro2020_events = pull_tournament_events(EURO_2020_COMP, EURO_2020_SEASON, "euro2020")

    print("\n" + "=" * 60)
    print("STEP 4: Euro 2020 360 frames")
    print("=" * 60)
    euro2020_match_ids = euro2020_events['match_id'].unique()
    euro2020_frames = pull_360_frames(euro2020_match_ids, "euro2020")

    # ── Leverkusen 2023/24 360 frames ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Leverkusen 2023/24 360 frames (events already cached)")
    print("=" * 60)
    lev_frames_path = RAW_DIR / "frame_lookup.pkl"
    if lev_frames_path.exists():
        print("  Loading cached Leverkusen frames...")
        with open(lev_frames_path, 'rb') as f:
            lev_frames = pickle.load(f)
    else:
        lev_events = pd.read_parquet(RAW_DIR / "leverkusen_2324_events.parquet")
        lev_match_ids = lev_events['match_id'].unique()
        print(f"  Pulling 360 frames for {len(lev_match_ids)} Leverkusen matches...")
        lev_frames = pull_360_frames(lev_match_ids, "leverkusen",
                                      output_path=lev_frames_path)

    # ── Player Coverage ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Player coverage analysis")
    print("=" * 60)

    coverage_2024 = identify_target_players(
        euro2024_events, euro2024_frames, EURO_2024_PLAYERS, "Euro 2024"
    )
    coverage_2020 = identify_target_players(
        euro2020_events, euro2020_frames, EURO_2020_PLAYERS, "Euro 2020"
    )

    all_coverage = pd.concat([coverage_2024, coverage_2020], ignore_index=True)

    # Save coverage table
    coverage_path = OUTPUTS_DIR / "player_coverage.csv"
    all_coverage.to_csv(coverage_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 INGESTION COMPLETE — FILE SUMMARY")
    print("=" * 70)

    files_to_check = [
        RAW_DIR / "euro2024_events.parquet",
        RAW_DIR / "euro2024_matches.parquet",
        RAW_DIR / "euro2024_frames.pkl",
        RAW_DIR / "euro2020_events.parquet",
        RAW_DIR / "euro2020_matches.parquet",
        RAW_DIR / "euro2020_frames.pkl",
        RAW_DIR / "frame_lookup.pkl",
    ]
    for fp in files_to_check:
        size_mb = fp.stat().st_size / 1e6 if fp.exists() else 0
        status = "OK" if fp.exists() else "MISSING"
        print(f"  [{status}] {fp.name:45s}  {size_mb:7.2f} MB")

    print("\n" + "=" * 70)
    print("PLAYER COVERAGE TABLE")
    print("=" * 70)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 35)
    print(
        all_coverage[
            ['full_name', 'team', 'tournament', 'matches_played',
             'total_passes', 'passes_with_ff', 'sufficient_data']
        ].to_string(index=False)
    )

    insufficient = all_coverage[~all_coverage['sufficient_data']]
    if len(insufficient) > 0:
        print(f"\n  WARNING: {len(insufficient)} player(s) have insufficient data:")
        for _, row in insufficient.iterrows():
            print(
                f"    - {row['full_name'] or row['search_term']} "
                f"({row['tournament']}): "
                f"{row['total_passes']} passes, {row['passes_with_ff']} with FF"
            )

    return {
        'euro2024_events': euro2024_events,
        'euro2024_frames': euro2024_frames,
        'euro2020_events': euro2020_events,
        'euro2020_frames': euro2020_frames,
        'leverkusen_frames': lev_frames,
        'coverage': all_coverage,
    }


if __name__ == "__main__":
    run_tournament_ingestion()
