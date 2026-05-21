"""
Phase 2 Historical Comparison — The Architect Framework
Pulls StatsBomb La Liga open data for Barcelona/Real Madrid seasons and computes
event-based Architect Score metrics (PACV, PRV, CIR, TVI) for Xavi Hernández,
Sergio Busquets, Andrés Iniesta, and Luka Modrić.

No 360 freeze frames available → only 4 event-based metrics computed.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Import helpers from phase2_pipeline ──────────────────────────────────────
sys.path.insert(0, str(ROOT))
from src.phase2_pipeline import (
    pos_value,
    compute_chain_positions,
    compute_cir,
    compute_tvi,
    TURNOVER_PENALTY,
)

# ─── Target player substrings ────────────────────────────────────────────────
TARGET_SUBSTRINGS = ["Xavi", "Busquets", "Iniesta", "Modrić", "Modric"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pull La Liga Events
# ══════════════════════════════════════════════════════════════════════════════

def pull_la_liga_events(force_refresh: bool = False) -> pd.DataFrame:
    """
    Pull all available La Liga season data from StatsBomb open data.
    Caches to data/raw/la_liga_historical_events.parquet.

    Returns events DataFrame with added match metadata columns.
    """
    cache_path = RAW_DIR / "la_liga_historical_events.parquet"

    if cache_path.exists() and not force_refresh:
        print(f"[pull_la_liga_events] Cache found — loading from {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(df):,} events from cache")
        return df

    from statsbombpy import sb

    print("[pull_la_liga_events] Fetching La Liga seasons from StatsBomb open data...")

    comps = sb.competitions()
    la_liga_seasons = comps[comps["competition_id"] == 11].copy()
    print(f"  Found {len(la_liga_seasons)} La Liga seasons")
    print(la_liga_seasons[["season_id", "season_name"]].to_string(index=False))

    all_events = []
    total_matches = 0
    t0 = time.time()

    for _, season_row in la_liga_seasons.iterrows():
        sid = season_row["season_id"]
        sname = season_row["season_name"]

        matches = sb.matches(competition_id=11, season_id=sid)
        n_matches = len(matches)
        total_matches += n_matches
        print(f"\n  Season {sname} (id={sid}): {n_matches} matches")

        for i, (_, match_row) in enumerate(matches.iterrows()):
            mid = match_row["match_id"]

            try:
                events = sb.events(match_id=mid)
            except Exception as e:
                print(f"    WARN: match {mid} failed: {e}")
                continue

            # Add match metadata
            events["match_id"] = mid
            events["home_team_name"] = match_row.get("home_team", "")
            events["away_team_name"] = match_row.get("away_team", "")
            events["home_score"] = match_row.get("home_score", None)
            events["away_score"] = match_row.get("away_score", None)
            events["match_date"] = match_row.get("match_date", None)
            events["season_id"] = sid
            events["season_name"] = sname

            all_events.append(events)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(
                    f"    {i+1}/{n_matches} matches "
                    f"({len(all_events):,} event chunks, "
                    f"{rate:.1f} matches/s)"
                )

        print(f"  Done season {sname}: {i+1} matches processed")

    print(f"\n  Concatenating {len(all_events)} match DataFrames...")
    combined = pd.concat(all_events, ignore_index=True)
    print(f"  Total: {len(combined):,} events across {total_matches} matches")

    # Save to cache
    combined.to_parquet(cache_path, index=False)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"  Saved to {cache_path} ({size_mb:.1f} MB)")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Pass Enrichment
# ══════════════════════════════════════════════════════════════════════════════

def enrich_passes_for_historical(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich pass events with derived geometry columns.
    Same logic as phase2_pipeline.enrich_passes() but writes to a
    la_liga-specific path and does NOT overwrite the phase2 enriched file.
    """
    print("\n[enrich_passes_for_historical] Enriching passes...")

    passes = events_df[events_df["type"] == "Pass"].copy()
    print(f"  Raw pass count: {len(passes):,}")

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

    passes["start_x"] = passes["location"].apply(_x)
    passes["start_y"] = passes["location"].apply(_y)
    passes["end_x"] = passes["pass_end_location"].apply(_x)
    passes["end_y"] = passes["pass_end_location"].apply(_y)

    # Derived geometry
    passes["dist_toward_goal"] = passes["end_x"] - passes["start_x"]
    passes["is_progressive"] = passes["dist_toward_goal"] >= 10
    passes["is_completed"] = passes["pass_outcome"].isna()
    passes["lateral_dist"] = (passes["end_y"] - passes["start_y"]).abs()

    # pass_length & pass_angle — use existing StatsBomb columns if present
    if "pass_length" not in passes.columns or passes["pass_length"].isna().all():
        dx = passes["end_x"] - passes["start_x"]
        dy = passes["end_y"] - passes["start_y"]
        passes["pass_length"] = np.sqrt(dx ** 2 + dy ** 2)

    if "pass_angle" not in passes.columns or passes["pass_angle"].isna().all():
        dx = passes["end_x"] - passes["start_x"]
        dy = passes["end_y"] - passes["start_y"]
        passes["pass_angle"] = np.arctan2(dy, dx)

    # Fill NaN pass_length/pass_angle from coordinates where missing
    mask_nan_len = passes["pass_length"].isna()
    mask_nan_ang = passes["pass_angle"].isna()
    if mask_nan_len.any():
        dx = passes.loc[mask_nan_len, "end_x"] - passes.loc[mask_nan_len, "start_x"]
        dy = passes.loc[mask_nan_len, "end_y"] - passes.loc[mask_nan_len, "start_y"]
        passes.loc[mask_nan_len, "pass_length"] = np.sqrt(dx ** 2 + dy ** 2)
    if mask_nan_ang.any():
        dx = passes.loc[mask_nan_ang, "end_x"] - passes.loc[mask_nan_ang, "start_x"]
        dy = passes.loc[mask_nan_ang, "end_y"] - passes.loc[mask_nan_ang, "start_y"]
        passes.loc[mask_nan_ang, "pass_angle"] = np.arctan2(dy, dx)

    # Boolean flags
    if "pass_switch" in passes.columns:
        passes["is_switch"] = passes["pass_switch"].fillna(False).astype(bool)
    else:
        passes["is_switch"] = False

    if "pass_through_ball" in passes.columns:
        passes["is_through_ball"] = passes["pass_through_ball"].fillna(False).astype(bool)
    else:
        passes["is_through_ball"] = False

    if "pass_cross" in passes.columns:
        passes["is_cross"] = passes["pass_cross"].fillna(False).astype(bool)
    else:
        passes["is_cross"] = False

    passes["is_under_pressure"] = passes["under_pressure"].fillna(False).astype(bool)

    print(f"  Enriched {len(passes):,} passes")
    return passes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Possession Chain Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_chains_historical(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract possession chains from La Liga events.
    Same logic as phase2_pipeline.extract_chains().
    """
    print("\n[extract_chains_historical] Extracting possession chains...")

    action_types = [
        "Pass", "Carry", "Shot", "Dribble", "Ball Receipt*",
        "Clearance", "Miscontrol", "Dispossessed", "Interception",
        "Ball Recovery", "Foul Won", "Goal Keeper",
    ]

    chains = []
    grouped = events_df.groupby(["match_id", "possession"])

    for (match_id, poss_num), poss_events in grouped:
        poss_events = poss_events.sort_values("index")
        poss_team = poss_events["possession_team"].iloc[0]

        team_actions = poss_events[
            (poss_events["team"] == poss_team)
            & (poss_events["type"].isin(action_types))
        ].copy()

        if len(team_actions) < 2:
            continue

        shots = poss_events[poss_events["type"] == "Shot"]
        if len(shots) > 0:
            last_shot = shots.iloc[-1]
            terminal_xg = last_shot.get("shot_statsbomb_xg", 0)
            terminal_xg = 0 if pd.isna(terminal_xg) else float(terminal_xg)
            ended_in_shot = True
            shot_outcome = last_shot.get("shot_outcome", "Unknown")
        else:
            terminal_xg = 0.0
            ended_in_shot = False
            shot_outcome = None

        season_id = poss_events["season_id"].iloc[0] if "season_id" in poss_events.columns else None
        season_name = poss_events["season_name"].iloc[0] if "season_name" in poss_events.columns else None

        chains.append({
            "match_id": match_id,
            "possession_num": poss_num,
            "team": poss_team,
            "season_id": season_id,
            "season_name": season_name,
            "n_actions": len(team_actions),
            "terminal_xg": terminal_xg,
            "ended_in_shot": ended_in_shot,
            "shot_outcome": shot_outcome,
            "play_pattern": poss_events["play_pattern"].iloc[0],
            "start_minute": team_actions["minute"].iloc[0],
            "action_ids": team_actions["id"].tolist(),
            "action_types": team_actions["type"].tolist(),
            "players": team_actions["player"].tolist(),
            "player_ids": team_actions["player_id"].tolist(),
            "locations": team_actions["location"].tolist(),
            "timestamps": team_actions["timestamp"].tolist(),
            "durations": team_actions["duration"].tolist(),
        })

    chain_df = pd.DataFrame(chains)
    n_shot = chain_df["ended_in_shot"].sum()
    print(f"  {len(chain_df):,} chains ({n_shot:,} ending in shots)")
    return chain_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Compute Historical Metrics Per Player Per Season
# ══════════════════════════════════════════════════════════════════════════════

def _compute_prv_for_historical(passes_df: pd.DataFrame) -> pd.DataFrame:
    """
    PRV: mean positional value of completed pressured passes per player/season.
    """
    pressured = passes_df[passes_df["is_under_pressure"]].copy()
    pressured["pass_value"] = pressured.apply(
        lambda p: pos_value(p["end_x"], p["end_y"]) if p["is_completed"] else TURNOVER_PENALTY,
        axis=1,
    )

    # Per player + season
    prv = pressured.groupby(["player", "player_id", "team", "season_id", "season_name"]).agg(
        n_pressured_passes=("pass_value", "count"),
        total_prv=("pass_value", "sum"),
        mean_prv=("pass_value", "mean"),
        pressured_completion=("is_completed", "mean"),
    ).reset_index()

    return prv


def _compute_cir_per_season(chains_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    CIR: fraction of each team's shot chains initiated by the player (per season).
    """
    shot_chains = chains_df[chains_df["ended_in_shot"]].copy()

    player_id_lookup = {}
    if "player_id" in events_df.columns:
        player_id_lookup = events_df.groupby("player")["player_id"].first().to_dict()

    rows = []
    for (team, season_id, season_name), team_chains in shot_chains.groupby(
        ["team", "season_id", "season_name"]
    ):
        total = len(team_chains)
        initiator_counts = {}
        initiator_player_ids = {}

        for _, chain in team_chains.iterrows():
            players = chain["players"]
            player_ids = chain.get("player_ids", [])
            if len(players) > 0 and not pd.isna(players[0]):
                init = players[0]
                initiator_counts[init] = initiator_counts.get(init, 0) + 1
                if init not in initiator_player_ids:
                    pid = player_ids[0] if len(player_ids) > 0 else None
                    initiator_player_ids[init] = pid

        for player, cnt in initiator_counts.items():
            pid = initiator_player_ids.get(player) or player_id_lookup.get(player, None)
            rows.append({
                "player": player,
                "player_id": pid,
                "team": team,
                "season_id": season_id,
                "season_name": season_name,
                "chains_initiated": cnt,
                "team_shot_chains": total,
                "cir": cnt / total if total > 0 else 0,
            })

    return pd.DataFrame(rows)


def _compute_tvi_per_season(events_df: pd.DataFrame, min_passes: int = 50) -> pd.DataFrame:
    """
    TVI per player per season: CV(hold_duration) × mean(|direction_change_degrees|).
    """
    passes = events_df[events_df["type"] == "Pass"].copy()

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

    if "start_x" not in passes.columns:
        passes["start_x"] = passes["location"].apply(_x)
        passes["start_y"] = passes["location"].apply(_y)
    if "end_x" not in passes.columns:
        passes["end_x"] = passes["pass_end_location"].apply(_x)
        passes["end_y"] = passes["pass_end_location"].apply(_y)

    if "pass_angle" in passes.columns and passes["pass_angle"].notna().any():
        passes["_out_angle"] = passes["pass_angle"]
    else:
        passes["_out_angle"] = np.arctan2(
            passes["end_y"] - passes["start_y"],
            passes["end_x"] - passes["start_x"],
        )

    results = []
    for (player, season_id, season_name), grp in passes.groupby(
        ["player", "season_id", "season_name"]
    ):
        if len(grp) < min_passes:
            continue

        hold_times = grp["duration"].dropna().astype(float).values
        if len(hold_times) < 10:
            continue

        grp_sorted = grp.sort_values(
            ["match_id", "index"] if "index" in grp.columns else ["match_id"]
        )
        angles = grp_sorted["_out_angle"].dropna().values
        if len(angles) < 2:
            continue

        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        direction_changes = np.abs(np.degrees(angle_diffs))

        hold_cv = hold_times.std() / hold_times.mean() if hold_times.mean() > 0 else 0
        mean_dir_change = float(np.mean(direction_changes))
        tvi = hold_cv * mean_dir_change

        team = grp["team"].mode().iloc[0]
        results.append({
            "player": player,
            "team": team,
            "season_id": season_id,
            "season_name": season_name,
            "n_passes": len(grp),
            "hold_time_cv": float(hold_cv),
            "mean_dir_change": mean_dir_change,
            "tvi": float(tvi),
        })

    return pd.DataFrame(results)


def _compute_chain_positions_per_season(
    chains_df: pd.DataFrame, events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute PACV-related chain position profiles per player per season.
    PACV = total_xg_involved * pre_assist_ratio
    """
    shot_chains = chains_df[chains_df["ended_in_shot"]].copy()

    player_id_lookup = {}
    if "player_id" in events_df.columns:
        player_id_lookup = events_df.groupby("player")["player_id"].first().to_dict()

    all_profiles = []

    for (season_id, season_name), s_chains in shot_chains.groupby(
        ["season_id", "season_name"]
    ):
        profiles = {}

        for _, chain in s_chains.iterrows():
            n = chain["n_actions"]
            xg = chain["terminal_xg"]
            chain_player_ids = chain.get("player_ids", [])

            for i, (player, action_type) in enumerate(
                zip(chain["players"], chain["action_types"])
            ):
                if pd.isna(player):
                    continue
                pos_from_end = n - 1 - i

                if player not in profiles:
                    pid = (
                        chain_player_ids[i]
                        if i < len(chain_player_ids)
                        else player_id_lookup.get(player, None)
                    )
                    profiles[player] = {
                        "player_id": pid,
                        "shot_pos": 0,
                        "assist_pos": 0,
                        "pre_assist_2": 0,
                        "pre_assist_3": 0,
                        "pre_assist_4_5": 0,
                        "buildup_6_plus": 0,
                        "total_chain_actions": 0,
                        "total_xg_involved": 0.0,
                        "chains_involved": set(),
                        "team": chain["team"],
                    }

                p = profiles[player]
                p["total_chain_actions"] += 1
                p["total_xg_involved"] += xg
                p["chains_involved"].add((chain["match_id"], chain["possession_num"]))

                if pos_from_end == 0:
                    p["shot_pos"] += 1
                elif pos_from_end == 1:
                    p["assist_pos"] += 1
                elif pos_from_end <= 3:
                    p["pre_assist_2" if pos_from_end == 2 else "pre_assist_3"] += 1
                elif pos_from_end <= 5:
                    p["pre_assist_4_5"] += 1
                else:
                    p["buildup_6_plus"] += 1

        for player, p in profiles.items():
            n_c = len(p["chains_involved"])
            pre_assist_actions = p["pre_assist_2"] + p["pre_assist_3"] + p["pre_assist_4_5"]
            pre_assist_ratio = (
                pre_assist_actions / p["total_chain_actions"]
                if p["total_chain_actions"] > 0
                else 0
            )
            pacv = p["total_xg_involved"] * pre_assist_ratio

            all_profiles.append({
                "player": player,
                "player_id": p["player_id"],
                "team": p["team"],
                "season_id": season_id,
                "season_name": season_name,
                "chains_involved": n_c,
                "total_chain_actions": p["total_chain_actions"],
                "total_xg_involved": p["total_xg_involved"],
                "pre_assist_actions": pre_assist_actions,
                "pre_assist_ratio": pre_assist_ratio,
                "pacv": pacv,
            })

    return pd.DataFrame(all_profiles)


def compute_historical_metrics(
    events_df: pd.DataFrame,
    chains_df: pd.DataFrame,
    passes_df: pd.DataFrame,
    target_players: list,
) -> pd.DataFrame:
    """
    Compute all 4 event-based metrics per player per season.

    Returns a merged DataFrame with columns:
    player, player_id, team, season_id, season_name, n_passes,
    pacv, prv, cir, tvi
    """
    print("\n[compute_historical_metrics] Computing metrics per player per season...")

    # ── PACV via chain positions ──────────────────────────────────────────────
    print("  Computing chain positions (PACV)...")
    pacv_df = _compute_chain_positions_per_season(chains_df, events_df)
    print(f"  Chain positions: {len(pacv_df):,} rows")

    # ── PRV ──────────────────────────────────────────────────────────────────
    print("  Computing PRV...")
    prv_df = _compute_prv_for_historical(passes_df)
    print(f"  PRV: {len(prv_df):,} rows")

    # ── CIR ──────────────────────────────────────────────────────────────────
    print("  Computing CIR...")
    cir_df = _compute_cir_per_season(chains_df, events_df)
    print(f"  CIR: {len(cir_df):,} rows")

    # ── TVI ──────────────────────────────────────────────────────────────────
    print("  Computing TVI...")
    tvi_df = _compute_tvi_per_season(events_df, min_passes=50)
    print(f"  TVI: {len(tvi_df):,} rows")

    # ── Pass counts per player/season ─────────────────────────────────────────
    n_passes_df = (
        passes_df.groupby(["player", "player_id", "team", "season_id", "season_name"])
        .size()
        .reset_index(name="n_passes")
    )

    # ── Merge all metrics ─────────────────────────────────────────────────────
    merge_keys = ["player", "season_id", "season_name"]

    df = n_passes_df.merge(
        pacv_df[merge_keys + ["pacv", "pre_assist_ratio", "total_xg_involved"]],
        on=merge_keys,
        how="left",
    )

    df = df.merge(
        prv_df[merge_keys + ["mean_prv"]].rename(columns={"mean_prv": "prv"}),
        on=merge_keys,
        how="left",
    )

    df = df.merge(
        cir_df[merge_keys + ["cir"]],
        on=merge_keys,
        how="left",
    )

    df = df.merge(
        tvi_df[merge_keys + ["tvi"]],
        on=merge_keys,
        how="left",
    )

    # Fill NaNs with 0 for players with no metric data
    for col in ["pacv", "prv", "cir", "tvi"]:
        df[col] = df[col].fillna(0.0)

    print(f"  Merged metrics: {len(df):,} rows, {df['player'].nunique():,} unique players")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Assemble Historical Comparison
# ══════════════════════════════════════════════════════════════════════════════

def assemble_historical_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score the 4 metrics against the full La Liga midfielder population
    (all players with >= 100 passes across any season).

    Compute architect_score_event = mean of 4 z-scores.
    Add Xhaka's Leverkusen Phase 1 row for comparison.

    Returns DataFrame with columns:
    player, team, competition, season_name, n_passes,
    pacv, prv, cir, tvi,
    pacv_z, prv_z, cir_z, tvi_z, architect_score_event
    """
    print("\n[assemble_historical_comparison] Assembling comparison...")

    # ── Normalization pool: all players with >= 100 passes in any season ──────
    pool = metrics_df[metrics_df["n_passes"] >= 100].copy()
    print(f"  Normalization pool: {len(pool):,} player-seasons, {pool['player'].nunique():,} unique players")

    # Z-score parameters from pool
    z_stats = {}
    for metric in ["pacv", "prv", "cir", "tvi"]:
        mu = pool[metric].mean()
        sigma = pool[metric].std()
        z_stats[metric] = (mu, sigma)
        print(f"  {metric}: mean={mu:.4f}, std={sigma:.4f}")

    def _zscore(series, metric):
        mu, sigma = z_stats[metric]
        if sigma == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    # Z-score the pool
    pool = pool.copy()
    for metric in ["pacv", "prv", "cir", "tvi"]:
        pool[f"{metric}_z"] = _zscore(pool[metric], metric)

    pool["architect_score_event"] = pool[["pacv_z", "prv_z", "cir_z", "tvi_z"]].mean(axis=1)
    pool["competition"] = "La Liga"

    # ── Load Xhaka Phase 1 Leverkusen data ────────────────────────────────────
    xhaka_row = None
    for fname in ["architect_scores_full.parquet", "architect_scores_final.parquet"]:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            phase1_df = pd.read_parquet(fpath)
            xhaka_rows = phase1_df[phase1_df["player"].str.contains("Xhaka", na=False)]
            if len(xhaka_rows) > 0:
                xhaka_row = xhaka_rows.iloc[0]
                print(f"  Loaded Xhaka from {fname}")
                break

    comparison_rows = []

    if xhaka_row is not None:
        # Build Xhaka comparison row using his Phase 1 raw metrics,
        # z-scored against the La Liga pool for fair comparison
        xhaka_pacv = float(xhaka_row.get("pacv", 0))
        xhaka_prv = float(xhaka_row.get("prv", 0))
        xhaka_cir = float(xhaka_row.get("cir", 0))
        xhaka_tvi = float(xhaka_row.get("tvi", 0))

        xhaka_comparison = {
            "player": "Granit Xhaka",
            "team": "Bayer Leverkusen",
            "competition": "Bundesliga",
            "season_name": "2023/24",
            "n_passes": int(xhaka_row.get("total_passes", 0)),
            "pacv": xhaka_pacv,
            "prv": xhaka_prv,
            "cir": xhaka_cir,
            "tvi": xhaka_tvi,
            "pacv_z": _zscore(pd.Series([xhaka_pacv]), "pacv").iloc[0],
            "prv_z": _zscore(pd.Series([xhaka_prv]), "prv").iloc[0],
            "cir_z": _zscore(pd.Series([xhaka_cir]), "cir").iloc[0],
            "tvi_z": _zscore(pd.Series([xhaka_tvi]), "tvi").iloc[0],
        }
        xhaka_comparison["architect_score_event"] = np.mean([
            xhaka_comparison["pacv_z"],
            xhaka_comparison["prv_z"],
            xhaka_comparison["cir_z"],
            xhaka_comparison["tvi_z"],
        ])
        comparison_rows.append(xhaka_comparison)
        print(
            f"  Xhaka (Bundesliga 2023/24): AS_event={xhaka_comparison['architect_score_event']:.3f} "
            f"(pacv_z={xhaka_comparison['pacv_z']:.3f}, prv_z={xhaka_comparison['prv_z']:.3f}, "
            f"cir_z={xhaka_comparison['cir_z']:.3f}, tvi_z={xhaka_comparison['tvi_z']:.3f})"
        )
    else:
        print("  WARN: Xhaka Phase 1 data not found")

    # ── Assemble output columns ───────────────────────────────────────────────
    output_cols = [
        "player", "team", "competition", "season_id", "season_name",
        "n_passes", "pacv", "prv", "cir", "tvi",
        "pacv_z", "prv_z", "cir_z", "tvi_z", "architect_score_event",
    ]

    pool_out = pool.copy()
    pool_out["competition"] = "La Liga"
    pool_out = pool_out[[c for c in output_cols if c in pool_out.columns]]
    # Add missing cols with None
    for c in output_cols:
        if c not in pool_out.columns:
            pool_out[c] = None

    # Add Xhaka row
    if comparison_rows:
        xhaka_df = pd.DataFrame(comparison_rows)
        xhaka_df["season_id"] = None
        for c in output_cols:
            if c not in xhaka_df.columns:
                xhaka_df[c] = None
        xhaka_df = xhaka_df[output_cols]
        result = pd.concat([pool_out[output_cols], xhaka_df], ignore_index=True)
    else:
        result = pool_out[output_cols]

    print(f"  Final comparison: {len(result):,} player-seasons, {result['player'].nunique():,} unique players")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_historical_ingestion(force_refresh: bool = False):
    """
    Orchestrate full historical comparison pipeline:
    1. Pull La Liga events (cached)
    2. Enrich passes
    3. Extract possession chains
    4. Compute metrics
    5. Assemble comparison
    6. Print results
    7. Save output
    """
    pipeline_start = time.time()

    print("\n" + "=" * 70)
    print("PHASE 2 HISTORICAL COMPARISON — THE ARCHITECT FRAMEWORK")
    print("=" * 70)

    # ── Step 1: Pull data ──────────────────────────────────────────────────────
    print("\n[1/5] Pulling La Liga events (StatsBomb open data)...")
    events_df = pull_la_liga_events(force_refresh=force_refresh)
    print(f"  Events loaded: {len(events_df):,}")
    print(f"  Seasons: {events_df['season_name'].nunique()}")
    print(f"  Matches: {events_df['match_id'].nunique():,}")

    # Spot-check target players
    print("\n  Target player check:")
    for substr in TARGET_SUBSTRINGS:
        found = events_df[events_df["player"].str.contains(substr, na=False)]["player"].value_counts()
        if len(found) > 0:
            print(f"    {substr}: {list(found.index)[:3]}")
        else:
            print(f"    {substr}: NOT FOUND")

    # ── Step 2: Enrich passes ──────────────────────────────────────────────────
    print("\n[2/5] Enriching passes...")
    passes_df = enrich_passes_for_historical(events_df)

    # ── Step 3: Extract chains ─────────────────────────────────────────────────
    print("\n[3/5] Extracting possession chains...")
    chains_df = extract_chains_historical(events_df)

    # ── Step 4: Compute metrics ────────────────────────────────────────────────
    print("\n[4/5] Computing historical metrics...")
    metrics_df = compute_historical_metrics(
        events_df, chains_df, passes_df, TARGET_SUBSTRINGS
    )

    # ── Step 5: Assemble comparison ────────────────────────────────────────────
    print("\n[5/5] Assembling historical comparison...")
    comparison_df = assemble_historical_comparison(metrics_df)

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "historical_comparison.parquet"
    comparison_df.to_parquet(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved historical_comparison.parquet ({size_kb:.0f} KB, {len(comparison_df):,} rows)")

    # Also save metrics_df for downstream use
    metrics_path = PROCESSED_DIR / "la_liga_metrics_per_season.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"  Saved la_liga_metrics_per_season.parquet ({metrics_path.stat().st_size/1024:.0f} KB)")

    # ── Print results ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("RESULTS: TARGET PLAYER METRICS")
    print("=" * 70)

    # Per-season breakdown for target players
    target_mask = comparison_df["player"].apply(
        lambda p: any(
            s.lower() in str(p).lower() for s in TARGET_SUBSTRINGS + ["Xhaka"]
        )
    )
    target_df = comparison_df[target_mask].copy()

    print("\n  Per-season architect scores (sorted by player then season):")
    target_display = target_df[
        ["player", "team", "season_name", "n_passes", "pacv_z", "prv_z", "cir_z", "tvi_z", "architect_score_event"]
    ].sort_values(["player", "season_name"])
    print(target_display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Career aggregate summary
    print("\n  Career aggregate (all seasons combined):")
    career_agg = (
        target_df.groupby("player")
        .agg(
            team=("team", "first"),
            seasons_found=("season_name", "nunique"),
            total_passes=("n_passes", "sum"),
            mean_pacv_z=("pacv_z", "mean"),
            mean_prv_z=("prv_z", "mean"),
            mean_cir_z=("cir_z", "mean"),
            mean_tvi_z=("tvi_z", "mean"),
            mean_architect_score_event=("architect_score_event", "mean"),
            peak_architect_score_event=("architect_score_event", "max"),
        )
        .reset_index()
        .sort_values("mean_architect_score_event", ascending=False)
    )
    print(career_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Normalization pool stats
    la_liga_pool = comparison_df[comparison_df["competition"] == "La Liga"]
    print(f"\n  La Liga normalization pool: {la_liga_pool['player'].nunique():,} unique players")
    print(f"  Total La Liga player-seasons: {len(la_liga_pool):,}")

    # Not-found targets
    found_names = set(comparison_df["player"].dropna().tolist())
    for substr in TARGET_SUBSTRINGS:
        if not any(substr.lower() in n.lower() for n in found_names):
            print(f"\n  WARNING: '{substr}' NOT FOUND in La Liga data")

    print(f"\n  Total pipeline time: {total_elapsed/60:.1f} min")
    print("=" * 70)

    return {
        "events_df": events_df,
        "passes_df": passes_df,
        "chains_df": chains_df,
        "metrics_df": metrics_df,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Historical Comparison")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-pull all data even if cache exists",
    )
    args = parser.parse_args()

    results = run_historical_ingestion(force_refresh=args.force_refresh)
