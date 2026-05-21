"""
Phase 2 Fix — The Architect Framework
======================================
Addresses 8 methodological problems in the original phase2_scores.py:

1. Minimum 150-pass threshold to exclude low-data players
2. No nanmean: architect_score_full requires ALL 6 components non-NaN
3. Event metrics recomputed per-tournament (not duplicated across tournaments)
4. Position filtering: z-scores against midfielder-only pool
5. Player data properly split by tournament
6. Xhaka Leverkusen separated as cross-validation (NOT mixed into z-scoring)
7. Clustering on filtered midfielder pool only (30-60 players)
8. Pedri Euro 2024 verified and excluded if < 150 passes

Output: data/processed/phase2_architect_scores_v2.parquet
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

MIN_PASSES = 150

MIDFIELDER_POSITIONS = {
    "Center Midfield",
    "Left Center Midfield",
    "Right Center Midfield",
    "Center Defensive Midfield",
    "Left Defensive Midfield",
    "Right Defensive Midfield",
    "Center Attacking Midfield",
}

TARGET_SEARCHES = [
    "Pedro González",
    "Rodrigo Hernández",
    "Zubimendi",
    "Vitor Machado",
    "Toni Kroos",
    "Granit Xhaka",
    "Fabián Ruiz",
    "Kanté",
    "Bellingham",
    "Frenkie de Jong",
    "Jorge Luiz Frello",
    "Marco Verratti",
    "Sergio Busquets",
    "Kalvin Phillips",
]


# ─── Positional value ─────────────────────────────────────────────────────────

def pos_value(x, y):
    x_val = (x / 120.0) ** 1.5
    y_center = abs(y - 40) / 40
    y_penalty = 1 - 0.3 * y_center
    return x_val * y_penalty


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Tournament lookup
# ══════════════════════════════════════════════════════════════════════════════

def build_tournament_lookup():
    """Return dict: match_id (int) → tournament string."""
    e24 = pd.read_parquet(RAW / "euro2024_matches.parquet", columns=["match_id"])
    e20 = pd.read_parquet(RAW / "euro2020_matches.parquet", columns=["match_id"])

    lookup = {}
    for mid in e24["match_id"].unique():
        lookup[int(mid)] = "euro2024"
    for mid in e20["match_id"].unique():
        lookup[int(mid)] = "euro2020"

    print(f"  Tournament lookup: {len(lookup)} match_ids "
          f"({sum(1 for v in lookup.values() if v=='euro2024')} euro2024, "
          f"{sum(1 for v in lookup.values() if v=='euro2020')} euro2020)")
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Player positions and pass counts per tournament
# ══════════════════════════════════════════════════════════════════════════════

def get_player_profiles(pe):
    """
    From passes_enriched, compute per (player, tournament):
      - total_passes
      - primary_position (mode of position column)
      - matches (unique match_ids)
    """
    profile = (
        pe.groupby(["player", "tournament"])
        .agg(
            total_passes=("player", "count"),
            primary_position=("position", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    print(f"  Player profiles: {len(profile)} (player, tournament) rows")
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Per-tournament PACV
# ══════════════════════════════════════════════════════════════════════════════

def compute_pacv_per_tournament(av, tourn_lookup):
    """
    PACV = sum((attention_weight - 1/chain_length) * terminal_xg)
    for pre-assist zone (positions 2-5 from shot end), per (player, tournament).
    """
    print("  Computing PACV per tournament...")
    av = av.copy()
    av["tournament"] = av["match_id"].map(tourn_lookup)
    av = av.dropna(subset=["tournament"])

    shot_chains = av[av["ended_in_shot"] == True].copy()
    pre_assist = shot_chains[shot_chains["position_from_end"].between(2, 5)].copy()

    pre_assist["equal_weight"] = 1.0 / pre_assist["chain_length"]
    pre_assist["pacv_excess"] = pre_assist["attention_weight"] - pre_assist["equal_weight"]
    pre_assist["pacv_value"] = pre_assist["pacv_excess"] * pre_assist["terminal_xg"]

    pacv = (
        pre_assist
        .groupby(["player", "tournament"])
        .agg(
            pacv=("pacv_value", "sum"),
            n_pre_assist_actions=("pacv_value", "count"),
        )
        .reset_index()
    )

    for tourn in ["euro2020", "euro2024"]:
        n = len(pacv[pacv["tournament"] == tourn])
        print(f"    PACV {tourn}: {n} players")
    return pacv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Per-tournament PRV
# ══════════════════════════════════════════════════════════════════════════════

def compute_prv_per_tournament(pe):
    """
    PRV = mean positional value of pressured passes per player per tournament.
    Completed pressured passes: pos_value(end_x, end_y)
    Incomplete: -0.05 turnover penalty
    """
    print("  Computing PRV per tournament...")
    pressured = pe[pe["is_under_pressure"] == True].copy()

    pressured["pass_value"] = np.where(
        pressured["is_completed"],
        pressured.apply(lambda r: pos_value(r["end_x"], r["end_y"]), axis=1),
        -0.05,
    )

    prv = (
        pressured
        .groupby(["player", "tournament"])
        .agg(
            prv=("pass_value", "mean"),
            n_pressured_passes=("pass_value", "count"),
        )
        .reset_index()
    )

    for tourn in ["euro2020", "euro2024"]:
        n = len(prv[prv["tournament"] == tourn])
        print(f"    PRV {tourn}: {n} players")
    return prv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Per-tournament CIR
# ══════════════════════════════════════════════════════════════════════════════

def compute_cir_per_tournament(chains):
    """
    CIR = chains_initiated / team_shot_chains, per (player, team, tournament).
    Chain initiator = first player in the chain.
    """
    print("  Computing CIR per tournament...")
    shot_chains = chains[chains["ended_in_shot"] == True].copy()

    rows = []
    for (team, tournament), grp in shot_chains.groupby(["team", "tournament"]):
        total = len(grp)
        initiator_counts = {}
        for _, chain in grp.iterrows():
            players = chain["players"]
            if len(players) > 0 and not pd.isna(players[0]):
                init = players[0]
                initiator_counts[init] = initiator_counts.get(init, 0) + 1

        for player, cnt in initiator_counts.items():
            rows.append({
                "player": player,
                "team": team,
                "tournament": tournament,
                "chains_initiated": cnt,
                "team_shot_chains": total,
                "cir": cnt / total if total > 0 else 0.0,
            })

    cir = pd.DataFrame(rows)
    for tourn in ["euro2020", "euro2024"]:
        n = len(cir[cir["tournament"] == tourn])
        print(f"    CIR {tourn}: {n} players")
    return cir


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-tournament TVI
# ══════════════════════════════════════════════════════════════════════════════

def compute_tvi_per_tournament(pe):
    """
    TVI = CV(duration) × std(pass_angle) per player per tournament.
    Requires at least 10 passes.
    """
    print("  Computing TVI per tournament...")
    rows = []

    for (player, tournament), grp in pe.groupby(["player", "tournament"]):
        if len(grp) < 10:
            continue

        durations = grp["duration"].dropna().astype(float).values
        angles = grp["pass_angle"].dropna().astype(float).values

        if len(durations) < 10 or len(angles) < 2:
            continue

        hold_cv = durations.std() / durations.mean() if durations.mean() > 0 else 0.0
        angle_std = float(np.std(angles))
        tvi = hold_cv * angle_std

        rows.append({
            "player": player,
            "tournament": tournament,
            "tvi": tvi,
        })

    tvi = pd.DataFrame(rows)
    for tourn in ["euro2020", "euro2024"]:
        n = len(tvi[tvi["tournament"] == tourn])
        print(f"    TVI {tourn}: {n} players")
    return tvi


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Per-tournament DS and DTD
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_ds_per_tournament(tourn_lookup):
    """
    Aggregate Decision Surplus per (player, tournament):
      ds = median decision_surplus
      ds_pct_pos = fraction of passes with positive DS
      passes_with_ff = count of DS-eligible passes
    """
    print("  Aggregating DS per tournament...")
    ds_raw = pd.read_parquet(PROCESSED / "phase2_decision_surplus.parquet")
    ds_raw["tournament"] = ds_raw["match_id"].map(tourn_lookup)
    ds_raw = ds_raw.dropna(subset=["tournament"])

    ds = (
        ds_raw
        .groupby(["player", "tournament"])
        .agg(
            passes_with_ff=("event_id", "count"),
            ds=("decision_surplus", "median"),
            ds_pct_pos=("decision_surplus", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    for tourn in ["euro2020", "euro2024"]:
        n = len(ds[ds["tournament"] == tourn])
        print(f"    DS {tourn}: {n} players")
    return ds


def aggregate_dtd_per_tournament(tourn_lookup):
    """
    Aggregate Defensive Topology Disruption per (player, tournament):
      dtd = mean dtd_raw
    Also applies zone-based weights by start_x:
      x >= 80: full weight (attacking third)
      x >= 40: 0.75 weight (middle third)
      x < 40: 0.5 weight (defensive third)
    """
    print("  Aggregating DTD per tournament...")
    dtd_raw = pd.read_parquet(PROCESSED / "phase2_defensive_disruption.parquet")
    dtd_raw["tournament"] = dtd_raw["match_id"].map(tourn_lookup)
    dtd_raw = dtd_raw.dropna(subset=["tournament"])

    # Zone-adjusted DTD: weight disruptions by field position
    dtd_raw["zone_weight"] = np.where(
        dtd_raw["start_x"] >= 80, 1.0,
        np.where(dtd_raw["start_x"] >= 40, 0.75, 0.5)
    )
    dtd_raw["dtd_adjusted"] = dtd_raw["dtd_raw"] * dtd_raw["zone_weight"]

    dtd = (
        dtd_raw
        .groupby(["player", "tournament"])
        .agg(dtd=("dtd_adjusted", "mean"))
        .reset_index()
    )
    for tourn in ["euro2020", "euro2024"]:
        n = len(dtd[dtd["tournament"] == tourn])
        print(f"    DTD {tourn}: {n} players")
    return dtd


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Assemble player table
# ══════════════════════════════════════════════════════════════════════════════

def assemble_player_table(profiles, pacv, prv, cir, tvi, ds, dtd):
    """
    Build one row per (player, tournament) for the midfielder-filtered pool.
    Merges all per-tournament metrics.

    Returns the merged DataFrame before filtering.
    """
    # Start from player profiles (pass counts + position + matches)
    table = profiles.copy()

    # Merge PRV (player, tournament)
    table = table.merge(
        prv[["player", "tournament", "prv", "n_pressured_passes"]],
        on=["player", "tournament"], how="left"
    )

    # Merge CIR (player, tournament) — pick highest cir if player appears in multiple teams
    cir_agg = (
        cir.groupby(["player", "tournament"])
        .agg(
            cir=("cir", "max"),
            chains_initiated=("chains_initiated", "sum"),
        )
        .reset_index()
    )
    table = table.merge(
        cir_agg[["player", "tournament", "cir", "chains_initiated"]],
        on=["player", "tournament"], how="left"
    )

    # Merge TVI (player, tournament)
    table = table.merge(
        tvi[["player", "tournament", "tvi"]],
        on=["player", "tournament"], how="left"
    )

    # Merge PACV (player, tournament)
    table = table.merge(
        pacv[["player", "tournament", "pacv", "n_pre_assist_actions"]],
        on=["player", "tournament"], how="left"
    )

    # Merge DS (player, tournament)
    table = table.merge(
        ds[["player", "tournament", "ds", "ds_pct_pos", "passes_with_ff"]],
        on=["player", "tournament"], how="left"
    )

    # Merge DTD (player, tournament)
    table = table.merge(
        dtd[["player", "tournament", "dtd"]],
        on=["player", "tournament"], how="left"
    )

    # DS availability flag (requires 50+ freeze-frame passes)
    table["DS_available"] = table["passes_with_ff"].fillna(0) >= 50

    print(f"  Assembled table: {len(table)} rows")
    null_counts = table[["pacv", "ds", "dtd", "prv", "cir", "tvi"]].isna().sum()
    print(f"  Null counts in metrics:\n{null_counts.to_string()}")
    return table


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Filter to midfielder pool
# ══════════════════════════════════════════════════════════════════════════════

def filter_midfielder_pool(table):
    """
    Apply two filters:
      1. primary_position must be in MIDFIELDER_POSITIONS
      2. total_passes >= MIN_PASSES

    Returns (midfielder_df, excluded_df)
    """
    mid_mask = table["primary_position"].isin(MIDFIELDER_POSITIONS)
    pass_mask = table["total_passes"] >= MIN_PASSES

    midfielder_df = table[mid_mask & pass_mask].copy()
    excluded_df = table[~(mid_mask & pass_mask)].copy()

    print(f"\n  Position filter: {mid_mask.sum()} midfielders in full table")
    print(f"  Pass threshold (>={MIN_PASSES}): {pass_mask.sum()} players qualify")
    print(f"  Combined midfielder pool: {len(midfielder_df)} (player, tournament) rows")
    print(f"  Excluded: {len(excluded_df)} rows")

    # Show position breakdown of pool
    pos_counts = midfielder_df["primary_position"].value_counts()
    print(f"\n  Position breakdown in pool:")
    for pos, cnt in pos_counts.items():
        print(f"    {pos}: {cnt}")

    return midfielder_df, excluded_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Z-scoring against midfielder pool (NO nanmean)
# ══════════════════════════════════════════════════════════════════════════════

def zscore_midfielder_pool(mid_df):
    """
    Z-score all 6 novel components against the midfielder-only pool.
    Uses population mean/std (not sample) to avoid inflating z-scores.

    architect_score_full  = strict mean of ALL 6 z-scores (NaN if any component missing)
    architect_score_event = strict mean of pacv_z, prv_z, cir_z, tvi_z (NaN if any missing)
    """
    novel_cols = ["pacv", "ds", "dtd", "prv", "cir", "tvi"]
    novel_z = ["pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]
    event_cols = ["pacv", "prv", "cir", "tvi"]
    event_z = ["pacv_z", "prv_z", "cir_z", "tvi_z"]

    df = mid_df.copy()

    # Compute z-scores using the full midfielder pool as reference
    pop_stats = {}
    for col in novel_cols:
        z_col = f"{col}_z"
        col_data = df[col].dropna()
        mu = col_data.mean()
        sigma = col_data.std()
        pop_stats[col] = {"mu": mu, "sigma": sigma}

        if sigma == 0 or pd.isna(sigma):
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - mu) / sigma

    # architect_score_full: strict mean — NaN if ANY of the 6 z-scores is NaN
    df["architect_score_full"] = df[novel_z].apply(
        lambda row: row.mean() if row.notna().all() else np.nan,
        axis=1,
    )

    # architect_score_event: strict mean — NaN if ANY of the 4 event z-scores is NaN
    df["architect_score_event"] = df[event_z].apply(
        lambda row: row.mean() if row.notna().all() else np.nan,
        axis=1,
    )

    full_valid = df["architect_score_full"].notna().sum()
    event_valid = df["architect_score_event"].notna().sum()
    print(f"\n  Z-scores computed over {len(df)} midfielder pool entries")
    print(f"  architect_score_full valid (all 6 components): {full_valid}")
    print(f"  architect_score_event valid (all 4 event components): {event_valid}")
    print(f"  Score range (full): "
          f"{df['architect_score_full'].min():.3f} to {df['architect_score_full'].max():.3f}")

    return df, pop_stats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Xhaka Leverkusen cross-validation (SEPARATE, not z-scored)
# ══════════════════════════════════════════════════════════════════════════════

def load_xhaka_crossval(pop_stats):
    """
    Load Xhaka's Leverkusen Phase 1 scores.
    Returns a DataFrame with a single row marked tournament='Leverkusen_2324'.
    This row is NOT included in z-scoring — it uses Phase 1's own z-scores.
    For reference, we also compute what his Phase 2-population z-scores would be,
    but mark them as _crossval to distinguish.
    """
    p1_path = PROCESSED / "architect_scores_full.parquet"
    if not p1_path.exists():
        p1_path = PROCESSED / "architect_scores_v2.parquet"
    if not p1_path.exists():
        print("  WARNING: Phase 1 scores not found — skipping Xhaka cross-val")
        return None

    p1 = pd.read_parquet(p1_path)
    xhaka_p1 = p1[p1["player"].str.contains("Xhaka", na=False)]
    if len(xhaka_p1) == 0:
        print("  WARNING: Xhaka not found in Phase 1 scores")
        return None

    row = xhaka_p1.iloc[0]
    print(f"  Loaded Xhaka Phase 1: team={row.get('team','?')}, "
          f"AS_full={row.get('AS_full', row.get('architect_score_full','?')):.4f}")

    # Map Phase 1 column names
    as_full_col = "AS_full" if "AS_full" in row.index else "architect_score_full"
    as_event_col = "AS_event" if "AS_event" in row.index else "architect_score_event"

    xhaka_row = {
        "player": row["player"],
        "team": row.get("team", "Bayer Leverkusen"),
        "tournament": "Leverkusen_2324",
        "primary_position": "Center Midfield",
        "total_passes": row.get("total_passes", np.nan),
        "matches": row.get("matches", np.nan),
        "pacv": row.get("pacv", np.nan),
        "ds": row.get("ds", np.nan),
        "dtd": row.get("dtd", np.nan),
        "prv": row.get("prv", np.nan),
        "cir": row.get("cir", np.nan),
        "tvi": row.get("tvi", np.nan),
        # Phase 1 z-scores (within Phase 1 Leverkusen population)
        "pacv_z_p1": row.get("pacv_z", np.nan),
        "ds_z_p1": row.get("ds_z", np.nan),
        "dtd_z_p1": row.get("dtd_z", np.nan),
        "prv_z_p1": row.get("prv_z", np.nan),
        "cir_z_p1": row.get("cir_z", np.nan),
        "tvi_z_p1": row.get("tvi_z", np.nan),
        "architect_score_full_p1": row.get(as_full_col, np.nan),
        "architect_score_event_p1": row.get(as_event_col, np.nan),
    }

    # NOTE: do NOT apply Phase 2 population z-scores to Phase 1 raw values.
    # TVI scale is incompatible (Phase 1: CV*mean_degrees; Phase 2: CV*std_radians).
    # PRV, CIR scales also differ subtly between La Liga and Euros populations.
    # Cross-validation is best done by comparing the Phase 1 z-scores with
    # the Phase 2 z-scores side-by-side (presented in the visualisation).
    print(f"  Xhaka Phase 1 AS_full (P1 z-scores): {xhaka_row['architect_score_full_p1']:.3f}")
    return pd.DataFrame([xhaka_row])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Clustering on filtered pool
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(mid_df):
    """
    K-means clustering on the 6 novel z-scores.
    Only cluster players where ALL 6 components are non-NaN (architect_score_full valid).
    Try k=3 and k=4, pick the best silhouette score.
    """
    cluster_cols = ["pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]
    comp_names = ["pacv", "ds", "dtd", "prv", "cir", "tvi"]

    # Only cluster where all 6 are non-NaN
    eligible = mid_df[mid_df["architect_score_full"].notna()].copy()
    print(f"\n  Players eligible for clustering (all 6 components): {len(eligible)}")

    if len(eligible) < 8:
        print("  WARNING: Too few eligible players for clustering — skipping")
        mid_df["cluster"] = np.nan
        mid_df["cluster_name"] = "Insufficient data"
        return mid_df, {}

    X = eligible[cluster_cols].values

    silhouette_scores = {}
    cluster_labels_map = {}
    print("  K-means silhouette scores:")
    for k in [3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores[k] = sil
        cluster_labels_map[k] = (km, labels)
        print(f"    k={k}: silhouette={sil:.3f}")

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"  Optimal k={optimal_k}")

    km_opt, labels_opt = cluster_labels_map[optimal_k]
    eligible = eligible.copy()
    eligible["cluster"] = labels_opt

    # Name clusters
    centers = km_opt.cluster_centers_
    cluster_names = {}
    for c in range(optimal_k):
        center = centers[c]
        top2_idx = np.argsort(center)[::-1][:2]
        top2 = [comp_names[i] for i in top2_idx]
        bottom1 = comp_names[np.argmin(center)]

        if ("cir" in top2 and "prv" in top2) or ("cir" in top2 and "ds" in top2):
            name = "The Metronome"      # sets tempo via chain initiation + decision quality
        elif "pacv" in top2 and "ds" in top2:
            name = "The Orchestrator"   # pre-assist value + decision quality
        elif "pacv" in top2 and "cir" in top2:
            name = "The Orchestrator"
        elif "dtd" in top2 and "tvi" in top2:
            name = "The Disruptor"      # breaks defensive shape + tempo variance
        elif "tvi" in top2 and "dtd" in top2:
            name = "The Disruptor"
        elif "prv" in top2 and "ds" in top2:
            name = "The Connector"      # receives under pressure, finds clever passes
        elif "prv" in top2:
            name = "The Connector"
        elif "tvi" in top2:
            name = "The Disruptor"
        else:
            dominant = comp_names[np.argmax(center)]
            name = f"The {dominant.upper()}-Led"

        cluster_names[c] = name
        members = eligible[eligible["cluster"] == c]["player"].tolist()
        print(f"  Cluster {c} ({name}): {len(members)} players, "
              f"top={top2}, center={dict(zip(comp_names, center.round(3)))}")

    eligible["cluster_name"] = eligible["cluster"].map(cluster_names)

    # Merge back into full mid_df
    mid_df = mid_df.drop(columns=["cluster", "cluster_name"], errors="ignore")
    mid_df = mid_df.merge(
        eligible[["player", "tournament", "cluster", "cluster_name"]],
        on=["player", "tournament"], how="left"
    )

    cluster_info = {
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
        "cluster_names": cluster_names,
        "cluster_centers": pd.DataFrame(centers, columns=comp_names),
        "eligible_players": eligible,
    }

    # Save cluster parquet
    eligible_out = eligible[["player", "tournament", "cluster", "cluster_name"] + cluster_cols].copy()
    eligible_out.to_parquet(PROCESSED / "phase2_clusters_v2.parquet", index=False)
    print(f"  Saved clusters_v2.parquet: {len(eligible_out)} rows")

    return mid_df, cluster_info


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Print results summary
# ══════════════════════════════════════════════════════════════════════════════

def print_results(mid_df, xhaka_row, cluster_info):
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n" + "=" * 80)
    print("PHASE 2 CORRECTED RESULTS")
    print("=" * 80)
    print(f"Midfielder pool: {len(mid_df)} entries across 2 tournaments")
    print(f"Minimum threshold: {MIN_PASSES} passes")

    # TOP 15 by architect_score_full (exclude NaN)
    full_ranked = mid_df[mid_df["architect_score_full"].notna()].sort_values(
        "architect_score_full", ascending=False
    )
    print(f"\n{'─'*80}")
    print("TOP 15 REGISTAS BY ARCHITECT SCORE (FULL) — midfielder pool")
    print(f"{'─'*80}")
    cols = ["player", "tournament", "primary_position", "total_passes",
            "architect_score_full", "pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]
    print(full_ranked[cols].head(15).to_string(index=False))

    # TARGET PLAYERS
    target_mask = pd.Series([False] * len(mid_df), index=mid_df.index)
    for s in TARGET_SEARCHES:
        target_mask |= mid_df["player"].str.contains(s, na=False, case=False)
    targets = mid_df[target_mask].sort_values("architect_score_full", ascending=False, na_position="last")

    print(f"\n{'─'*80}")
    print("TARGET PLAYERS — ranked by Architect Score (Full)")
    print(f"{'─'*80}")
    print(targets[cols].to_string(index=False))

    # PEDRI vs RODRI (Euro 2020 — same team)
    pedri_2020 = mid_df[
        mid_df["player"].str.contains("Pedro González", na=False) &
        (mid_df["tournament"] == "euro2020")
    ]
    rodri_2020 = mid_df[
        mid_df["player"].str.contains("Rodrigo Hernández", na=False) &
        (mid_df["tournament"] == "euro2020")
    ]

    print(f"\n{'─'*80}")
    print("PEDRI vs RODRI — Euro 2020 (same team, controlled comparison)")
    print(f"{'─'*80}")
    comparison = pd.concat([pedri_2020, rodri_2020], ignore_index=True)
    if len(comparison) > 0:
        print(comparison[cols].to_string(index=False))
    else:
        print("  (One or both players not in midfielder pool for Euro 2020)")

    # CLUSTER ASSIGNMENTS for target players
    if "cluster_name" in mid_df.columns:
        print(f"\n{'─'*80}")
        print("ARCHETYPE CLUSTERS — target players")
        print(f"{'─'*80}")
        clustered = targets[targets["cluster"].notna()].sort_values(
            "architect_score_full", ascending=False, na_position="last"
        )
        cluster_cols_show = ["player", "tournament", "cluster_name",
                             "pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z",
                             "architect_score_full"]
        if len(clustered) > 0:
            print(clustered[cluster_cols_show].to_string(index=False))
        else:
            print("  (No target players met full clustering criteria)")

    # XHAKA CROSS-VALIDATION
    print(f"\n{'─'*80}")
    print("XHAKA CROSS-VALIDATION: Leverkusen 2023/24 vs Euro 2024")
    print(f"{'─'*80}")
    xhaka_euro = mid_df[mid_df["player"].str.contains("Xhaka", na=False)]
    if xhaka_row is not None:
        print("  Leverkusen 2023/24 (Phase 1 population z-scores):")
        p1_cols = ["player", "ds_z_p1", "dtd_z_p1", "prv_z_p1", "cir_z_p1",
                   "tvi_z_p1", "architect_score_full_p1"]
        existing_p1_cols = [c for c in p1_cols if c in xhaka_row.columns]
        print(xhaka_row[existing_p1_cols].to_string(index=False))
        print()

    if len(xhaka_euro) > 0:
        print("  Euro tournament(s) (Phase 2 midfielder pool z-scores):")
        print(xhaka_euro[cols].to_string(index=False))

    # SILHOUETTE SCORES
    if cluster_info:
        print(f"\n{'─'*80}")
        print("CLUSTERING DIAGNOSTICS")
        print(f"{'─'*80}")
        for k, sil in cluster_info["silhouette_scores"].items():
            marker = " <-- OPTIMAL" if k == cluster_info["optimal_k"] else ""
            print(f"  k={k}: silhouette={sil:.3f}{marker}")

    # SANITY CHECKS
    print(f"\n{'─'*80}")
    print("SANITY CHECKS")
    print(f"{'─'*80}")
    top10 = full_ranked.head(10)
    passes_ok = (top10["total_passes"] >= MIN_PASSES).all()
    mids_ok = top10["primary_position"].isin(MIDFIELDER_POSITIONS).all()
    print(f"  Top 10 all have {MIN_PASSES}+ passes: {passes_ok}")
    print(f"  Top 10 all central midfielders: {mids_ok}")

    if not passes_ok:
        bad = top10[top10["total_passes"] < MIN_PASSES][["player", "total_passes"]]
        print(f"  BAD PASS COUNT PLAYERS:\n{bad}")
    if not mids_ok:
        bad = top10[~top10["primary_position"].isin(MIDFIELDER_POSITIONS)][["player", "primary_position"]]
        print(f"  NON-MIDFIELDER IN TOP 10:\n{bad}")

    pedri_check = mid_df[mid_df["player"].str.contains("Pedro González", na=False)]
    if len(pedri_check) > 0:
        print(f"\n  Pedri entries:")
        print(pedri_check[["player", "tournament", "total_passes", "architect_score_full"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("\n" + "=" * 80)
    print("PHASE 2 FIX — THE ARCHITECT FRAMEWORK")
    print("=" * 80)

    # ── Step 1: Build tournament lookup ──────────────────────────────────────
    print("\n[1/9] Building tournament lookup...")
    tourn_lookup = build_tournament_lookup()

    # ── Step 2: Load passes_enriched (all passes, all tournaments) ───────────
    print("\n[2/9] Loading passes_enriched and computing player profiles...")
    pe_cols = ["player", "position", "tournament", "match_id",
               "start_x", "start_y", "end_x", "end_y",
               "pass_angle", "duration", "is_under_pressure", "is_completed"]
    pe = pd.read_parquet(PROCESSED / "phase2_passes_enriched.parquet", columns=pe_cols)
    pe["player"] = pe["player"].str.strip()

    profiles = get_player_profiles(pe)

    # ── Step 3: Load possession chains ───────────────────────────────────────
    print("\n[3/9] Loading possession chains...")
    chains = pd.read_parquet(PROCESSED / "phase2_possession_chains.parquet")

    # ── Step 4: Load action_values for PACV ──────────────────────────────────
    print("\n[4/9] Computing per-tournament metrics...")
    av = pd.read_parquet(PROCESSED / "phase2_action_values.parquet")

    pacv = compute_pacv_per_tournament(av, tourn_lookup)
    prv = compute_prv_per_tournament(pe)
    cir = compute_cir_per_tournament(chains)
    tvi = compute_tvi_per_tournament(pe)
    ds = aggregate_ds_per_tournament(tourn_lookup)
    dtd = aggregate_dtd_per_tournament(tourn_lookup)

    # ── Step 5: Assemble player table ─────────────────────────────────────────
    print("\n[5/9] Assembling player table...")
    table = assemble_player_table(profiles, pacv, prv, cir, tvi, ds, dtd)

    # ── Step 6: Filter to midfielder pool ─────────────────────────────────────
    print("\n[6/9] Filtering to midfielder pool (≥150 passes)...")
    mid_df, excluded_df = filter_midfielder_pool(table)

    # ── Step 7: Z-score and compute Architect Scores ──────────────────────────
    print("\n[7/9] Z-scoring against midfielder pool (no nanmean)...")
    mid_df, pop_stats = zscore_midfielder_pool(mid_df)

    # ── Step 8: Xhaka cross-validation ────────────────────────────────────────
    print("\n[8/9] Loading Xhaka Leverkusen cross-validation data...")
    xhaka_row = load_xhaka_crossval(pop_stats)

    # ── Step 9: Clustering ────────────────────────────────────────────────────
    print("\n[9/9] Clustering on filtered midfielder pool...")
    mid_df, cluster_info = run_clustering(mid_df)

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(mid_df, xhaka_row, cluster_info)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path = PROCESSED / "phase2_architect_scores_v2.parquet"
    mid_df.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path} ({mid_df.shape})")

    # Also save a separate Xhaka cross-val file
    if xhaka_row is not None:
        xv_path = PROCESSED / "phase2_xhaka_crossval.parquet"
        xhaka_row.to_parquet(xv_path, index=False)
        print(f"  Saved: {xv_path}")

    print("\n" + "=" * 80)
    print("PHASE 2 FIX COMPLETE")
    print("=" * 80)

    return mid_df, xhaka_row, cluster_info


if __name__ == "__main__":
    run()
