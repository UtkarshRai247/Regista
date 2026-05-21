"""
Phase 2 Score Assembly — The Architect Framework
=================================================
Merges all Phase 2 tournament metrics into a cross-player comparison DataFrame,
z-scores vs full tournament midfielder population, computes Architect Scores,
and runs K-means archetype clustering.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"

# ─── Target player search strings ────────────────────────────────────────────
TARGET_SEARCHES = [
    "Pedro González",   # Pedri (Euro 2024 + 2020)
    "Rodrigo Hernández",  # Rodri
    "Zubimendi",
    "Vitor Machado",    # Vitinha
    "Toni Kroos",
    "Granit Xhaka",
    "Fabián Ruiz",
    "N'Golo Kanté",     # also N''Golo
    "Kanté",
    "Bellingham",
    "Frenkie de Jong",
    "Jorge Luiz Frello",  # Jorginho
    "Marco Verratti",
    "Sergio Busquets",
    "Kalvin Phillips",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Compute per-player PACV from action_values
# ══════════════════════════════════════════════════════════════════════════════

def compute_pacv():
    """
    PACV (Pre-Assist Chain Value) = excess attention credit in pre-assist zone
    (positions 2-5 from shot end) for shot-ending chains.

    Formula: sum(attention_weight - equal_weight) * terminal_xg per player
    where equal_weight = 1/chain_length

    Returns player-level DataFrame with columns: player, team, pacv, n_pre_assist_actions
    """
    av_path = PROCESSED / "phase2_action_values.parquet"
    av = pd.read_parquet(av_path)

    print(f"  action_values shape: {av.shape}")
    print(f"  action_values columns: {av.columns.tolist()}")

    # Check if we have the expected columns
    required = {"player", "team", "ended_in_shot", "position_from_end",
                "attention_weight", "chain_length", "terminal_xg"}
    if not required.issubset(set(av.columns)):
        print("  WARNING: action_values missing expected columns — using fallback PACV")
        return _pacv_fallback()

    print(f"  Using Transformer attention-based PACV computation")
    shot_chains = av[av["ended_in_shot"] == True].copy()
    print(f"  Shot-ending chain actions: {len(shot_chains):,}")

    # Pre-assist zone: positions 2-5 from shot end
    pre_assist = shot_chains[shot_chains["position_from_end"].between(2, 5)].copy()
    print(f"  Pre-assist zone rows: {len(pre_assist):,}")

    pre_assist["equal_weight"] = 1.0 / pre_assist["chain_length"]
    pre_assist["pacv_excess"] = pre_assist["attention_weight"] - pre_assist["equal_weight"]
    # Weight excess by chain terminal xg to get value-based PACV
    pre_assist["pacv_value"] = pre_assist["pacv_excess"] * pre_assist["terminal_xg"]

    pacv_df = (
        pre_assist
        .groupby(["player", "team"])
        .agg(
            pacv=("pacv_value", "sum"),
            n_pre_assist_actions=("pacv_value", "count"),
        )
        .reset_index()
    )

    print(f"  PACV computed for {len(pacv_df):,} players")
    return pacv_df


def _pacv_fallback():
    """
    Fallback PACV from chain_positions:
    pacv = total_xg_involved * pre_assist_ratio
    """
    print("  Using fallback PACV from chain_positions")
    cp = pd.read_parquet(PROCESSED / "phase2_chain_positions.parquet")
    cp["pacv"] = cp["total_xg_involved"] * cp["pre_assist_ratio"]
    cp = cp.rename(columns={"pre_assist_actions": "n_pre_assist_actions"})
    return cp[["player", "team", "pacv", "n_pre_assist_actions"]]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Aggregate DS and DTD per player (with tournament)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_ds_dtd():
    """
    Aggregate Decision Surplus and Defensive Topology Disruption per
    (player, team, tournament) — joining tournament via match_id.

    DS:
      ds         = median decision_surplus per player
      ds_pct_pos = fraction of passes with positive DS
      passes_with_ff = count of DS-eligible passes (freeze-frame passes)

    DTD:
      dtd = mean dtd_raw per player

    Returns (ds_df, dtd_df) each indexed by (player, team, tournament).
    """
    # ── Build match → tournament lookup ──────────────────────────────────────
    pe = pd.read_parquet(PROCESSED / "phase2_passes_enriched.parquet",
                         columns=["match_id", "tournament"])
    match_tourn = pe[["match_id", "tournament"]].drop_duplicates()

    # ── Decision Surplus ─────────────────────────────────────────────────────
    ds_raw = pd.read_parquet(PROCESSED / "phase2_decision_surplus.parquet")
    ds_raw = ds_raw.merge(match_tourn, on="match_id", how="left")

    ds_df = (
        ds_raw
        .groupby(["player", "team", "tournament"])
        .agg(
            passes_with_ff=("event_id", "count"),
            ds=("decision_surplus", "median"),
            ds_pct_pos=("decision_surplus", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    print(f"  DS aggregated: {len(ds_df):,} (player, team, tournament) rows")

    # ── Defensive Topology Disruption ────────────────────────────────────────
    dtd_raw = pd.read_parquet(PROCESSED / "phase2_defensive_disruption.parquet")
    dtd_raw = dtd_raw.merge(match_tourn, on="match_id", how="left")

    dtd_df = (
        dtd_raw
        .groupby(["player", "team", "tournament"])
        .agg(dtd=("dtd_raw", "mean"))
        .reset_index()
    )
    print(f"  DTD aggregated: {len(dtd_df):,} (player, team, tournament) rows")

    return ds_df, dtd_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Assemble player-level scores
# ══════════════════════════════════════════════════════════════════════════════

def assemble_player_scores(pacv_df, ds_df, dtd_df):
    """
    Build one row per (player, team, tournament) with all metrics.

    Base: passes_enriched → player, team, tournament, total_passes
    Joined metrics (not tournament-split):
      - baselines:  matches, progressive_passes, sca, xa (actual_assists)
      - prv_df:     prv (mean_prv)
      - cir_df:     cir
      - tvi_df:     tvi
      - pacv_df:    pacv (no tournament split — whole-tournament value)
    Tournament-aware:
      - ds_df:      ds, ds_pct_pos, passes_with_ff
      - dtd_df:     dtd
    """
    # ── Step 1: player-tournament base from passes_enriched ─────────────────
    pe = pd.read_parquet(
        PROCESSED / "phase2_passes_enriched.parquet",
        columns=["type", "player", "team", "tournament", "id"]
    )
    pe_passes = pe[pe["type"] == "Pass"]

    base = (
        pe_passes
        .groupby(["player", "team", "tournament"])
        .agg(total_passes_event=("id", "count"))
        .reset_index()
    )
    base["player"] = base["player"].str.strip()
    base["team"] = base["team"].str.strip()
    print(f"  Base (player × tournament): {len(base):,} rows")

    # ── Step 2: Load and clean per-player metrics ────────────────────────────
    bl = pd.read_parquet(PROCESSED / "phase2_baselines.parquet")
    bl["player"] = bl["player"].str.strip()
    bl["team"] = bl["team"].str.strip()
    bl = bl.rename(columns={"actual_assists": "xa"})

    prv = pd.read_parquet(PROCESSED / "phase2_prv.parquet")
    prv["player"] = prv["player"].str.strip()
    prv["team"] = prv["team"].str.strip()
    prv = prv.rename(columns={"mean_prv": "prv"})
    prv = prv[["player", "team", "prv", "n_pressured_passes"]]

    cir = pd.read_parquet(PROCESSED / "phase2_cir.parquet")
    cir["player"] = cir["player"].str.strip()
    cir["team"] = cir["team"].str.strip()
    cir = cir[["player", "team", "cir", "chains_initiated"]]

    tvi = pd.read_parquet(PROCESSED / "phase2_tvi.parquet")
    tvi["player"] = tvi["player"].str.strip()
    tvi["team"] = tvi["team"].str.strip()
    tvi = tvi[["player", "team", "tvi"]]

    pacv_clean = pacv_df.copy()
    pacv_clean["player"] = pacv_clean["player"].str.strip()
    pacv_clean["team"] = pacv_clean["team"].str.strip()

    ds_clean = ds_df.copy()
    ds_clean["player"] = ds_clean["player"].str.strip()
    ds_clean["team"] = ds_clean["team"].str.strip()

    dtd_clean = dtd_df.copy()
    dtd_clean["player"] = dtd_clean["player"].str.strip()
    dtd_clean["team"] = dtd_clean["team"].str.strip()

    # ── Step 3: Merge ────────────────────────────────────────────────────────
    scores = base.copy()

    # baselines (not tournament-specific)
    scores = scores.merge(
        bl[["player", "team", "matches", "total_passes", "progressive_passes", "sca", "xa"]],
        on=["player", "team"], how="left"
    )
    # Use event-count passes where baselines is missing
    scores["total_passes"] = scores["total_passes"].fillna(scores["total_passes_event"])
    scores = scores.drop(columns=["total_passes_event"])

    # PRV (not tournament-specific — combined across both tournaments)
    scores = scores.merge(prv, on=["player", "team"], how="left")

    # CIR (not tournament-specific)
    scores = scores.merge(cir, on=["player", "team"], how="left")

    # TVI (not tournament-specific)
    scores = scores.merge(tvi, on=["player", "team"], how="left")

    # PACV (not tournament-specific — Transformer trained on combined data)
    scores = scores.merge(pacv_clean, on=["player", "team"], how="left")

    # DS (tournament-specific)
    scores = scores.merge(ds_clean, on=["player", "team", "tournament"], how="left")

    # DTD (tournament-specific)
    scores = scores.merge(dtd_clean, on=["player", "team", "tournament"], how="left")

    # ── Step 4: DS availability flag ─────────────────────────────────────────
    scores["DS_available"] = scores["passes_with_ff"].fillna(0) >= 50

    print(f"  Assembled scores shape: {scores.shape}")
    print(f"  DS available for {scores['DS_available'].sum():,} rows")

    # ── Step 5: Check merge quality ──────────────────────────────────────────
    null_counts = scores[["pacv", "ds", "dtd", "prv", "cir", "tvi"]].isna().sum()
    print(f"  Null counts in key metrics:\n{null_counts}")

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Xhaka Leverkusen Phase 1 cross-validation row
# ══════════════════════════════════════════════════════════════════════════════

def load_xhaka_phase1():
    """
    Load Xhaka's Leverkusen 2023/24 row from Phase 1 and reformat
    to match Phase 2 column schema for direct comparison.

    Phase 1 column mapping:
      xa                  → xa
      progressive_passes  → progressive_passes
      sca                 → sca
      pacv                → pacv
      ds (= median_ds)    → ds
      dtd (= mean_dtd)    → dtd
      prv (= mean_prv)    → prv
      cir                 → cir
      tvi                 → tvi
      z-scores: already named pacv_z, ds_z, etc.
      AS_full             → architect_score_full
      AS_event            → architect_score_event
      trad_score          → traditional_score
    """
    p1 = pd.read_parquet(PROCESSED / "architect_scores_full.parquet")
    xhaka_p1 = p1[p1["player"].str.contains("Xhaka", na=False)].copy()

    if len(xhaka_p1) == 0:
        print("  WARNING: Xhaka not found in Phase 1 scores")
        return None

    row = xhaka_p1.iloc[0].copy()

    xhaka_row = {
        "player": row["player"],
        "team": row["team"],
        "tournament": "Leverkusen_2324",
        "matches": row.get("matches", np.nan),
        "total_passes": row.get("total_passes", np.nan),
        "passes_with_ff": np.nan,
        "DS_available": False,
        "n_pre_assist_actions": np.nan,
        "n_pressured_passes": row.get("n_pressured_passes", np.nan),
        "chains_initiated": np.nan,
        "xa": row.get("xa", np.nan),
        "progressive_passes": row.get("progressive_passes", np.nan),
        "sca": row.get("sca", np.nan),
        "pacv": row.get("pacv", np.nan),
        "ds": row.get("ds", np.nan),
        "dtd": row.get("dtd", np.nan),
        "prv": row.get("prv", np.nan),
        "cir": row.get("cir", np.nan),
        "tvi": row.get("tvi", np.nan),
        # Phase 1 z-scores (normalized within Phase 1 population, for reference)
        "pacv_z_p1": row.get("pacv_z", np.nan),
        "ds_z_p1": row.get("ds_z", np.nan),
        "dtd_z_p1": row.get("dtd_z", np.nan),
        "prv_z_p1": row.get("prv_z", np.nan),
        "cir_z_p1": row.get("cir_z", np.nan),
        "tvi_z_p1": row.get("tvi_z", np.nan),
        "architect_score_full_p1": row.get("AS_full", np.nan),
        "architect_score_event_p1": row.get("AS_event", np.nan),
        "traditional_score_p1": row.get("trad_score", np.nan),
    }

    print(f"  Loaded Xhaka Phase 1 row: team={xhaka_row['team']}, "
          f"pacv={xhaka_row['pacv']:.4f}, AS_full={xhaka_row['architect_score_full_p1']:.4f}")

    return pd.DataFrame([xhaka_row])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Z-scoring and Architect Score computation
# ══════════════════════════════════════════════════════════════════════════════

def zscore_and_architect(scores_df):
    """
    Z-score all metrics against FULL tournament midfielder population
    (all rows in scores_df, no pre-filtering).

    Novel components: pacv, ds, dtd, prv, cir, tvi
    Traditional:      xa, progressive_passes, sca

    Architect scores:
      architect_score_full  = mean of 6 novel z-scores
      architect_score_event = mean of pacv_z, prv_z, cir_z, tvi_z
      traditional_score     = mean of xa_z, progressive_passes_z, sca_z
    """
    novel_cols = ["pacv", "ds", "dtd", "prv", "cir", "tvi"]
    trad_cols = ["xa", "progressive_passes", "sca"]

    # Use full population for normalization parameters
    for col in novel_cols + trad_cols:
        z_col = f"{col}_z"
        col_data = scores_df[col]
        mu = col_data.mean()
        sigma = col_data.std()
        if sigma == 0 or pd.isna(sigma):
            scores_df[z_col] = 0.0
        else:
            scores_df[z_col] = (col_data - mu) / sigma

    # Architect Scores
    novel_z = ["pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]
    event_z = ["pacv_z", "prv_z", "cir_z", "tvi_z"]
    trad_z = ["xa_z", "progressive_passes_z", "sca_z"]

    scores_df["architect_score_full"] = scores_df[novel_z].mean(axis=1, skipna=True)
    scores_df["architect_score_event"] = scores_df[event_z].mean(axis=1, skipna=True)
    scores_df["traditional_score"] = scores_df[trad_z].mean(axis=1, skipna=True)

    print(f"  Z-scores computed over {len(scores_df):,} players")
    print(f"  Architect Score (full) range: "
          f"{scores_df['architect_score_full'].min():.3f} to "
          f"{scores_df['architect_score_full'].max():.3f}")

    return scores_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — K-means archetype clustering
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(scores_df):
    """
    K-means clustering on 6 novel z-scored components for players with
    sufficient data (100+ total passes AND DS_available = True, i.e. 50+ FF passes).

    Tries k=3,4,5 and selects optimal k by silhouette score.
    Names clusters based on dominant sub-components.

    Returns (scores_with_cluster_df, cluster_info_dict)
    """
    cluster_cols = ["pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]

    # Filter for players with sufficient data
    eligible = scores_df[
        (scores_df["total_passes"].fillna(0) >= 100) &
        (scores_df["DS_available"] == True)
    ].copy()

    # Drop rows with any NaN in clustering columns
    eligible_clean = eligible.dropna(subset=cluster_cols).copy()
    print(f"\n  Players eligible for clustering: {len(eligible_clean):,} "
          f"(out of {len(scores_df):,} total)")

    if len(eligible_clean) < 10:
        print("  WARNING: Too few eligible players for clustering")
        scores_df["cluster"] = np.nan
        scores_df["cluster_name"] = "Insufficient data"
        return scores_df, {}

    X = eligible_clean[cluster_cols].values

    silhouette_scores = {}
    cluster_labels_map = {}
    print("\n  K-means clustering results:")
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores[k] = sil
        cluster_labels_map[k] = (km, labels)
        print(f"    k={k}: silhouette={sil:.3f}")

    # Optimal k
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n  Optimal k={optimal_k} (silhouette={silhouette_scores[optimal_k]:.3f})")

    km_opt, labels_opt = cluster_labels_map[optimal_k]
    eligible_clean = eligible_clean.copy()
    eligible_clean["cluster"] = labels_opt

    # ── Name clusters based on dominant components ────────────────────────────
    cluster_names = {}
    centers = km_opt.cluster_centers_
    # centers shape: (k, 6) corresponding to [pacv_z, ds_z, dtd_z, prv_z, cir_z, tvi_z]
    comp_names = ["pacv", "ds", "dtd", "prv", "cir", "tvi"]

    for c in range(optimal_k):
        center = centers[c]
        top2_idx = np.argsort(center)[::-1][:2]
        top2 = [comp_names[i] for i in top2_idx]
        bottom1_idx = np.argmin(center)
        bottom1 = comp_names[bottom1_idx]

        if "pacv" in top2 and "cir" in top2:
            name = "The Orchestrator"       # controls chains end-to-end
        elif "ds" in top2 and "dtd" in top2:
            name = "The Disruptor"          # hard passes, breaks defensive structure
        elif "prv" in top2 and bottom1 == "tvi":
            name = "The Metronome"          # calm under pressure, consistent tempo
        elif "pacv" in top2 and "prv" in top2:
            name = "The Creator"            # high pre-assist value + press-resistance
        elif "ds" in top2 and "pacv" in top2:
            name = "The Progressive"        # decision quality + chain influence
        elif "cir" in top2 and "prv" in top2:
            name = "The Initiator"          # chain starts + press resistance
        elif "tvi" in top2:
            name = "The Tempo-Setter"       # high variance tempo player
        elif "dtd" in top2:
            name = "The Press-Breaker"      # disrupts defensive topology
        else:
            dominant = comp_names[np.argmax(center)]
            name = f"Cluster {c} ({dominant}-led)"

        cluster_names[c] = name
        members = eligible_clean[eligible_clean["cluster"] == c]["player"].tolist()
        print(f"    Cluster {c} ({name}): {len(members)} players, "
              f"top components: {top2}, center: {dict(zip(comp_names, center.round(3)))}")

    eligible_clean["cluster_name"] = eligible_clean["cluster"].map(cluster_names)

    # Merge cluster assignments back
    scores_df = scores_df.merge(
        eligible_clean[["player", "team", "tournament", "cluster", "cluster_name"]],
        on=["player", "team", "tournament"], how="left"
    )

    # For all k, store labels on eligible players
    cluster_info = {
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
        "cluster_names": cluster_names,
        "cluster_centers": pd.DataFrame(centers, columns=comp_names),
        "all_labels": {k: cluster_labels_map[k][1] for k in [3, 4, 5]},
    }

    # Save cluster parquet (eligible players only, with all cluster assignments)
    eligible_final = eligible_clean[
        ["player", "team", "tournament", "cluster", "cluster_name"] + cluster_cols
    ].copy()
    # Add all-k cluster assignments
    for k in [3, 4, 5]:
        km_k, labels_k = cluster_labels_map[k]
        eligible_clean_cp = eligible_clean.copy()
        eligible_clean_cp[f"cluster_k{k}"] = labels_k
        eligible_final[f"cluster_k{k}"] = labels_k

    eligible_final.to_parquet(PROCESSED / "phase2_clusters.parquet", index=False)
    print(f"\n  Saved clusters to data/processed/phase2_clusters.parquet "
          f"({len(eligible_final):,} rows)")

    return scores_df, cluster_info


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Rankings and cross-validation summary
# ══════════════════════════════════════════════════════════════════════════════

def print_rankings(scores_df, cluster_info):
    """
    Print ranked table for target players + cross-validation summary.
    """
    print("\n" + "=" * 80)
    print("ARCHITECT FRAMEWORK — PHASE 2 RANKINGS")
    print("=" * 80)

    # ── Find target player rows ───────────────────────────────────────────────
    target_mask = pd.Series([False] * len(scores_df), index=scores_df.index)
    for search in TARGET_SEARCHES:
        target_mask |= scores_df["player"].str.contains(search, na=False, case=False)

    target_df = scores_df[target_mask].copy()

    # Top 10 by Architect Score (full)
    print("\n[1] TOP 10 BY ARCHITECT SCORE (FULL) — Target Players")
    print("-" * 80)
    top10_cols = [
        "player", "team", "tournament", "matches",
        "architect_score_full", "architect_score_event", "traditional_score",
        "pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z",
    ]
    top10 = target_df.sort_values("architect_score_full", ascending=False).head(20)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(top10[top10_cols].to_string(index=False))

    # Global top 10 (all players)
    print("\n[2] GLOBAL TOP 10 BY ARCHITECT SCORE (FULL)")
    print("-" * 80)
    global_top10 = scores_df[
        scores_df["tournament"] != "Leverkusen_2324"
    ].sort_values("architect_score_full", ascending=False).head(10)
    print(global_top10[top10_cols].to_string(index=False))

    # ── Xhaka cross-validation ────────────────────────────────────────────────
    print("\n[3] XHAKA CROSS-VALIDATION: LEVERKUSEN vs TOURNAMENT")
    print("-" * 80)
    xhaka_rows = scores_df[
        scores_df["player"].str.contains("Xhaka", na=False, case=False)
    ].copy()

    cv_cols = [
        "player", "team", "tournament",
        "pacv", "ds", "dtd", "prv", "cir", "tvi",
        "pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z",
        "architect_score_full", "architect_score_event", "traditional_score",
    ]
    print(xhaka_rows[cv_cols].to_string(index=False))

    if "architect_score_full_p1" in xhaka_rows.columns:
        xhaka_lev = xhaka_rows[xhaka_rows["tournament"] == "Leverkusen_2324"]
        if len(xhaka_lev) > 0:
            p1_full = xhaka_lev["architect_score_full_p1"].values[0]
            p2_full_euro2024 = xhaka_rows[
                xhaka_rows["tournament"] == "euro2024"
            ]["architect_score_full"].values
            p2_full_euro2020 = xhaka_rows[
                xhaka_rows["tournament"] == "euro2020"
            ]["architect_score_full"].values
            print(f"\n  Xhaka Leverkusen Phase 1 AS_full (within-P1 z-score): {p1_full:.3f}")
            if len(p2_full_euro2024) > 0:
                print(f"  Xhaka Euro 2024 AS_full (within-P2 z-score): {p2_full_euro2024[0]:.3f}")
            if len(p2_full_euro2020) > 0:
                print(f"  Xhaka Euro 2020 AS_full (within-P2 z-score): {p2_full_euro2020[0]:.3f}")

    # ── Cluster assignments for target players ────────────────────────────────
    if "cluster_name" in scores_df.columns:
        print("\n[4] CLUSTER ASSIGNMENTS — Target Players")
        print("-" * 80)
        cluster_cols_print = [
            "player", "team", "tournament",
            "cluster", "cluster_name",
            "pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z",
            "architect_score_full",
        ]
        clustered_targets = target_df[target_df["cluster"].notna()].sort_values(
            "architect_score_full", ascending=False
        )
        if len(clustered_targets) > 0:
            print(clustered_targets[cluster_cols_print].to_string(index=False))
        else:
            print("  No target players met clustering eligibility criteria (100+ passes, 50+ FF)")

    # ── Silhouette scores ─────────────────────────────────────────────────────
    if cluster_info:
        print("\n[5] SILHOUETTE SCORES")
        print("-" * 80)
        for k, sil in cluster_info["silhouette_scores"].items():
            marker = " <-- OPTIMAL" if k == cluster_info["optimal_k"] else ""
            print(f"  k={k}: silhouette={sil:.3f}{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_score_assembly():
    """
    Orchestrate all sections:
    1. Compute PACV from action_values
    2. Aggregate DS and DTD per (player, team, tournament)
    3. Assemble player-level scores
    4. Add Xhaka Phase 1 cross-validation row
    5. Z-score vs full population & compute Architect Scores
    6. K-means archetype clustering
    7. Print rankings & cross-validation summary
    8. Save outputs
    """
    print("\n" + "=" * 80)
    print("PHASE 2 SCORE ASSEMBLY — THE ARCHITECT FRAMEWORK")
    print("=" * 80)

    # ── Section 1: PACV ───────────────────────────────────────────────────────
    print("\n[1/7] Computing PACV from action_values...")
    pacv_df = compute_pacv()

    # ── Section 2: DS + DTD per tournament ────────────────────────────────────
    print("\n[2/7] Aggregating DS and DTD per (player, team, tournament)...")
    ds_df, dtd_df = aggregate_ds_dtd()

    # ── Section 3: Assemble scores ────────────────────────────────────────────
    print("\n[3/7] Assembling player-level scores...")
    scores = assemble_player_scores(pacv_df, ds_df, dtd_df)

    # ── Section 4: Xhaka Phase 1 row ──────────────────────────────────────────
    print("\n[4/7] Loading Xhaka Phase 1 cross-validation row...")
    xhaka_row = load_xhaka_phase1()

    if xhaka_row is not None:
        # Align columns
        for col in scores.columns:
            if col not in xhaka_row.columns:
                xhaka_row[col] = np.nan
        for col in xhaka_row.columns:
            if col not in scores.columns:
                scores[col] = np.nan
        scores = pd.concat([scores, xhaka_row], ignore_index=True)
        print(f"  Combined scores shape (with Xhaka P1 row): {scores.shape}")

    # ── Section 5: Z-scores and Architect Scores ──────────────────────────────
    print("\n[5/7] Z-scoring and computing Architect Scores...")

    novel_cols = ["pacv", "ds", "dtd", "prv", "cir", "tvi"]
    trad_cols = ["xa", "progressive_passes", "sca"]
    novel_z = ["pacv_z", "ds_z", "dtd_z", "prv_z", "cir_z", "tvi_z"]
    event_z = ["pacv_z", "prv_z", "cir_z", "tvi_z"]
    trad_z = ["xa_z", "progressive_passes_z", "sca_z"]

    # Use only Phase 2 tournament rows for normalization (exclude Phase 1 row)
    phase2_mask = scores["tournament"] != "Leverkusen_2324"
    phase2_scores = scores[phase2_mask].copy()

    # Compute z-scores on Phase 2 population
    phase2_scores = zscore_and_architect(phase2_scores)

    # Capture population stats from Phase 2 for cross-population z-scoring of Xhaka P1
    pop_stats = {}
    for col in novel_cols + trad_cols:
        pop_stats[col] = {"mu": phase2_scores[col].mean(), "sigma": phase2_scores[col].std()}

    # Merge Phase 2 z-scores back into the full scores table
    zscore_cols = [c for c in phase2_scores.columns if c.endswith("_z") or
                   c in ["architect_score_full", "architect_score_event", "traditional_score"]]
    merge_key_cols = ["player", "team", "tournament"]

    scores = scores.drop(columns=[c for c in zscore_cols if c in scores.columns], errors="ignore")
    scores = scores.merge(
        phase2_scores[merge_key_cols + zscore_cols],
        on=merge_key_cols, how="left"
    )

    # Now fill in Xhaka Leverkusen z-scores using Phase 2 population parameters
    if xhaka_row is not None:
        lev_mask = scores["tournament"] == "Leverkusen_2324"
        if lev_mask.any():
            # Ensure z-score columns exist
            for z_col in novel_z + trad_z:
                if z_col not in scores.columns:
                    scores[z_col] = np.nan

            for col in novel_cols + trad_cols:
                z_col = f"{col}_z"
                mu = pop_stats[col]["mu"]
                sigma = pop_stats[col]["sigma"]
                raw_val = scores.loc[lev_mask, col].values[0]
                if not pd.isna(raw_val) and sigma > 0:
                    scores.loc[lev_mask, z_col] = (raw_val - mu) / sigma

            # PACV SCALE WARNING: Phase 1 PACV = total_xg_involved * pre_assist_ratio
            # (chain-position proxy). Phase 2 PACV = sum(attention_excess * terminal_xg)
            # (Transformer-based). These are on fundamentally different scales.
            # Set pacv_z for Leverkusen to NaN to avoid inflating architect scores.
            scores.loc[lev_mask, "pacv_z"] = np.nan

            # Compute Architect Scores for Leverkusen row
            # architect_score_full excludes pacv (incompatible scales, NaN handled by skipna)
            scores.loc[lev_mask, "architect_score_full"] = (
                scores.loc[lev_mask, novel_z].mean(axis=1, skipna=True)
            )
            scores.loc[lev_mask, "architect_score_event"] = (
                scores.loc[lev_mask, event_z].mean(axis=1, skipna=True)
            )
            scores.loc[lev_mask, "traditional_score"] = (
                scores.loc[lev_mask, trad_z].mean(axis=1, skipna=True)
            )
            lev_as = scores.loc[lev_mask, "architect_score_full"].values[0]
            print(f"  Xhaka Leverkusen z-scored vs Phase 2 population: AS_full={lev_as:.3f}")
            print(f"  NOTE: pacv_z set to NaN for Leverkusen row — Phase 1 PACV uses "
                  f"chain-position proxy (scale incompatible with Phase 2 Transformer-based PACV)")

    # ── Section 6: Clustering ──────────────────────────────────────────────────
    print("\n[6/7] Running K-means archetype clustering...")
    # Cluster on Phase 2 rows only (exclude Leverkusen)
    phase2_for_cluster = phase2_scores.copy()
    phase2_for_cluster_clustered, cluster_info = run_clustering(phase2_for_cluster)

    # Merge cluster results back to full scores
    if "cluster" in phase2_for_cluster_clustered.columns:
        cluster_merge_cols = ["player", "team", "tournament", "cluster", "cluster_name"]
        scores = scores.drop(columns=["cluster", "cluster_name"], errors="ignore")
        scores = scores.merge(
            phase2_for_cluster_clustered[cluster_merge_cols].dropna(subset=["cluster"]),
            on=["player", "team", "tournament"], how="left"
        )

    # ── Section 7: Rankings ────────────────────────────────────────────────────
    print("\n[7/7] Printing rankings and cross-validation summary...")
    print_rankings(scores, cluster_info)

    # ── Save outputs ──────────────────────────────────────────────────────────
    scores_path = PROCESSED / "phase2_architect_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    print(f"\n  Saved architect scores: {scores_path} ({scores.shape})")

    print("\n" + "=" * 80)
    print("PHASE 2 SCORE ASSEMBLY COMPLETE")
    print("=" * 80)
    print(f"  phase2_architect_scores.parquet: {scores.shape}")
    print(f"  phase2_clusters.parquet: (see above)")

    return scores, cluster_info


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores, cluster_info = run_score_assembly()
