"""
Phase 2 Visualizations — The Architect Framework
8 publication-quality charts
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.table import Table
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────
BG_COLOR   = '#1a1a2e'
GRID_COLOR = '#2a2a4e'
TEXT_COLOR = '#e0e0e0'
ACCENT_COLOR = '#4a9eff'

PLAYER_COLORS = {
    'Pedro González López':        '#E32636',   # Pedri
    'Rodrigo Hernández Cascante':  '#FFD700',   # Rodri
    'Vitor Machado Ferreira':      '#4169E1',   # Vitinha
    'Toni Kroos':                  '#FFFFFF',
    'Granit Xhaka':                '#E32636',
    'Frenkie de Jong':             '#FF6B35',
    'Jorge Luiz Frello Filho':     '#00CC88',   # Jorginho
    'Martín Zubimendi Ibáñez':     '#CC44FF',
    'Sergio Busquets i Burgos':    '#888888',
    'Marco Verratti':              '#44CCFF',
    'Fabián Ruiz Peña':            '#FF8C00',
    'Kalvin Phillips':             '#99FF44',
    'Jude Bellingham':             '#FF44BB',
    "N'Golo Kanté":                '#44FFDD',
    'Xavier Hernández Creus':      '#FFD700',   # Xavi
    'Andrés Iniesta Luján':        '#4169E1',   # Iniesta
    'Luka Modrić':                 '#FF6B35',
}

SHORT_NAMES = {
    'Pedro González López':        'Pedri',
    'Rodrigo Hernández Cascante':  'Rodri',
    'Vitor Machado Ferreira':      'Vitinha',
    'Toni Kroos':                  'Kroos',
    'Granit Xhaka':                'Xhaka',
    'Frenkie de Jong':             'Frenkie',
    'Jorge Luiz Frello Filho':     'Jorginho',
    'Martín Zubimendi Ibáñez':     'Zubimendi',
    'Sergio Busquets i Burgos':    'Busquets',
    'Marco Verratti':              'Verratti',
    'Fabián Ruiz Peña':            'Fabián',
    'Kalvin Phillips':             'Phillips',
    'Jude Bellingham':             'Bellingham',
    "N'Golo Kanté":                'Kanté',
    'Xavier Hernández Creus':      'Xavi',
    'Andrés Iniesta Luján':        'Iniesta',
    'Luka Modrić':                 'Modrić',
}

# Xhaka color distinction
XHAKA_EURO2020_COLOR = '#FF4444'
XHAKA_EURO2024_COLOR = '#E32636'

from pathlib import Path
_BASE = Path(__file__).parents[1]
OUTPUT_DIR = str(_BASE / 'outputs' / 'phase2')

def get_player_color(full_name):
    for key, color in PLAYER_COLORS.items():
        if key in full_name or full_name in key:
            return color
    return ACCENT_COLOR

def short_name(full_name):
    for key, sn in SHORT_NAMES.items():
        if key in full_name or full_name in key:
            return sn
    return full_name.split()[0]

def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, alpha=0.5)

def draw_radar(ax, values, labels, color, title, alpha=0.3):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = list(values) + [values[0]]

    ax.set_facecolor(BG_COLOR)
    ax.plot(angles, values_plot, 'o-', linewidth=2, color=color)
    ax.fill(angles, values_plot, alpha=alpha, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color=TEXT_COLOR)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(title, size=9, color=color, pad=10)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.4)
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=0.4)
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.set_yticklabels([])

# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────
def load_scores():
    return pd.read_parquet(_BASE / 'data/processed/phase2_architect_scores_v2.parquet')

def load_clusters():
    return pd.read_parquet(_BASE / 'data/processed/phase2_clusters_v2.parquet')

def load_xhaka_crossval():
    return pd.read_parquet(_BASE / 'data/processed/phase2_xhaka_crossval.parquet')

def load_decision_surplus():
    return pd.read_parquet(_BASE / 'data/processed/phase2_decision_surplus.parquet')

def load_historical():
    return pd.read_parquet(_BASE / 'data/processed/historical_comparison.parquet')

def load_chain_positions():
    return pd.read_parquet(_BASE / 'data/processed/phase2_chain_positions.parquet')

# Target player list with preferred tournament.
# NOTE: Pedri prefers euro2020 — his Euro 2024 entry is excluded (92 passes < 150 threshold).
TARGET_PLAYERS = [
    # (full_name, preferred_tournament, fallback_tournament)
    ('Pedro González López',       'euro2020', None),        # injured 2024, only 92 passes
    ('Toni Kroos',                 'euro2024', 'euro2020'),
    ('Marco Verratti',             'euro2020', None),
    ('Fabián Ruiz Peña',           'euro2024', 'euro2020'),
    ('Jorge Luiz Frello Filho',    'euro2024', 'euro2020'),
    ('Vitor Machado Ferreira',     'euro2024', None),
    ('Granit Xhaka',               'euro2024', 'euro2020'),
    ('Rodrigo Hernández Cascante', 'euro2024', 'euro2020'),
    ('Jude Bellingham',            'euro2024', 'euro2020'),
    ('Sergio Busquets i Burgos',   'euro2020', None),
    ('Frenkie de Jong',            'euro2020', None),
    ('Martín Zubimendi Ibáñez',    'euro2024', None),
    ("N'Golo Kanté",               'euro2024', None),
    ('Kalvin Phillips',            'euro2020', None),
    # Xhaka Euro2020 as 15th entry
    ('Granit Xhaka',               'euro2020', None),
]

Z_COLS = ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
RADAR_LABELS = ['PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI']


def get_target_rows(scores_df):
    """Return one row per target player/tournament from TARGET_PLAYERS list."""
    rows = []
    for player, pref, fallback in TARGET_PLAYERS:
        mask = (scores_df['player'] == player) & (scores_df['tournament'] == pref)
        sub = scores_df[mask]
        if sub.empty and fallback:
            mask = (scores_df['player'] == player) & (scores_df['tournament'] == fallback)
            sub = scores_df[mask]
        if not sub.empty:
            row = sub.iloc[0].copy()
            row['_tournament_used'] = pref if not scores_df[
                (scores_df['player'] == player) & (scores_df['tournament'] == pref)
            ].empty else (fallback or pref)
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────
# Chart 1: Radar Grid
# ─────────────────────────────────────────────
def chart1_radar_grid():
    print("Creating Chart 1: Radar Grid...")
    scores = load_scores()
    target_df = get_target_rows(scores)

    # Sort by architect_score_full descending
    target_df = target_df.sort_values('architect_score_full', ascending=False).reset_index(drop=True)

    n_players = len(target_df)
    n_cols = 4
    n_rows = 4  # 4×4 = 16 cells, 15 players + 1 empty

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("The Architect Framework — Player Radar Profiles",
                 color=TEXT_COLOR, fontsize=16, fontweight='bold', y=0.98)

    for idx, row_data in target_df.iterrows():
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='polar')
        player_name = row_data['player']
        tourn = row_data['tournament']
        color = get_player_color(player_name)

        # Special: distinguish Xhaka euro2020 vs euro2024
        if player_name == 'Granit Xhaka' and tourn == 'euro2020':
            color = XHAKA_EURO2020_COLOR

        values = [float(row_data[c]) if pd.notna(row_data[c]) else 0.0 for c in Z_COLS]
        values_clipped = np.clip(values, -2, 2).tolist()

        sn = short_name(player_name)
        # Add tournament suffix for Xhaka
        if player_name == 'Granit Xhaka':
            sn = f"{sn} ({tourn[-4:]})"

        score_label = f"{sn}\n{row_data['architect_score_full']:.3f}"
        draw_radar(ax, values_clipped, RADAR_LABELS, color, score_label, alpha=0.3)

    # If fewer than 16, hide last cell
    for i in range(n_players + 1, n_rows * n_cols + 1):
        ax_empty = fig.add_subplot(n_rows, n_cols, i)
        ax_empty.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUTPUT_DIR}/phase2_radar_grid.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 2: Decision Surplus KDE Comparison
# ─────────────────────────────────────────────
def chart2_ds_comparison():
    print("Creating Chart 2: DS KDE Comparison...")
    from scipy.stats import gaussian_kde

    ds_df = load_decision_surplus()

    target_names = [
        'Pedro González López',
        'Rodrigo Hernández Cascante',
        'Vitor Machado Ferreira',
        'Toni Kroos',
        'Granit Xhaka',
        'Jorge Luiz Frello Filho',
        'Marco Verratti',
        'Fabián Ruiz Peña',
        "N'Golo Kanté",
        'Sergio Busquets i Burgos',
        'Jude Bellingham',
        'Frenkie de Jong',
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    apply_dark_style(fig, ax)

    x_range = np.linspace(-0.35, 0.35, 400)

    valid_players = []
    for player in target_names:
        pdata = ds_df[ds_df['player'] == player]['decision_surplus'].dropna()
        if len(pdata) >= 100:
            valid_players.append((player, pdata))

    for player, pdata in valid_players:
        color = get_player_color(player)
        try:
            kde = gaussian_kde(pdata, bw_method=0.3)
            y = kde(x_range)
            sn = short_name(player)
            # Special: Xhaka uses euro2020 color (only one entry in DS)
            ax.plot(x_range, y, linewidth=2, color=color, label=sn, alpha=0.9)
            median_val = np.median(pdata)
            ax.axvline(median_val, color=color, linestyle='--', linewidth=1.0, alpha=0.6)
        except Exception as e:
            print(f"  KDE failed for {player}: {e}")

    ax.axvline(0, color=TEXT_COLOR, linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xlabel("Decision Surplus (actual value − mean alternative value)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Density", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Decision Surplus Distributions — Target Players\n(dashed lines = player median)",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    legend = ax.legend(
        loc='upper left', framealpha=0.15, facecolor=BG_COLOR,
        edgecolor=GRID_COLOR, fontsize=9, ncol=2
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_ds_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 3: Rankings Table
# ─────────────────────────────────────────────
def chart3_rankings_table():
    print("Creating Chart 3: Rankings Table...")
    scores = load_scores()
    target_df = get_target_rows(scores)

    # Add display name
    target_df = target_df.copy()
    target_df['display'] = target_df['player'].apply(short_name)
    for i, row in target_df.iterrows():
        if row['player'] == 'Granit Xhaka':
            target_df.at[i, 'display'] = f"Xhaka ({row['tournament'][-4:]})"

    # Architect rank (full score only; event score as tiebreaker)
    target_df = target_df.sort_values(
        ['architect_score_full', 'architect_score_event'], ascending=False, na_position='last'
    ).reset_index(drop=True)
    target_df['arch_rank'] = range(1, len(target_df) + 1)

    # Build table data (remove Trad.Rank column — not computed in v2)
    col_labels = ['Player', 'Tourn.', 'Arch.Rank', 'Arch.Score',
                  'PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI']

    table_data = []
    for _, row in target_df.iterrows():
        z_vals = []
        for c in Z_COLS:
            v = row[c]
            z_vals.append(f"{v:+.2f}" if pd.notna(v) else 'N/A')
        as_str = f"{row['architect_score_full']:.3f}" if pd.notna(row['architect_score_full']) else 'N/A'
        table_data.append([
            row['display'],
            row['tournament'].replace('euro', 'E').replace('Leverkusen_2324', 'Lev'),
            str(int(row['arch_rank'])),
            as_str,
            *z_vals
        ])

    n_rows = len(table_data)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    fig.suptitle("The Architect Framework — Player Rankings (Target Players)",
                 color=TEXT_COLOR, fontsize=15, fontweight='bold', y=0.97)

    # Column widths (10 cols: Player, Tourn., Arch.Rank, Arch.Score, 6 z-scores)
    col_widths = [0.14, 0.06, 0.08, 0.09,
                  0.072, 0.072, 0.072, 0.072, 0.072, 0.072]

    # Draw header
    header_y = 0.91
    x = 0.02
    for j, (label, w) in enumerate(zip(col_labels, col_widths)):
        ax.text(x + w / 2, header_y, label,
                ha='center', va='center', color=ACCENT_COLOR,
                fontsize=9, fontweight='bold',
                transform=ax.transAxes)
        x += w

    # Draw header underline
    ax.plot([0.02, 0.98], [header_y - 0.018, header_y - 0.018],
            color=ACCENT_COLOR, linewidth=1.2, transform=ax.transAxes, zorder=5)

    # Draw rows
    row_height = 0.054
    top_y = 0.87

    for i, row_vals in enumerate(table_data):
        y = top_y - i * row_height
        arch_rank = int(row_vals[2])

        # Row background
        if arch_rank <= 3:
            bg_color = '#2a2a00'  # gold tint
            text_c = '#FFD700'
        elif i % 2 == 0:
            bg_color = '#1f1f38'
            text_c = TEXT_COLOR
        else:
            bg_color = BG_COLOR
            text_c = TEXT_COLOR

        # Background rect
        rect = FancyBboxPatch(
            (0.02, y - row_height * 0.45),
            0.96, row_height * 0.88,
            boxstyle="round,pad=0.002",
            facecolor=bg_color, edgecolor='none',
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)

        x = 0.02
        for j, (val, w) in enumerate(zip(row_vals, col_widths)):
            # Color z-scores
            cell_color = text_c
            if j >= 5:  # z-score columns
                try:
                    fval = float(val)
                    if fval >= 1.0:
                        cell_color = '#00CC88'
                    elif fval <= -1.0:
                        cell_color = '#FF6666'
                    else:
                        cell_color = TEXT_COLOR
                except ValueError:
                    pass

            ax.text(x + w / 2, y, val,
                    ha='center', va='center', color=cell_color,
                    fontsize=8.5, fontweight='bold' if j == 0 else 'normal',
                    transform=ax.transAxes, zorder=2)
            x += w

    # Legend
    ax.text(0.02, 0.03,
            "Green z-score ≥ +1.0  |  Red z-score ≤ −1.0  |  Gold rows = Top 3 Architect Score",
            color=TEXT_COLOR, fontsize=8, transform=ax.transAxes, alpha=0.7)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_rankings_table.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 4: Cluster Radar
# ─────────────────────────────────────────────
def chart4_cluster_radar():
    print("Creating Chart 4: Cluster Radar...")
    clusters = load_clusters()

    cluster_colors_map = {
        'The Orchestrator': '#E32636',
        'The Metronome':    '#FFD700',
        'The Disruptor':    '#FF6B35',
        'The Connector':    '#4a9eff',
    }
    fallback_colors = ['#44FFDD', '#FF6B35', '#4a9eff', '#CC44FF']

    unique_clusters = clusters[['cluster', 'cluster_name']].drop_duplicates().sort_values('cluster')
    n_clusters = len(unique_clusters)

    fig = plt.figure(figsize=(5 * n_clusters, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("The Architect Framework — Archetype Cluster Profiles (k=4)",
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', y=1.01)

    for i, (_, cluster_row) in enumerate(unique_clusters.iterrows()):
        cluster_id = cluster_row['cluster']
        cluster_name = cluster_row['cluster_name']
        subset = clusters[clusters['cluster'] == cluster_id]
        n = len(subset)

        center = [subset[c].mean() if c in subset.columns else 0.0 for c in Z_COLS]
        center_clipped = np.clip(center, -2, 2).tolist()

        color = cluster_colors_map.get(cluster_name, fallback_colors[i % len(fallback_colors)])

        ax = fig.add_subplot(1, n_clusters, i + 1, projection='polar')
        title_str = f"{cluster_name}\n(n={n})"
        draw_radar(ax, center_clipped, RADAR_LABELS, color, title_str, alpha=0.35)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_cluster_radar.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 5: Xhaka Cross-Validation
# ─────────────────────────────────────────────
def chart5_xhaka_crossval():
    print("Creating Chart 5: Xhaka Cross-Validation...")
    scores = load_scores()
    xv = load_xhaka_crossval()

    xhaka_e24 = scores[(scores['player'] == 'Granit Xhaka') & (scores['tournament'] == 'euro2024')]

    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Granit Xhaka — Cross-Context Validation\nLeverkusen 2023/24 (Phase 1) vs Switzerland Euro 2024 (Phase 2)",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', y=1.02)

    # Left panel: Leverkusen radar using Phase 1 z-scores
    ax1 = fig.add_subplot(1, 3, 1, projection='polar')
    if not xv.empty:
        lev_row = xv.iloc[0]
        p1_z_cols = ['pacv_z_p1', 'ds_z_p1', 'dtd_z_p1', 'prv_z_p1', 'cir_z_p1', 'tvi_z_p1']
        lev_vals = np.clip(
            [float(lev_row.get(c, 0.0)) if pd.notna(lev_row.get(c, np.nan)) else 0.0
             for c in p1_z_cols],
            -2, 2
        ).tolist()
        as_p1 = float(lev_row.get('architect_score_full_p1', np.nan))
        as_label = f"{as_p1:.3f}" if not np.isnan(as_p1) else "N/A"
        draw_radar(ax1, lev_vals, RADAR_LABELS, '#00CC88',
                   f"Leverkusen 2023/24\nAS(P1)={as_label}", alpha=0.35)

    # Middle panel: Euro 2024 radar using Phase 2 z-scores
    ax2 = fig.add_subplot(1, 3, 2, projection='polar')
    if not xhaka_e24.empty:
        row = xhaka_e24.iloc[0]
        e24_vals = np.clip(
            [float(row[c]) if pd.notna(row[c]) else 0.0 for c in Z_COLS],
            -2, 2
        ).tolist()
        draw_radar(ax2, e24_vals, RADAR_LABELS, XHAKA_EURO2024_COLOR,
                   f"Euro 2024 (Switzerland)\nAS(P2)={row['architect_score_full']:.3f}", alpha=0.35)

    # Right panel: side-by-side z-score comparison
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(BG_COLOR)
    ax3.axis('off')

    def _safe_float(v):
        try:
            f = float(v)
            return f if not np.isnan(f) else None
        except Exception:
            return None

    lines = [
        ("Cross-Validation Summary", '#FFD700', 12, True),
        ("", TEXT_COLOR, 9, False),
        ("Population z-scores are not directly", '#aaaaaa', 8, False),
        ("comparable (different reference pools).", '#aaaaaa', 8, False),
        ("Use as directional consistency check.", '#aaaaaa', 8, False),
        ("", TEXT_COLOR, 8, False),
        ("Metric     Lev'24   E2024", TEXT_COLOR, 9, True),
        ("(P1 pop)  (P2 mid)", '#888888', 8, False),
        ("", TEXT_COLOR, 8, False),
    ]

    metric_pairs = [
        ('DS',   'ds_z_p1',  'ds_z'),
        ('DTD',  'dtd_z_p1', 'dtd_z'),
        ('PRV',  'prv_z_p1', 'prv_z'),
        ('CIR',  'cir_z_p1', 'cir_z'),
        ('TVI',  'tvi_z_p1', 'tvi_z'),
    ]

    if not xv.empty and not xhaka_e24.empty:
        lev_row = xv.iloc[0]
        e24_row = xhaka_e24.iloc[0]
        for label, p1_col, p2_col in metric_pairs:
            p1_val = _safe_float(lev_row.get(p1_col))
            p2_val = _safe_float(e24_row.get(p2_col))
            p1_str = f"{p1_val:+.2f}" if p1_val is not None else " N/A "
            p2_str = f"{p2_val:+.2f}" if p2_val is not None else " N/A "
            # color by sign agreement
            if p1_val is not None and p2_val is not None:
                agree = (p1_val > 0) == (p2_val > 0)
                col = '#00CC88' if agree else '#FF8C00'
            else:
                col = TEXT_COLOR
            lines.append((f"{label:6s}  {p1_str}   {p2_str}", col, 9, False))

        as_p1 = _safe_float(lev_row.get('architect_score_full_p1'))
        as_p2 = _safe_float(e24_row.get('architect_score_full'))
        lines += [
            ("", TEXT_COLOR, 9, False),
            (f"AS     {(as_p1 or 0):+.3f}  {(as_p2 or 0):+.3f}", TEXT_COLOR, 9, True),
            ("", TEXT_COLOR, 8, False),
            ("Green = sign agreement across", '#00CC88', 8, False),
            ("contexts (good consistency)", '#00CC88', 8, False),
        ]

    y = 0.94
    for text, color, size, bold in lines:
        ax3.text(0.04, y, text, color=color, fontsize=size,
                 fontweight='bold' if bold else 'normal',
                 transform=ax3.transAxes, va='top',
                 fontfamily='monospace' if '  ' in text else None)
        y -= 0.072 if size >= 11 else 0.062

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_xhaka_crossval.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 6: Historical Timeline
# ─────────────────────────────────────────────
def chart6_historical_timeline():
    print("Creating Chart 6: Historical Timeline...")
    hist = pd.read_parquet(_BASE / 'data/processed/historical_comparison.parquet')

    # Historical players
    hist_players = [
        'Xavier Hernández Creus',
        'Andrés Iniesta Luján',
        'Sergio Busquets i Burgos',
        'Luka Modrić',
    ]
    hist_colors = {
        'Xavier Hernández Creus':  PLAYER_COLORS['Xavier Hernández Creus'],
        'Andrés Iniesta Luján':    PLAYER_COLORS['Andrés Iniesta Luján'],
        'Sergio Busquets i Burgos': PLAYER_COLORS['Sergio Busquets i Burgos'],
        'Luka Modrić':             PLAYER_COLORS['Luka Modrić'],
    }

    # Get Xhaka Leverkusen data point from Phase 1 cross-val file
    xv = load_xhaka_crossval()
    xhaka_lev = xv if not xv.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(16, 8))
    apply_dark_style(fig, ax)

    # Get all seasons
    all_seasons = sorted(hist[hist['player'].isin(hist_players)]['season_name'].unique())

    for player in hist_players:
        pdata = hist[hist['player'] == player].copy()
        pdata = pdata.sort_values('season_name')
        color = hist_colors[player]
        sn = short_name(player)

        ax.plot(pdata['season_name'], pdata['architect_score_event'],
                'o-', color=color, linewidth=2, markersize=6,
                label=sn, alpha=0.9, zorder=3)

        # Label last point
        if not pdata.empty:
            last = pdata.iloc[-1]
            ax.annotate(f"{sn} ({last['architect_score_event']:.2f})",
                        xy=(last['season_name'], last['architect_score_event']),
                        xytext=(8, 0), textcoords='offset points',
                        color=color, fontsize=8, va='center')

    # Xhaka reference line — use Phase 1 AS_event z-score
    if not xhaka_lev.empty:
        as_col = 'architect_score_event_p1' if 'architect_score_event_p1' in xhaka_lev.columns else 'AS_event'
        xhaka_as_raw = xhaka_lev.iloc[0].get(as_col, None)
        xhaka_as = float(xhaka_as_raw) if xhaka_as_raw is not None and not pd.isna(xhaka_as_raw) else None
        if xhaka_as is not None:
            ax.axhline(y=xhaka_as, color='#E32636', linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f"Xhaka Lev23/24 ({xhaka_as:.2f})")
            ax.annotate(f"Xhaka Lev 23/24\n({xhaka_as:.2f})",
                        xy=(all_seasons[-1] if all_seasons else '2021/2022', xhaka_as),
                        xytext=(-90, 12), textcoords='offset points',
                        color='#E32636', fontsize=8, alpha=0.9,
                        arrowprops=dict(arrowstyle='->', color='#E32636', alpha=0.5))

    ax.set_xlabel("Season", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Architect Score (event-based)", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Historical Architect Scores — Elite Midfielders Over Time",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    plt.xticks(rotation=45, ha='right', color=TEXT_COLOR, fontsize=8)
    ax.set_xlim(-0.5, len(all_seasons) - 0.5 if all_seasons else 10)

    legend = ax.legend(loc='upper right', framealpha=0.15,
                       facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_historical_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 7: Chain Positions Grid
# ─────────────────────────────────────────────
def chart7_chain_positions_grid():
    print("Creating Chart 7: Chain Positions Grid...")
    chain = load_chain_positions()
    scores = load_scores()

    # Get target players list (unique names)
    target_names = [
        'Pedro González López',
        'Rodrigo Hernández Cascante',
        'Vitor Machado Ferreira',
        'Toni Kroos',
        'Granit Xhaka',
        'Frenkie de Jong',
        'Jorge Luiz Frello Filho',
        'Martín Zubimendi Ibáñez',
        'Sergio Busquets i Burgos',
        'Marco Verratti',
        'Fabián Ruiz Peña',
        'Kalvin Phillips',
        'Jude Bellingham',
        "N'Golo Kanté",
    ]

    # Aggregate chain positions per player (sum across tournaments)
    chain_agg = chain[chain['player'].isin(target_names)].groupby('player').agg({
        'pre_assist_ratio': 'mean',
        'buildup_ratio': 'mean',
        'final_ratio': 'mean',
        'total_chain_actions': 'sum',
    }).reset_index()

    # Compute "other" ratio if pre+buildup+final don't sum to 1
    chain_agg['other_ratio'] = (1.0 - chain_agg['pre_assist_ratio']
                                - chain_agg['buildup_ratio']
                                - chain_agg['final_ratio']).clip(lower=0)

    # Sort by pre_assist_ratio descending
    chain_agg = chain_agg.sort_values('pre_assist_ratio', ascending=False).reset_index(drop=True)

    # Short names
    chain_agg['display'] = chain_agg['player'].apply(short_name)

    fig, ax = plt.subplots(figsize=(16, 8))
    apply_dark_style(fig, ax)

    x = np.arange(len(chain_agg))
    bar_width = 0.65

    colors_stack = {
        'pre_assist_ratio': '#FF8C00',
        'buildup_ratio':    '#4a9eff',
        'final_ratio':      '#00CC88',
        'other_ratio':      '#555577',
    }
    labels_stack = {
        'pre_assist_ratio': 'Pre-Assist',
        'buildup_ratio':    'Build-up',
        'final_ratio':      'Final (Shot/Assist)',
        'other_ratio':      'Other',
    }

    bottom = np.zeros(len(chain_agg))
    for col in ['pre_assist_ratio', 'buildup_ratio', 'final_ratio', 'other_ratio']:
        vals = chain_agg[col].fillna(0).values
        bars = ax.bar(x, vals, bar_width, bottom=bottom,
                      color=colors_stack[col], label=labels_stack[col],
                      edgecolor=BG_COLOR, linewidth=0.5)
        bottom += vals

    # Player labels
    ax.set_xticks(x)
    ax.set_xticklabels(chain_agg['display'].tolist(), rotation=45, ha='right',
                       color=TEXT_COLOR, fontsize=9)

    ax.set_ylabel("Proportion of Chain Actions", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Chain Position Distributions — Target Players\n(ordered by pre-assist ratio)",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)

    legend = ax.legend(loc='upper right', framealpha=0.2,
                       facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_chain_positions_grid.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Chart 8: Pedri vs Rodri Head-to-Head
# ─────────────────────────────────────────────
def chart8_head_to_head():
    print("Creating Chart 8: Head-to-Head Pedri vs Rodri (Euro 2020)...")
    from scipy.stats import gaussian_kde

    scores = load_scores()
    ds_df = load_decision_surplus()
    chain = load_chain_positions()

    # Primary comparison: Euro 2020 (same team, same tournament = controlled)
    # Pedri Euro 2024 is excluded from pool (only 92 passes due to injury)
    pedri_e20 = scores[(scores['player'] == 'Pedro González López') & (scores['tournament'] == 'euro2020')]
    rodri_e20 = scores[(scores['player'] == 'Rodrigo Hernández Cascante') & (scores['tournament'] == 'euro2020')]
    rodri_e24 = scores[(scores['player'] == 'Rodrigo Hernández Cascante') & (scores['tournament'] == 'euro2024')]

    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Pedri vs Rodri — Spain Euro 2020 (same team, controlled comparison)\n"
                 "Note: Pedri's Euro 2024 excluded — 92 passes below 150-pass threshold",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', y=1.02)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: Overlapping radars ──────────────────
    ax_radar = fig.add_subplot(gs[0], projection='polar')

    def get_z_vals(df):
        if df.empty:
            return [0.0] * 6
        row = df.iloc[0]
        return np.clip([float(row[c]) if pd.notna(row[c]) else 0.0 for c in Z_COLS], -2, 2).tolist()

    def plot_radar_overlay(ax, values, color, style='-', lw=2, alpha=0.25, label=''):
        N = 6
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vp = list(values) + [values[0]]
        ax.plot(angles, vp, style, linewidth=lw, color=color, label=label, alpha=min(alpha * 3.5, 1.0))
        ax.fill(angles, vp, alpha=alpha, color=color)

    ax_radar.set_facecolor(BG_COLOR)
    ax_radar.set_ylim(-2.5, 2.5)

    pedri_color = '#E32636'
    rodri_color = '#FFD700'

    # Primary: Euro 2020 (solid lines)
    if not pedri_e20.empty:
        plot_radar_overlay(ax_radar, get_z_vals(pedri_e20), pedri_color,
                           style='-', lw=2.5, alpha=0.3, label='Pedri E2020')
    if not rodri_e20.empty:
        plot_radar_overlay(ax_radar, get_z_vals(rodri_e20), rodri_color,
                           style='-', lw=2.5, alpha=0.3, label='Rodri E2020')
    # Secondary: Rodri E2024 (dashed, for comparison)
    if not rodri_e24.empty:
        plot_radar_overlay(ax_radar, get_z_vals(rodri_e24), rodri_color,
                           style='--', lw=1.5, alpha=0.12, label='Rodri E2024')

    ax_radar.set_xticks([n / 6.0 * 2 * np.pi for n in range(6)])
    ax_radar.set_xticklabels(RADAR_LABELS, size=8, color=TEXT_COLOR)
    ax_radar.set_yticklabels([])
    ax_radar.yaxis.grid(True, color=GRID_COLOR, alpha=0.4)
    ax_radar.xaxis.grid(True, color=GRID_COLOR, alpha=0.4)
    ax_radar.spines['polar'].set_color(GRID_COLOR)
    ax_radar.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax_radar.set_title("Radar Profile\n(solid=E2020, dashed=Rodri E2024)",
                       color=TEXT_COLOR, fontsize=10, pad=12)

    leg1 = ax_radar.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.1),
                            framealpha=0.2, facecolor=BG_COLOR,
                            edgecolor=GRID_COLOR, fontsize=8)
    for t in leg1.get_texts():
        t.set_color(TEXT_COLOR)

    # ── Panel 2: DS KDE ──────────────────────────────
    ax_ds = fig.add_subplot(gs[1])
    apply_dark_style(fig, ax_ds)

    x_range = np.linspace(-0.35, 0.35, 400)
    for player, color, label in [
        ('Pedro González López', pedri_color, 'Pedri'),
        ('Rodrigo Hernández Cascante', rodri_color, 'Rodri'),
    ]:
        pdata = ds_df[ds_df['player'] == player]['decision_surplus'].dropna()
        if len(pdata) >= 30:
            kde = gaussian_kde(pdata, bw_method=0.3)
            y = kde(x_range)
            ax_ds.plot(x_range, y, linewidth=2.5, color=color, label=label)
            ax_ds.axvline(np.median(pdata), color=color, linestyle='--',
                          linewidth=1.5, alpha=0.7)
            ax_ds.fill_between(x_range, y, alpha=0.12, color=color)

    ax_ds.axvline(0, color=TEXT_COLOR, linestyle='-', linewidth=0.7, alpha=0.35)
    ax_ds.set_xlabel("Decision Surplus", color=TEXT_COLOR, fontsize=10)
    ax_ds.set_ylabel("Density", color=TEXT_COLOR, fontsize=10)
    ax_ds.set_title("Decision Surplus Distribution\n(dashed = median)", color=TEXT_COLOR, fontsize=10)
    ax_ds.set_xlim(-0.35, 0.35)
    leg2 = ax_ds.legend(framealpha=0.2, facecolor=BG_COLOR,
                         edgecolor=GRID_COLOR, fontsize=9)
    for t in leg2.get_texts():
        t.set_color(TEXT_COLOR)

    # ── Panel 3: Chain Positions ──────────────────────
    ax_chain = fig.add_subplot(gs[2])
    apply_dark_style(fig, ax_chain)

    chain_target = chain[chain['player'].isin([
        'Pedro González López', 'Rodrigo Hernández Cascante'
    ])]
    chain_agg = chain_target.groupby('player').agg({
        'pre_assist_ratio': 'mean',
        'buildup_ratio': 'mean',
        'final_ratio': 'mean',
    }).reset_index()
    chain_agg['other'] = (1.0 - chain_agg['pre_assist_ratio']
                          - chain_agg['buildup_ratio']
                          - chain_agg['final_ratio']).clip(lower=0)
    chain_agg['display'] = chain_agg['player'].apply(short_name)

    x = np.arange(len(chain_agg))
    bar_colors = ['#FF8C00', '#4a9eff', '#00CC88', '#555577']
    ratio_cols = ['pre_assist_ratio', 'buildup_ratio', 'final_ratio', 'other']
    ratio_labels = ['Pre-Assist', 'Build-up', 'Final', 'Other']
    bottom = np.zeros(len(chain_agg))

    for col, col_color, col_label in zip(ratio_cols, bar_colors, ratio_labels):
        vals = chain_agg[col].fillna(0).values
        ax_chain.bar(x, vals, 0.5, bottom=bottom,
                     color=col_color, label=col_label,
                     edgecolor=BG_COLOR, linewidth=0.5)
        bottom += vals

    ax_chain.set_xticks(x)
    ax_chain.set_xticklabels(chain_agg['display'].tolist(), color=TEXT_COLOR, fontsize=12)
    ax_chain.set_ylim(0, 1.05)
    ax_chain.set_ylabel("Proportion", color=TEXT_COLOR, fontsize=10)
    ax_chain.set_title("Chain Position Distribution", color=TEXT_COLOR, fontsize=10)

    leg3 = ax_chain.legend(loc='upper right', framealpha=0.2,
                            facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=8)
    for t in leg3.get_texts():
        t.set_color(TEXT_COLOR)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/phase2_head_to_head.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
def create_all_visualizations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    chart1_radar_grid()
    chart2_ds_comparison()
    chart3_rankings_table()
    chart4_cluster_radar()
    chart5_xhaka_crossval()
    chart6_historical_timeline()
    chart7_chain_positions_grid()
    chart8_head_to_head()

    print("=" * 60)
    print("All 8 charts created successfully.")

    # Report file sizes
    for fname in [
        'phase2_radar_grid.png',
        'phase2_ds_comparison.png',
        'phase2_rankings_table.png',
        'phase2_cluster_radar.png',
        'phase2_xhaka_crossval.png',
        'phase2_historical_timeline.png',
        'phase2_chain_positions_grid.png',
        'phase2_head_to_head.png',
    ]:
        fpath = f"{OUTPUT_DIR}/{fname}"
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname}: {size_kb:.1f} KB")
        else:
            print(f"  {fname}: MISSING")


if __name__ == '__main__':
    create_all_visualizations()
