"""
The Architect Framework — Interactive Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# ─── CONFIG ───
st.set_page_config(
    page_title="The Architect Framework",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

XHAKA_COLOR = '#E32636'
WIRTZ_COLOR = '#FFD700'
ANDRICH_COLOR = '#4169E1'
BG_COLOR = '#1a1a2e'
PLAYER_COLORS = {
    'Granit Xhaka': XHAKA_COLOR,
    'Florian Wirtz': WIRTZ_COLOR,
    'Robert Andrich': ANDRICH_COLOR,
}

DATA_DIR = Path("data/processed")


@st.cache_data
def load_data():
    scores = pd.read_parquet(DATA_DIR / "architect_scores_final.parquet")
    ds_df = pd.read_parquet(DATA_DIR / "decision_surplus.parquet")
    dtd_df = pd.read_parquet(DATA_DIR / "defensive_disruption.parquet")
    chain_pos = pd.read_parquet(DATA_DIR / "chain_positions.parquet")
    action_vals = pd.read_parquet(DATA_DIR / "action_values.parquet")
    return scores, ds_df, dtd_df, chain_pos, action_vals


@st.cache_data
def load_phase2_data():
    try:
        p2_scores = pd.read_parquet(DATA_DIR / "phase2_architect_scores_v2.parquet")
        p2_clusters = pd.read_parquet(DATA_DIR / "phase2_clusters_v2.parquet")
        p2_ds = pd.read_parquet(DATA_DIR / "phase2_decision_surplus.parquet")
        historical = pd.read_parquet(DATA_DIR / "historical_comparison.parquet")
        chain_pos = pd.read_parquet(DATA_DIR / "phase2_chain_positions.parquet")
        return p2_scores, p2_clusters, p2_ds, historical, chain_pos
    except FileNotFoundError:
        return None


scores, ds_df, dtd_df, chain_pos, action_vals = load_data()

# ─── SIDEBAR ───
st.sidebar.title("⚽ The Architect Framework")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Player Profiles",
    "Head-to-Head",
    "Xhaka Deep Dive",
    "Methodology",
    "── Phase 2 ──",
    "Cross-Tournament Comparison",
    "Archetype Analysis",
    "Historical Context",
    "Xhaka Cross-Validation",
])


# ─── PHASE 2 CONSTANTS ───
P2_PLAYER_COLORS = {
    'Pedro González López': '#E32636',
    'Rodrigo Hernández Cascante': '#FFD700',
    'Vitor Machado Ferreira': '#4169E1',
    'Toni Kroos': '#FFFFFF',
    'Granit Xhaka': '#E32636',
    'Frenkie de Jong': '#FF6B35',
    'Jorge Luiz Frello Filho': '#00CC88',
    'Martín Zubimendi Ibáñez': '#CC44FF',
    'Sergio Busquets i Burgos': '#888888',
    'Marco Verratti': '#44CCFF',
    'Fabián Ruiz Peña': '#FF8C00',
    'Kalvin Phillips': '#99FF44',
    'Jude Bellingham': '#FF44BB',
    "N'Golo Kanté": '#44FFDD',
}

SHORT_NAMES = {
    'Pedro González López': 'Pedri',
    'Rodrigo Hernández Cascante': 'Rodri',
    'Vitor Machado Ferreira': 'Vitinha',
    'Toni Kroos': 'Kroos',
    'Granit Xhaka': 'Xhaka',
    'Frenkie de Jong': 'Frenkie',
    'Jorge Luiz Frello Filho': 'Jorginho',
    'Martín Zubimendi Ibáñez': 'Zubimendi',
    'Sergio Busquets i Burgos': 'Busquets',
    'Marco Verratti': 'Verratti',
    'Fabián Ruiz Peña': 'Fabián',
    'Kalvin Phillips': 'Phillips',
    'Jude Bellingham': 'Bellingham',
    "N'Golo Kanté": 'Kanté',
    'Xavier Hernández Creus': 'Xavi',
    'Andrés Iniesta Luján': 'Iniesta',
    'Luka Modrić': 'Modrić',
}


def short_name(full_name):
    for k, v in SHORT_NAMES.items():
        if k in full_name or full_name in k:
            return v
    return full_name.split()[0]


def p2_player_color(full_name):
    for k, c in P2_PLAYER_COLORS.items():
        if k in full_name or full_name in k:
            return c
    return '#88ccff'


def make_p2_radar(ax, values, labels, color, title, alpha=0.3):
    """Draw a radar chart on a polar axes."""
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = [float(v) if pd.notna(v) else 0.0 for v in values]
    vals_plot = vals + [vals[0]]
    ax.set_facecolor(BG_COLOR)
    ax.plot(angles, vals_plot, 'o-', linewidth=2, color=color)
    ax.fill(angles, vals_plot, alpha=alpha, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color='white')
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(title, size=9, color=color, pad=10)
    ax.set_yticklabels([])
    ax.grid(color='#2a2a4e', alpha=0.5)
    ax.spines['polar'].set_color('#2a2a4e')
    ax.tick_params(colors='white', labelsize=7)


# ─── HELPER FUNCTIONS ───
def make_radar(players, scores_df, title=""):
    categories = ['PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI']
    z_cols = ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for name in players:
        row = scores_df[scores_df['player'] == name]
        if len(row) == 0:
            continue
        vals = [row.iloc[0][c] for c in z_cols] + [row.iloc[0][z_cols[0]]]
        color = PLAYER_COLORS.get(name, '#88ccff')
        ax.plot(angles, vals, color=color, linewidth=2.5, label=name)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(-2, 3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.grid(color='gray', alpha=0.3)
    ax.tick_params(colors='white')
    return fig


def make_ds_distribution(players):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for name in players:
        data = ds_df[ds_df['player'] == name]['decision_surplus'].clip(-0.3, 0.4)
        color = PLAYER_COLORS.get(name, '#88ccff')
        med = ds_df[ds_df['player'] == name]['decision_surplus'].median()
        ax.hist(data, bins=50, alpha=0.4, color=color, label=f"{name} (med: {med:.3f})", density=True)
        ax.axvline(med, color=color, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Decision Surplus', fontsize=12, color='white')
    ax.set_ylabel('Density', fontsize=12, color='white')
    ax.legend(fontsize=10, framealpha=0.8)
    ax.tick_params(colors='white')
    return fig


# ════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("The Architect Framework")
    st.markdown("### Quantifying the Hidden Value of the Regista")
    st.markdown("""
    > *"How can we measure the full creative influence of midfielders whose primary contribution 
    > to attack occurs before the final pass — and do existing analytics frameworks 
    > systematically undervalue these players?"*
    """)

    st.divider()

    # Key metrics cards
    xhaka = scores[scores['player'] == 'Granit Xhaka'].iloc[0]
    wirtz = scores[scores['player'] == 'Florian Wirtz'].iloc[0]
    andrich = scores[scores['player'] == 'Robert Andrich'].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Xhaka Architect Score", f"{xhaka['AS_full']:.3f}", f"Rank {xhaka['AS_rank']:.0f}")
    col2.metric("Xhaka Traditional Rank", f"{xhaka['trad_rank']:.0f}", f"Δ {xhaka['rank_change']:.0f}")
    col3.metric("Decision Surplus (p-val)", "4.34e-06", "Xhaka > Andrich")
    col4.metric("Team xG Without Xhaka", "1.70", "-0.44 vs with him")

    st.divider()

    # Radar chart
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Player Profiles")
        fig = make_radar(['Granit Xhaka', 'Florian Wirtz', 'Robert Andrich'], scores)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Ranking Table")
        display_cols = ['player', 'AS_full', 'pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z', 'trad_score', 'rank_change']
        display = scores.nlargest(15, 'AS_full')[display_cols].copy()
        display.columns = ['Player', 'AS', 'PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI', 'Traditional', 'Rank Δ']
        st.dataframe(display.style.format({
            'AS': '{:.2f}', 'PACV': '{:.2f}', 'DS': '{:.2f}', 'DTD': '{:.2f}',
            'PRV': '{:.2f}', 'CIR': '{:.2f}', 'TVI': '{:.2f}',
            'Traditional': '{:.2f}', 'Rank Δ': '{:+.0f}'
        }), use_container_width=True, hide_index=True)

    # Key findings
    st.divider()
    st.subheader("Key Findings")
    st.markdown("""
    - **Xhaka's total chain value (6.09) nearly equals Wirtz's (6.50)** — but 80% of Xhaka's comes from deep buildup, while Wirtz concentrates near the final action.
    - **Decision Surplus statistically separates Xhaka from Andrich** (p = 4.34e-06). Xhaka consistently finds passes that are more valuable than the alternatives available to him.
    - **DS is completely independent of progressive passing** (r = -0.040), proving it captures a fundamentally different dimension of creative value.
    - **Andrich drops from rank 8 (traditional) to rank 15 (Architect Score)** — the framework correctly identifies that despite playing the same position as Xhaka, he doesn't orchestrate.
    - **Leverkusen created 2.14 xG/match with Xhaka vs 1.70 without** — a 26% drop in the one match he missed.
    """)


# ════════════════════════════════════════════════════════════════
# PAGE 2: PLAYER PROFILES
# ════════════════════════════════════════════════════════════════
elif page == "Player Profiles":
    st.title("Player Profiles")

    player_list = scores.sort_values('AS_full', ascending=False)['player'].tolist()
    selected = st.selectbox("Select a player", player_list)

    row = scores[scores['player'] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Architect Score", f"{row['AS_full']:.3f}", f"Rank {row['AS_rank']:.0f}")
    col2.metric("Traditional Score", f"{row['trad_score']:.3f}", f"Rank {row['trad_rank']:.0f}")
    col3.metric("Rank Change", f"{row['rank_change']:+.0f}",
                "Undervalued" if row['rank_change'] > 0 else "Overvalued" if row['rank_change'] < 0 else "Fair")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Architect Score Radar")
        fig = make_radar([selected], scores, title=selected)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Sub-Component Breakdown")
        components = {
            'PACV (Pre-Assist Chain Value)': row['pacv_z'],
            'DS (Decision Surplus)': row['ds_z'],
            'DTD (Defensive Disruption)': row['dtd_z'],
            'PRV (Press Resistance)': row['prv_z'],
            'CIR (Chain Initiation)': row['cir_z'],
            'TVI (Tempo Variance)': row['tvi_z'],
        }
        comp_df = pd.DataFrame({
            'Metric': components.keys(),
            'Z-Score': components.values()
        })
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        colors = [XHAKA_COLOR if v > 0 else ANDRICH_COLOR for v in comp_df['Z-Score']]
        ax.barh(comp_df['Metric'], comp_df['Z-Score'], color=colors, alpha=0.8)
        ax.axvline(0, color='white', linewidth=0.5)
        ax.set_xlabel('Z-Score', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close()

    # DS detail if available
    player_ds = ds_df[ds_df['player'] == selected]
    if len(player_ds) > 0:
        st.subheader("Decision Surplus Detail")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median DS", f"{player_ds['decision_surplus'].median():.4f}")
        col2.metric("% Positive", f"{(player_ds['decision_surplus'] > 0).mean()*100:.1f}%")
        col3.metric("DS Under Pressure", f"{player_ds[player_ds['is_under_pressure']]['decision_surplus'].median():.4f}")
        col4.metric("Passes Evaluated", f"{len(player_ds)}")


# ════════════════════════════════════════════════════════════════
# PAGE 3: HEAD-TO-HEAD
# ════════════════════════════════════════════════════════════════
elif page == "Head-to-Head":
    st.title("Head-to-Head Comparison")

    player_list = scores.sort_values('AS_full', ascending=False)['player'].tolist()
    col1, col2 = st.columns(2)
    p1 = col1.selectbox("Player 1", player_list, index=player_list.index('Granit Xhaka'))
    p2 = col2.selectbox("Player 2", player_list, index=player_list.index('Robert Andrich'))

    add_third = st.checkbox("Add third player")
    p3 = None
    if add_third:
        p3 = st.selectbox("Player 3", player_list, index=player_list.index('Florian Wirtz'))

    players = [p1, p2] + ([p3] if p3 else [])

    # Assign colors
    for i, p in enumerate(players):
        if p not in PLAYER_COLORS:
            PLAYER_COLORS[p] = ['#E32636', '#4169E1', '#FFD700', '#00cc88'][i % 4]

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Radar Comparison")
        fig = make_radar(players, scores)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("DS Distribution Comparison")
        valid = [p for p in players if len(ds_df[ds_df['player'] == p]) > 0]
        if valid:
            fig = make_ds_distribution(valid)
            st.pyplot(fig)
            plt.close()

    # Comparison table
    st.subheader("Full Metric Comparison")
    metrics = ['AS_full', 'pacv', 'ds', 'dtd', 'prv', 'cir', 'tvi', 'xa', 'progressive_passes', 'sca']
    labels = ['Architect Score', 'PACV', 'Decision Surplus', 'Def. Disruption',
              'Press Resistance', 'Chain Initiation', 'Tempo Variance',
              'xA', 'Progressive Passes', 'SCA']
    rows = []
    for m, l in zip(metrics, labels):
        row_data = {'Metric': l}
        for p in players:
            r = scores[scores['player'] == p]
            row_data[p] = r.iloc[0][m] if len(r) > 0 else np.nan
        rows.append(row_data)
    comp_table = pd.DataFrame(rows)
    st.dataframe(comp_table.style.format({p: '{:.4f}' for p in players}),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4: XHAKA DEEP DIVE
# ════════════════════════════════════════════════════════════════
elif page == "Xhaka Deep Dive":
    st.title("Xhaka Deep Dive — Bayer Leverkusen 2023/24")
    st.markdown("*The most celebrated regista season in recent European football, analyzed through the Architect Framework.*")

    # Key stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches", "33")
    col2.metric("Passes", "3,299")
    col3.metric("Completion", "92.3%")
    col4.metric("Progressive", "886")
    col5.metric("Chain Value", "6.09")

    st.divider()

    # Pass map
    st.subheader("Pass Map — Colored by Decision Surplus")
    try:
        from PIL import Image
        img = Image.open("outputs/xhaka_ds_passmap.png")
        st.image(img, use_container_width=True)
    except:
        st.info("Pass map image not found. Run the visualization pipeline first.")

    st.divider()

    # Chain position analysis
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Where Xhaka Contributes in Shot Chains")
        try:
            from PIL import Image
            img = Image.open("outputs/chain_positions.png")
            st.image(img, use_container_width=True)
        except:
            st.info("Chain position chart not found.")

    with col_right:
        st.subheader("Transformer Credit Assignment")
        xhaka_vals = action_vals[(action_vals['player'].str.contains('Xhaka')) & (action_vals['ended_in_shot'])]
        pos_attn = xhaka_vals.groupby('position_from_end')['attention_weight'].mean()

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        positions = range(min(15, len(pos_attn)))
        labels = ['Shot', 'Assist', 'Pre-2', 'Pre-3', 'Pre-4', 'Pre-5'] + [f'Pos {i}' for i in range(6, 15)]
        vals = [pos_attn.get(i, 0) for i in positions]
        colors = [XHAKA_COLOR if i >= 2 else '#888888' for i in positions]
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels[:len(vals)], rotation=45, ha='right', fontsize=9, color='white')
        ax.set_ylabel('Mean Attention Weight', color='white', fontsize=11)
        ax.set_title("Xhaka's Attention Weight by Chain Position", color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Split-half and per-match analysis
    st.subheader("Consistency Analysis")

    xhaka_ds = ds_df[ds_df['player'].str.contains('Xhaka')]
    match_ds = xhaka_ds.groupby('match_id')['decision_surplus'].median().reset_index()
    match_ds.columns = ['match_id', 'median_ds']
    match_ds['match_num'] = range(1, len(match_ds) + 1)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.bar(match_ds['match_num'], match_ds['median_ds'],
           color=[XHAKA_COLOR if v > 0 else ANDRICH_COLOR for v in match_ds['median_ds']], alpha=0.8)
    ax.axhline(match_ds['median_ds'].mean(), color='white', linestyle='--', alpha=0.5, label=f"Mean: {match_ds['median_ds'].mean():.4f}")
    ax.set_xlabel('Match Number', color='white')
    ax.set_ylabel('Median Decision Surplus', color='white')
    ax.set_title("Xhaka's Decision Surplus by Match — Consistency Check", color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.8)
    ax.tick_params(colors='white')
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""
    **Split-half reliability:** Odd matches median DS = {xhaka_ds[xhaka_ds['match_id'].isin(sorted(xhaka_ds['match_id'].unique())[::2])]['decision_surplus'].median():.4f},
    Even matches = {xhaka_ds[xhaka_ds['match_id'].isin(sorted(xhaka_ds['match_id'].unique())[1::2])]['decision_surplus'].median():.4f}.
    Consistent positive surplus across both halves of the season.
    """)

    # DS under different contexts
    st.subheader("Decision Surplus by Context")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("DS — All Passes", f"{xhaka_ds['decision_surplus'].median():.4f}")
        st.metric("DS — Under Pressure", f"{xhaka_ds[xhaka_ds['is_under_pressure']]['decision_surplus'].median():.4f}")
    with col2:
        st.metric("DS — Progressive Passes", f"{xhaka_ds[xhaka_ds['is_progressive']]['decision_surplus'].median():.4f}")
        st.metric("DS — Non-Progressive", f"{xhaka_ds[~xhaka_ds['is_progressive']]['decision_surplus'].median():.4f}")


# ════════════════════════════════════════════════════════════════
# PAGE 5: METHODOLOGY
# ════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.title("Methodology")

    st.markdown("""
    ## The Architect Score

    A composite metric decomposed into six sub-components, each measuring a different
    dimension of creative midfield influence.

    ### 1. Pre-Assist Chain Value (PACV)
    Using a **Transformer encoder** (4 layers, 4 heads, d=64) trained on 5,217 possession chains,
    we predict the terminal xG of each chain and extract **attention weights** to assign credit
    to each action. PACV sums the attention-weighted value for a player's actions in positions
    2-5 from the shot (the "pre-assist" zone).

    - **Val loss:** 0.00158 vs baseline 0.00405 (61% improvement)
    - **Key finding:** The model assigns 27% of credit to the pre-assist zone

    ### 2. Decision Surplus (DS)
    For each pass with a **freeze frame** (player positions), we generate all plausible
    alternative passes to visible teammates, evaluate each through a pass difficulty model
    (GBM, AUC 0.961), and compute the gap between the chosen pass and the average alternative.

    - **Independence:** r = -0.040 with progressive passing
    - **Discrimination:** p = 4.34e-06 (Xhaka vs Andrich)

    ### 3. Defensive Topology Disruption (DTD)
    We model the defensive team as a **graph** (nodes = defenders, edges = coverage within 15m)
    and measure how each action changes the graph's connectivity.

    DTD = 0.5 × edge_change + 0.3 × clustering_change + 0.2 × component_change

    - **Independence:** r = -0.141 with forward pass distance

    ### 4. Press Resistance Value (PRV)
    Mean positional value of passes completed while under defensive pressure.

    ### 5. Chain Initiation Rate (CIR)
    Rate at which a player's actions begin possession chains that end in shots.

    ### 6. Tempo Variance Index (TVI)
    CV(ball hold time) × std(pass direction). Captures rhythm variation.

    ---

    ## Data

    - **StatsBomb Open Data:** 34 Bundesliga matches (Leverkusen 2023/24)
    - **137,765 events** with x/y coordinates
    - **118,581 freeze frames** with all visible player positions
    - **3,012 Xhaka passes** with spatial context (91.3% coverage)

    ## Validation

    | Test | Result | Status |
    |------|--------|--------|
    | Split-half reliability (DS) | Consistent across match halves | ✅ |
    | Independence (DS vs progressive) | r = -0.040 | ✅ |
    | Role discrimination | p = 4.34e-06 | ✅ |
    | Absence effect | 2.14 xG with vs 1.70 without | ✅ |

    ## Limitations

    - Single-season case study — generalizability not proven
    - Freeze frame coverage excludes some off-screen players
    - Positional value proxy used in DS (will be replaced by Transformer values)
    - 5,217 chains is small for deep learning — more data would improve Transformer
    """)


# ════════════════════════════════════════════════════════════════
# PHASE 2 SEPARATOR
# ════════════════════════════════════════════════════════════════
elif page == "── Phase 2 ──":
    st.info("Select a Phase 2 page above.")


# ════════════════════════════════════════════════════════════════
# PAGE 6: CROSS-TOURNAMENT COMPARISON
# ════════════════════════════════════════════════════════════════
elif page == "Cross-Tournament Comparison":
    st.title("Phase 2: Cross-Tournament Comparison")
    st.markdown("*Comparing elite registas across Euro 2024 and Euro 2020 using the Architect Framework.*")

    _p2 = load_phase2_data()
    if _p2 is None:
        st.error("Phase 2 data not found. Run src/phase2_pipeline.py first.")
        st.stop()
    p2_scores, p2_clusters, p2_ds, historical, chain_pos = _p2

    # Filter to target players, prefer euro2024
    TARGET_SUBSTRINGS = ['Pedro González', 'Rodrigo Hernández', 'Zubimendi', 'Vitor Machado',
                          'Kroos', 'Xhaka', 'Fabián', 'Kanté', 'Bellingham', 'Frenkie',
                          'Jorge Luiz', 'Verratti', 'Busquets', 'Phillips']

    def get_target_rows(scores_df, prefer_tournament='euro2024'):
        rows = []
        for sub in TARGET_SUBSTRINGS:
            matches = scores_df[scores_df['player'].str.contains(sub, na=False, regex=False)]
            if len(matches) == 0:
                continue
            pref = matches[matches['tournament'] == prefer_tournament]
            rows.append(pref.iloc[0] if len(pref) > 0 else matches.iloc[0])
        return pd.DataFrame(rows)

    target_df = get_target_rows(p2_scores)
    target_df['display_name'] = target_df['player'].apply(short_name)
    target_df = target_df.sort_values('architect_score_full', ascending=False)

    # Full rankings table
    st.subheader("Architect Score Rankings")
    available_cols = ['display_name', 'tournament', 'total_passes', 'architect_score_full',
                      'architect_score_event', 'pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
    available_cols = [c for c in available_cols if c in target_df.columns]
    display = target_df[available_cols].copy()
    col_map = {'display_name': 'Player', 'tournament': 'Tournament', 'total_passes': 'Passes',
               'architect_score_full': 'AS Full', 'architect_score_event': 'AS Event',
               'pacv_z': 'PACV', 'ds_z': 'DS', 'dtd_z': 'DTD',
               'prv_z': 'PRV', 'cir_z': 'CIR', 'tvi_z': 'TVI'}
    display.columns = [col_map.get(c, c) for c in display.columns]
    st.dataframe(display.style.format({c: '{:.3f}' for c in display.columns if c not in ['Player', 'Tournament']}),
                 use_container_width=True, hide_index=True)

    st.divider()

    # Player radar selector
    st.subheader("Player Radar")
    player_list = target_df['player'].tolist()
    display_names = [short_name(p) for p in player_list]
    selected_idx = st.selectbox("Select player", range(len(player_list)),
                                 format_func=lambda i: display_names[i])
    selected_player = player_list[selected_idx]

    col_left, col_right = st.columns([1, 1])
    with col_left:
        row = target_df[target_df['player'] == selected_player].iloc[0]
        z_vals = [row.get(c, 0) for c in ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
        make_p2_radar(ax, z_vals, ['PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI'],
                      p2_player_color(selected_player), short_name(selected_player))
        fig.patch.set_facecolor(BG_COLOR)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Head-to-Head Selector")
        p_a_idx = st.selectbox("Player A", range(len(player_list)), format_func=lambda i: display_names[i], key='hth_a')
        p_b_idx = st.selectbox("Player B", range(len(player_list)), format_func=lambda i: display_names[i],
                                index=1, key='hth_b')

        p_a = player_list[p_a_idx]
        p_b = player_list[p_b_idx]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
        for pname in [p_a, p_b]:
            row = target_df[target_df['player'] == pname].iloc[0]
            z_vals = [float(row.get(c, 0)) if pd.notna(row.get(c)) else 0.0
                      for c in ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']]
            angles = [n / 6.0 * 2 * np.pi for n in range(6)]
            angles += angles[:1]
            vals_plot = z_vals + [z_vals[0]]
            color = p2_player_color(pname)
            ax.plot(angles, vals_plot, 'o-', linewidth=2, color=color, label=short_name(pname))
            ax.fill(angles, vals_plot, alpha=0.1, color=color)
        ax.set_xticks([n / 6.0 * 2 * np.pi for n in range(6)])
        ax.set_xticklabels(['PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI'], size=9, color='white')
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticklabels([])
        ax.set_facecolor(BG_COLOR)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.8)
        ax.grid(color='#2a2a4e', alpha=0.5)
        ax.spines['polar'].set_color('#2a2a4e')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor(BG_COLOR)
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════
# PAGE 7: ARCHETYPE ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "Archetype Analysis":
    st.title("Phase 2: Archetype Analysis")
    st.markdown("*K-means clustering (k=4) identifies distinct regista archetypes among 79 midfielders "
                "with ≥150 passes and all 6 components available.*")

    _p2 = load_phase2_data()
    if _p2 is None:
        st.error("Phase 2 data not found. Run src/phase2_fix.py first.")
        st.stop()
    p2_scores, p2_clusters, p2_ds, historical, chain_pos = _p2

    # Cluster summary — dynamic columns based on k
    cluster_counts = p2_clusters['cluster_name'].value_counts()
    n_clusters = len(cluster_counts)
    cluster_cols = st.columns(n_clusters)
    for col, (cname, count) in zip(cluster_cols, cluster_counts.items()):
        col.metric(cname, f"{count} players")

    st.divider()

    # Cluster radar centers
    st.subheader("Cluster Profiles (Average Z-Scores)")
    z_cols = ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
    cluster_centers = p2_clusters.groupby('cluster_name')[z_cols].mean()

    colors_map = {
        'The Orchestrator': '#E32636',
        'The Metronome':    '#FFD700',
        'The Disruptor':    '#FF6B35',
        'The Connector':    '#4a9eff',
    }
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5),
                              subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    if n_clusters == 1:
        axes = [axes]
    for ax, (cname, cvals) in zip(axes, cluster_centers.iterrows()):
        color = colors_map.get(cname, '#88ccff')
        make_p2_radar(ax, cvals.values, ['PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI'],
                      color, f"{cname}\n(n={cluster_counts.get(cname, 0)})", alpha=0.4)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Player assignments — show all tournaments (not just first occurrence)
    st.subheader("Player Cluster Assignments")
    target_subs = ['Pedro González', 'Rodrigo Hernández', 'Zubimendi', 'Vitor Machado',
                    'Kroos', 'Xhaka', 'Fabián', 'Kanté', 'Bellingham', 'Frenkie',
                    'Jorge Luiz', 'Verratti', 'Busquets', 'Phillips']

    target_clusters = p2_clusters[p2_clusters['player'].apply(
        lambda p: any(s in p for s in target_subs)
    )][['player', 'tournament', 'cluster_name']].copy()
    target_clusters['display_name'] = target_clusters['player'].apply(short_name)
    target_clusters = target_clusters.sort_values(['cluster_name', 'player'])

    st.dataframe(target_clusters[['display_name', 'tournament', 'cluster_name']].rename(
        columns={'display_name': 'Player', 'tournament': 'Tournament', 'cluster_name': 'Archetype'}
    ), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# PAGE 8: HISTORICAL CONTEXT
# ════════════════════════════════════════════════════════════════
elif page == "Historical Context":
    st.title("Phase 2: Historical Context")
    st.markdown("*How does Xhaka compare to Xavi, Iniesta, and Busquets?*")
    st.info("📊 Only 4 event-based metrics available for historical data (no 360 freeze frames in La Liga open data).")

    _p2 = load_phase2_data()
    if _p2 is None:
        st.error("Phase 2 data not found. Run src/phase2_pipeline.py first.")
        st.stop()
    p2_scores, p2_clusters, p2_ds, historical, chain_pos = _p2

    # Display image if available
    try:
        from PIL import Image
        img = Image.open("outputs/phase2/phase2_historical_timeline.png")
        st.image(img, use_container_width=True)
    except Exception:
        st.info("Timeline chart not found.")

    st.divider()

    # Summary table
    TARGET_HISTORICAL = ['Xavier Hernández', 'Andrés Iniesta', 'Busquets', 'Modrić', 'Xhaka']
    hist_target = historical[historical['player'].apply(
        lambda p: any(s in p for s in TARGET_HISTORICAL)
    )].copy()
    hist_target['display_name'] = hist_target['player'].apply(short_name)

    # Career aggregates
    career = hist_target.groupby('display_name').agg(
        competition=('competition', 'first'),
        seasons=('season_name', 'nunique'),
        n_passes=('n_passes', 'sum'),
        mean_AS=('architect_score_event', 'mean'),
        peak_AS=('architect_score_event', 'max'),
        mean_PRV_z=('prv_z', 'mean'),
        mean_CIR_z=('cir_z', 'mean'),
    ).reset_index().sort_values('mean_AS', ascending=False)

    st.subheader("Career Aggregate (Event-Based Architect Score)")
    st.dataframe(career.style.format({
        'mean_AS': '{:.3f}', 'peak_AS': '{:.3f}',
        'mean_PRV_z': '{:.2f}', 'mean_CIR_z': '{:.2f}',
    }), use_container_width=True, hide_index=True)

    st.markdown("""
    **Key finding:** Xhaka's 2023/24 Leverkusen event-based Architect Score (1.343) places him
    **above Xavi's career average (0.987)** and above peak Iniesta seasons —
    driven overwhelmingly by PRV (press resistance: z=2.91), reflecting Leverkusen's extreme pressing context.
    """)


# ════════════════════════════════════════════════════════════════
# PAGE 9: XHAKA CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════
elif page == "Xhaka Cross-Validation":
    st.title("Phase 2: Xhaka Cross-Validation")
    st.markdown("*Does Xhaka's Leverkusen profile hold up at international level?*")

    _p2 = load_phase2_data()
    if _p2 is None:
        st.error("Phase 2 data not found. Run src/phase2_pipeline.py first.")
        st.stop()
    p2_scores, p2_clusters, p2_ds, historical, chain_pos = _p2

    # Display image
    try:
        from PIL import Image
        img = Image.open("outputs/phase2/phase2_xhaka_crossval.png")
        st.image(img, use_container_width=True)
    except Exception:
        st.info("Cross-validation chart not found.")

    st.divider()

    # Phase 2 scores (Euro 2020, Euro 2024) — midfielders only, no Leverkusen row
    xhaka_euro = p2_scores[p2_scores['player'].str.contains('Xhaka', na=False)].copy()
    xhaka_euro['context'] = xhaka_euro['tournament'].map({
        'euro2020': 'Euro 2020 — Switzerland (P2 mid. pool z-scores)',
        'euro2024': 'Euro 2024 — Switzerland (P2 mid. pool z-scores)',
    }).fillna(xhaka_euro['tournament'])

    # Leverkusen cross-val from separate file
    try:
        xv = pd.read_parquet(DATA_DIR / "phase2_xhaka_crossval.parquet")
        if not xv.empty:
            lev_row = xv.iloc[0]
            lev_display = pd.DataFrame([{
                'context': 'Leverkusen 2023/24 — Bundesliga (P1 pool z-scores)',
                'architect_score_full': lev_row.get('architect_score_full_p1', float('nan')),
                'architect_score_event': float('nan'),
                'pacv_z': float('nan'),
                'ds_z': lev_row.get('ds_z_p1', float('nan')),
                'dtd_z': lev_row.get('dtd_z_p1', float('nan')),
                'prv_z': lev_row.get('prv_z_p1', float('nan')),
                'cir_z': lev_row.get('cir_z_p1', float('nan')),
                'tvi_z': lev_row.get('tvi_z_p1', float('nan')),
            }])
            cols_order = ['context', 'architect_score_full', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
            xhaka_display = pd.concat([lev_display, xhaka_euro], ignore_index=True)
        else:
            xhaka_display = xhaka_euro
    except Exception:
        xhaka_display = xhaka_euro

    cols_show = [c for c in ['context', 'architect_score_full', 'architect_score_event',
                              'pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
                 if c in xhaka_display.columns]
    st.dataframe(xhaka_display[cols_show].rename(columns={
        'context': 'Context', 'architect_score_full': 'AS Full', 'architect_score_event': 'AS Event',
        'pacv_z': 'PACV', 'ds_z': 'DS', 'dtd_z': 'DTD', 'prv_z': 'PRV', 'cir_z': 'CIR', 'tvi_z': 'TVI'
    }).style.format({c: '{:.3f}' for c in ['AS Full', 'AS Event', 'PACV', 'DS', 'DTD', 'PRV', 'CIR', 'TVI']},
                    na_rep='—'),
    use_container_width=True, hide_index=True)

    st.markdown("""
    **Cross-validation finding:** Xhaka's PRV (press-resistance value) drops from **+2.22σ** at Leverkusen
    to **+0.25σ** at international level. This is expected: international opponents press more
    intensely and consistently than Bundesliga teams, reducing the positional value Xhaka extracts
    from pressured passes. His DS (Decision Surplus) remains stable at +0.11-0.56σ across all contexts,
    suggesting his decision-making quality is consistent — it is the *output* (PRV) that varies with
    competitive context, not the underlying process.
    """)
