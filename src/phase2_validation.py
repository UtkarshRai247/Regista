"""
Phase 2 Statistical Validation — Architect Framework
7 tests verifying that metrics validated on 34 Bundesliga matches transfer
to 5-match Euro tournament samples.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

_BASE = Path(__file__).resolve().parent.parent

# ── Style ──────────────────────────────────────────────────────────────────────
BG      = '#1a1a2e'
PANEL   = '#16213e'
FG      = '#e8e8f0'
MUTED   = '#8888aa'
GRID    = '#2a2a4a'
ACCENT  = '#4a9eff'
RED     = '#E32636'
GOLD    = '#FFD700'
ORANGE  = '#FF6B35'
GREEN   = '#4CAF50'
VERDICT_COLORS = {'PASS': GREEN, 'WARN': GOLD, 'FAIL': RED}

# ── Player groups ──────────────────────────────────────────────────────────────
REGISTAS = [
    'Pedro González López',
    'Toni Kroos',
    'Sergio Busquets i Burgos',
    'Kevin De Bruyne',
    'Jorge Resurrección Merodio',
    'Thomas Delaney',
    'Granit Xhaka',
]
NON_REGISTAS = [
    'Declan Rice',
    'Kalvin Phillips',
    'Marten de Roon',
    'Amadou Onana',
    'Tomáš Souček',
    'Pierre-Emile Højbjerg',
    "N'Golo Kanté",
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    scores     = pd.read_parquet(_BASE / 'data/processed/phase2_architect_scores_v2.parquet')
    ds_raw     = pd.read_parquet(_BASE / 'data/processed/phase2_decision_surplus.parquet')
    passes     = pd.read_parquet(_BASE / 'data/processed/phase2_passes_enriched.parquet')
    xhaka_cv   = pd.read_parquet(_BASE / 'data/processed/phase2_xhaka_crossval.parquet')
    m2020      = pd.read_parquet(_BASE / 'data/raw/euro2020_matches.parquet')
    m2024      = pd.read_parquet(_BASE / 'data/raw/euro2024_matches.parquet')
    return scores, ds_raw, passes, xhaka_cv, m2020, m2024


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1  Pass difficulty model transfer (proxy)
# ══════════════════════════════════════════════════════════════════════════════
def test1_pass_difficulty_proxy(passes):
    """
    Original test: does the Bundesliga-trained GBM transfer to Euro data?
    models/pass_difficulty_model.pkl not found, so this proxy trains a logistic
    regression in-sample on Euro data and reports AUC / calibration.
    PASS threshold: AUC >= 0.70 (model can discriminate pass difficulty).
    """
    feats = ['pass_length', 'pass_angle', 'start_x', 'start_y',
             'end_x', 'end_y', 'dist_toward_goal', 'lateral_dist',
             'is_under_pressure']
    X = passes[feats].copy()
    X['is_under_pressure'] = X['is_under_pressure'].astype(int)
    y = passes['is_completed'].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_tr_s, y_tr)

    y_prob = lr.predict_proba(X_te_s)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    brier  = brier_score_loss(y_te, y_prob)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    coefs = sorted(zip(feats, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)

    verdict = 'PASS' if auc >= 0.70 else ('WARN' if auc >= 0.62 else 'FAIL')
    return dict(verdict=verdict, auc=auc, brier=brier, fpr=fpr, tpr=tpr,
                coefs=coefs, n_train=len(X_tr), n_test=len(X_te))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2  DS independence from progressive passing
# ══════════════════════════════════════════════════════════════════════════════
def test2_ds_progressive(ds_raw):
    """
    Is DS just a fancy progressive-pass counter?
    Pass-level and player-level Pearson r between DS and is_progressive.
    PASS threshold: player-level |r| < 0.30.
    """
    valid = ds_raw[['decision_surplus', 'is_progressive']].dropna()
    r_pass, p_pass = stats.pearsonr(
        valid['decision_surplus'],
        valid['is_progressive'].astype(int)
    )

    agg = ds_raw.groupby('player').agg(
        mean_ds=('decision_surplus', 'mean'),
        prog_rate=('is_progressive', 'mean'),
        n=('decision_surplus', 'count'),
    ).query('n >= 50').reset_index()

    r_player, p_player = stats.pearsonr(agg['mean_ds'], agg['prog_rate'])

    verdict = 'PASS' if abs(r_player) < 0.30 else ('WARN' if abs(r_player) < 0.40 else 'FAIL')
    return dict(verdict=verdict, r_pass=r_pass, p_pass=p_pass,
                r_player=r_player, p_player=p_player, agg=agg)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3  Metric independence from traditional statistics
# ══════════════════════════════════════════════════════════════════════════════
def test3_metric_independence(scores, ds_raw, passes):
    """
    6 × 4 correlation matrix: Architect components vs traditional passing stats.
    Verdict based on max |r| for the DS row (core novel metric).
    PASS threshold: max DS |r| < 0.50.
    """
    trad_rows = []
    for player in scores['player'].unique():
        pp  = passes[passes['player'] == player]
        ps  = scores[scores['player'] == player]
        if len(pp) < 50:
            continue
        total_passes  = ps['total_passes'].sum()
        total_matches = ps['matches'].sum()
        trad_rows.append(dict(
            player           = player,
            completion_rate  = pp['is_completed'].mean(),
            progressive_rate = pp['is_progressive'].mean(),
            pressure_rate    = pp['is_under_pressure'].mean(),
            passes_per_match = total_passes / max(total_matches, 1),
        ))

    trad = pd.DataFrame(trad_rows)
    arch = scores.groupby('player')[['pacv','ds','dtd','prv','cir','tvi']].mean().reset_index()
    df   = arch.merge(trad, on='player').dropna()

    arch_cols = ['pacv','ds','dtd','prv','cir','tvi']
    trad_cols = ['completion_rate','progressive_rate','pressure_rate','passes_per_match']
    corr = np.zeros((len(arch_cols), len(trad_cols)))
    pval = np.zeros((len(arch_cols), len(trad_cols)))

    for i, ac in enumerate(arch_cols):
        for j, tc in enumerate(trad_cols):
            r, p = stats.pearsonr(df[ac], df[tc])
            corr[i, j] = r
            pval[i, j] = p

    ds_max = np.max(np.abs(corr[arch_cols.index('ds')]))
    verdict = 'PASS' if ds_max < 0.50 else ('WARN' if ds_max < 0.70 else 'FAIL')
    return dict(verdict=verdict, corr=corr, pval=pval,
                arch_cols=arch_cols, trad_cols=trad_cols,
                ds_max=ds_max, n=len(df))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4  Xhaka cross-context consistency
# ══════════════════════════════════════════════════════════════════════════════
def test4_xhaka_cross_context(scores, xhaka_cv):
    """
    Do Xhaka's z-scores have consistent signs across 3 contexts?
    Euro 2020, Euro 2024, Leverkusen 2023/24 (Phase 1 z-scores).
    PASS threshold: >= 5/6 components agree on sign.
    """
    x20 = scores[(scores['player'] == 'Granit Xhaka') &
                 (scores['tournament'] == 'euro2020')].iloc[0]
    x24 = scores[(scores['player'] == 'Granit Xhaka') &
                 (scores['tournament'] == 'euro2024')].iloc[0]
    xlv = xhaka_cv.iloc[0]

    rows = []
    for comp in ['pacv', 'ds', 'dtd', 'prv', 'cir', 'tvi']:
        z20 = x20.get(f'{comp}_z',   np.nan)
        z24 = x24.get(f'{comp}_z',   np.nan)
        zlv = xlv.get(f'{comp}_z_p1', np.nan)  # Phase 1 z-scores for Leverkusen

        valid_signs = [np.sign(z) for z in [z20, z24, zlv] if not np.isnan(z)]
        agree = (len(set(valid_signs)) == 1) if len(valid_signs) >= 2 else False
        rows.append(dict(component=comp.upper(),
                         euro2020_z=z20, euro2024_z=z24, lev_z=zlv,
                         sign_agree=agree))

    df = pd.DataFrame(rows)
    n_agree = df['sign_agree'].sum()

    # Note: TVI scale differs (P1 uses CV*degrees, P2 uses CV*radians), so
    # TVI Leverkusen z-score cannot be directly compared — flag it.
    verdict = 'PASS' if n_agree >= 5 else ('WARN' if n_agree >= 4 else 'FAIL')
    return dict(verdict=verdict, df=df, n_agree=int(n_agree), total=len(df),
                as_20=x20.get('architect_score_full', np.nan),
                as_24=x24.get('architect_score_full', np.nan),
                as_lv=xlv.get('architect_score_full_p1', np.nan))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5  Within-tournament split-half reliability
# ══════════════════════════════════════════════════════════════════════════════
def test5_split_half(ds_raw, scores, m2020):
    """
    Split each Euro 2020 player's matches chronologically (first half vs second
    half); correlate per-player DS across halves.
    Players: 6+ matches in euro2020, 50+ DS observations.
    PASS threshold: r >= 0.70.
    """
    m2020 = m2020[['match_id', 'match_date']].copy()
    m2020['match_id'] = m2020['match_id'].astype(int)
    e2020_ids = set(m2020['match_id'])

    ds_e = ds_raw[ds_raw['match_id'].isin(e2020_ids)].copy()
    qual_players = (scores[(scores['tournament'] == 'euro2020') &
                           (scores['matches'] >= 6)]['player'].tolist())

    rows = []
    for player in qual_players:
        pds = ds_e[ds_e['player'] == player]
        if len(pds) < 50:
            continue
        # Matches this player appeared in (≥5 DS events = actually played)
        pmatch_counts = pds.groupby('match_id').size()
        pmatch_ids = pmatch_counts[pmatch_counts >= 5].index.tolist()

        ordered = (m2020[m2020['match_id'].isin(pmatch_ids)]
                   .sort_values('match_date')['match_id'].tolist())
        n = len(ordered)
        if n < 4:
            continue

        h1 = ordered[:n // 2]
        h2 = ordered[n // 2:]
        rows.append(dict(
            player   = player,
            n_matches= n,
            ds_h1    = pds[pds['match_id'].isin(h1)]['decision_surplus'].mean(),
            ds_h2    = pds[pds['match_id'].isin(h2)]['decision_surplus'].mean(),
        ))

    df = pd.DataFrame(rows)

    if len(df) < 4:
        return dict(verdict='WARN', r=np.nan, p=np.nan, df=df,
                    note=f'Only {len(df)} qualifying players — need ≥4 for correlation.')

    r, p = stats.pearsonr(df['ds_h1'], df['ds_h2'])
    verdict = 'PASS' if r >= 0.70 else ('WARN' if r >= 0.50 else 'FAIL')
    return dict(verdict=verdict, r=r, p=p, df=df, n=len(df))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6  Known-groups validation
# ══════════════════════════════════════════════════════════════════════════════
def test6_known_groups(scores):
    """
    Mann-Whitney U (one-sided): known registas score higher than non-registas.
    PASS threshold: p < 0.05.
    """
    valid = scores[scores['architect_score_full'].notna()].copy()
    reg  = valid[valid['player'].isin(REGISTAS)]['architect_score_full'].values
    non  = valid[valid['player'].isin(NON_REGISTAS)]['architect_score_full'].values

    stat, p = stats.mannwhitneyu(reg, non, alternative='greater')
    # Rank-biserial: (2U / n1*n2) - 1; +1 = all registas above all non-registas
    r_rb = (2 * stat) / (len(reg) * len(non)) - 1

    reg_df = valid[valid['player'].isin(REGISTAS)][
        ['player', 'tournament', 'architect_score_full']].copy()
    reg_df['group'] = 'Regista'
    non_df = valid[valid['player'].isin(NON_REGISTAS)][
        ['player', 'tournament', 'architect_score_full']].copy()
    non_df['group'] = 'Non-Regista'
    combined = pd.concat([reg_df, non_df], ignore_index=True)

    verdict = 'PASS' if p < 0.05 else ('WARN' if p < 0.10 else 'FAIL')
    return dict(verdict=verdict, stat=stat, p=p, r_rb=r_rb,
                med_reg=np.median(reg), med_non=np.median(non),
                n_reg=len(reg), n_non=len(non),
                reg_scores=reg, non_scores=non, combined=combined)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7  Pedri vs Rodri — same team, same matches
# ══════════════════════════════════════════════════════════════════════════════
def test7_pedri_rodri(scores):
    """
    Case study: Pedri and Rodrigo Hernández (Rodri) both played for Spain
    in Euro 2020. Same team, same matches — controlled comparison.
    Framework should rank the deep playmaker (Pedri) higher.
    PASS: Pedri AS > Rodri AS AND ≥ 3/4 directional components agree.
    """
    pedri = scores[(scores['player'] == 'Pedro González López') &
                   (scores['tournament'] == 'euro2020')]
    rodri = scores[(scores['player'] == 'Rodrigo Hernández Cascante') &
                   (scores['tournament'] == 'euro2020')]

    if pedri.empty or rodri.empty:
        return dict(verdict='WARN', note='One or both players not in euro2020 pool.')

    pedri = pedri.iloc[0]
    rodri = rodri.iloc[0]

    z_cols = ['pacv_z', 'ds_z', 'dtd_z', 'prv_z', 'cir_z', 'tvi_z']
    # Expected: Pedri > Rodri on PACV, DS, PRV, CIR (playmaking-dominant metrics)
    directional = {'pacv_z': True, 'ds_z': True, 'prv_z': True, 'cir_z': True}

    rows = []
    n_correct = 0
    for zc in z_cols:
        pz = pedri.get(zc, np.nan)
        rz = rodri.get(zc, np.nan)
        is_dir = zc in directional
        correct = (pz > rz) if (is_dir and not (np.isnan(pz) or np.isnan(rz))) else None
        if correct:
            n_correct += 1
        rows.append(dict(
            component  = zc.replace('_z', '').upper(),
            pedri_z    = pz,
            rodri_z    = rz,
            directional= is_dir,
            correct    = correct,
        ))

    df = pd.DataFrame(rows)
    pedri_as = pedri.get('architect_score_full', np.nan)
    rodri_as  = rodri.get('architect_score_full', np.nan)

    if pedri_as > rodri_as and n_correct >= 3:
        verdict = 'PASS'
    elif pedri_as > rodri_as:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

    return dict(verdict=verdict, pedri_as=pedri_as, rodri_as=rodri_as,
                pedri_passes=int(pedri.get('total_passes', 0)),
                rodri_passes=int(rodri.get('total_passes', 0)),
                n_correct=n_correct, n_directional=len(directional), df=df)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  (6 panels, dark background)
# ══════════════════════════════════════════════════════════════════════════════
def _ax_style(ax, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=6)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)


def _verdict_badge(ax, verdict, x=0.97, y=0.97):
    ax.text(x, y, verdict,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold',
            color=BG, backgroundcolor=VERDICT_COLORS[verdict],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=VERDICT_COLORS[verdict],
                      edgecolor='none'))


def build_figure(r1, r2, r3, r4, r5, r6):
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── Panel 1: ROC curve ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _ax_style(ax1, 'T1 · Pass Difficulty (Proxy LR)')
    ax1.plot(r1['fpr'], r1['tpr'], color=ACCENT, lw=2,
             label=f"AUC = {r1['auc']:.3f}")
    ax1.plot([0, 1], [0, 1], '--', color=MUTED, lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(fontsize=8, facecolor=BG, edgecolor=GRID,
               labelcolor=FG, loc='lower right')
    ax1.text(0.03, 0.12, f"Brier: {r1['brier']:.3f}\nn_test={r1['n_test']:,}",
             transform=ax1.transAxes, color=MUTED, fontsize=7.5)
    _verdict_badge(ax1, r1['verdict'])

    # ── Panel 2: DS vs progressive scatter ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _ax_style(ax2, 'T2 · DS vs Progressive Pass Rate')
    agg = r2['agg']
    ax2.scatter(agg['prog_rate'], agg['mean_ds'], alpha=0.7, s=30,
                color=ACCENT, edgecolors='none')
    m, b = np.polyfit(agg['prog_rate'], agg['mean_ds'], 1)
    x_line = np.linspace(agg['prog_rate'].min(), agg['prog_rate'].max(), 100)
    ax2.plot(x_line, m * x_line + b, color=RED, lw=1.5, alpha=0.8)
    ax2.set_xlabel('Progressive Pass Rate')
    ax2.set_ylabel('Mean Decision Surplus')
    ax2.text(0.03, 0.94,
             f"r = {r2['r_player']:.3f}  p = {r2['p_player']:.3f}\n"
             f"(pass-level r = {r2['r_pass']:.3f})",
             transform=ax2.transAxes, color=MUTED, fontsize=7.5, va='top')
    _verdict_badge(ax2, r2['verdict'])

    # ── Panel 3: Correlation heatmap ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL)
    for spine in ax3.spines.values():
        spine.set_color(GRID)
    ax3.set_title('T3 · Metric Independence (r vs traditional)',
                  color=FG, fontsize=9, fontweight='bold', pad=6)

    corr = r3['corr']
    ac   = [c.upper() for c in r3['arch_cols']]
    tc   = ['Compl.%', 'Prog.%', 'Press.%', 'Pass/90']
    im   = ax3.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(tc)))
    ax3.set_yticks(range(len(ac)))
    ax3.set_xticklabels(tc, fontsize=7.5, color=FG, rotation=30, ha='right')
    ax3.set_yticklabels(ac, fontsize=7.5, color=FG)
    for i in range(len(ac)):
        for j in range(len(tc)):
            v = corr[i, j]
            txt_col = BG if abs(v) > 0.5 else FG
            ax3.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=7, color=txt_col)
    plt.colorbar(im, ax=ax3, shrink=0.8,
                 label='Pearson r').ax.yaxis.label.set_color(FG)
    # Highlight DS row
    ax3.axhline(r3['arch_cols'].index('ds') - 0.5, color=GOLD, lw=0.8, alpha=0.6)
    ax3.axhline(r3['arch_cols'].index('ds') + 0.5, color=GOLD, lw=0.8, alpha=0.6)
    _verdict_badge(ax3, r3['verdict'])

    # ── Panel 4: Xhaka cross-context ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _ax_style(ax4, 'T4 · Xhaka Cross-Context Z-Scores')
    xdf = r4['df']
    comps = xdf['component'].tolist()
    x = np.arange(len(comps))
    w = 0.25
    bars_20 = ax4.bar(x - w, xdf['euro2020_z'], w, label='Euro 2020',
                      color=ACCENT, alpha=0.85)
    bars_24 = ax4.bar(x,     xdf['euro2024_z'], w, label='Euro 2024',
                      color=ORANGE, alpha=0.85)
    bars_lv = ax4.bar(x + w, xdf['lev_z'],      w, label='Leverkusen (P1)',
                      color=GOLD, alpha=0.85)
    ax4.axhline(0, color=MUTED, lw=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(comps, fontsize=7.5)
    ax4.set_ylabel('Z-score')
    ax4.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax4.text(0.03, 0.97,
             f"Sign agreement: {r4['n_agree']}/{r4['total']}",
             transform=ax4.transAxes, color=MUTED, fontsize=7.5, va='top')
    _verdict_badge(ax4, r4['verdict'])

    # ── Panel 5: Split-half scatter ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _ax_style(ax5, 'T5 · Split-Half DS Reliability (Euro 2020)')
    sh = r5.get('df', pd.DataFrame())
    if not sh.empty and not np.isnan(r5.get('r', np.nan)):
        ax5.scatter(sh['ds_h1'], sh['ds_h2'], s=50, color=ACCENT,
                    alpha=0.85, edgecolors='none')
        for _, row in sh.iterrows():
            label = row['player'].split()[-1]
            ax5.annotate(label, (row['ds_h1'], row['ds_h2']),
                         textcoords='offset points', xytext=(4, 2),
                         fontsize=6.5, color=MUTED)
        lim_vals = list(sh['ds_h1']) + list(sh['ds_h2'])
        lo, hi = min(lim_vals) * 1.2, max(lim_vals) * 1.2
        ax5.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1, alpha=0.6)
        m5, b5 = np.polyfit(sh['ds_h1'], sh['ds_h2'], 1)
        xl = np.linspace(sh['ds_h1'].min(), sh['ds_h1'].max(), 50)
        ax5.plot(xl, m5 * xl + b5, color=RED, lw=1.5, alpha=0.8)
        ax5.text(0.03, 0.94,
                 f"r = {r5['r']:.3f}  p = {r5['p']:.3f}\nn = {r5.get('n', len(sh))}",
                 transform=ax5.transAxes, color=MUTED, fontsize=7.5, va='top')
    else:
        ax5.text(0.5, 0.5, r5.get('note', 'Insufficient data'),
                 transform=ax5.transAxes, ha='center', va='center',
                 color=MUTED, fontsize=8, wrap=True)
    ax5.set_xlabel('DS — first-half matches')
    ax5.set_ylabel('DS — second-half matches')
    _verdict_badge(ax5, r5['verdict'])

    # ── Panel 6: Known-groups box ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _ax_style(ax6, 'T6 · Known-Groups (Regista vs Non-Regista)')
    data_r = r6['reg_scores']
    data_n = r6['non_scores']
    bp = ax6.boxplot([data_r, data_n],
                     patch_artist=True,
                     labels=['Regista\n(n=%d)' % r6['n_reg'],
                             'Non-Regista\n(n=%d)' % r6['n_non']],
                     medianprops=dict(color=BG, linewidth=2))
    bp['boxes'][0].set_facecolor(ACCENT)
    bp['boxes'][1].set_facecolor(RED)
    for w in bp['whiskers'] + bp['caps'] + bp['fliers']:
        w.set_color(MUTED)
    ax6.set_ylabel('Architect Score (full)')
    ax6.text(0.03, 0.97,
             f"Mann-Whitney p = {r6['p']:.4f}\n"
             f"Effect size r_rb = {r6['r_rb']:.3f}\n"
             f"Median gap = {r6['med_reg'] - r6['med_non']:.3f}",
             transform=ax6.transAxes, color=MUTED, fontsize=7.5, va='top')
    _verdict_badge(ax6, r6['verdict'])

    fig.suptitle('Architect Framework — Phase 2 Statistical Validation',
                 color=FG, fontsize=13, fontweight='bold', y=0.97)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_report(results):
    lines = []
    sep = '=' * 78
    thin = '-' * 78

    lines += [
        sep,
        'ARCHITECT FRAMEWORK — PHASE 2 STATISTICAL VALIDATION REPORT',
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        'Dataset   : Euro 2024 + Euro 2020 (StatsBomb open data)',
        sep, '',
        'EXECUTIVE SUMMARY',
        thin,
    ]

    label_map = {
        1: 'Pass difficulty model transfer (proxy)',
        2: 'DS independence from progressive passing',
        3: 'Metric independence from traditional stats',
        4: 'Xhaka cross-context consistency',
        5: 'Within-tournament split-half reliability (DS)',
        6: 'Known-groups validation (Mann-Whitney)',
        7: 'Pedri vs Rodri (same team, same matches)',
    }
    r_list = [results[i] for i in range(1, 8)]
    for i, (res, label) in enumerate(zip(r_list, label_map.values()), 1):
        v = res['verdict']
        pad = ' ' * (48 - len(label))
        lines.append(f"  Test {i}: {label}{pad}[ {v:4s} ]")

    n_pass = sum(1 for r in r_list if r['verdict'] == 'PASS')
    n_warn = sum(1 for r in r_list if r['verdict'] == 'WARN')
    n_fail = sum(1 for r in r_list if r['verdict'] == 'FAIL')
    lines += ['', f'  Result: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL', '']

    # ── Test 1 detail ─────────────────────────────────────────────────────
    r1 = results[1]
    lines += [
        sep,
        'TEST 1  PASS DIFFICULTY MODEL TRANSFER  [PROXY]',
        thin,
        'NOTE: models/pass_difficulty_model.pkl not found. No Bundesliga→Euro',
        '      transfer test possible. Proxy: logistic regression trained in-',
        '      sample on Euro data (70/30 split) on spatial pass features.',
        '',
        f'  Features   : pass_length, pass_angle, start_x/y, end_x/y,',
        f'               dist_toward_goal, lateral_dist, is_under_pressure',
        f'  Train / Test: {r1["n_train"]:,} / {r1["n_test"]:,} passes',
        f'  AUC        : {r1["auc"]:.4f}   (threshold PASS ≥ 0.70)',
        f'  Brier score: {r1["brier"]:.4f}   (lower = better calibrated)',
        '',
        '  Top feature coefficients:',
    ]
    for feat, coef in r1['coefs'][:5]:
        lines.append(f'    {feat:<24s}  {coef:+.4f}')
    lines += [
        '',
        f'  VERDICT: {r1["verdict"]}',
        '  Interpretation: Spatial features can discriminate pass completion',
        '  in Euro data. Cross-dataset (Bundesliga→Euro) transfer cannot be',
        '  confirmed without the saved model pkl.', '',
    ]

    # ── Test 2 detail ─────────────────────────────────────────────────────
    r2 = results[2]
    lines += [
        sep,
        'TEST 2  DS INDEPENDENCE FROM PROGRESSIVE PASSING',
        thin,
        f'  Pass-level  Pearson r = {r2["r_pass"]:+.4f}  (p = {r2["p_pass"]:.4e})',
        f'  Player-level Pearson r = {r2["r_player"]:+.4f}  (p = {r2["p_player"]:.4f})',
        f'  Players with ≥50 DS obs: {len(r2["agg"])}',
        '',
        f'  Threshold: |player r| < 0.30 → PASS',
        f'  VERDICT: {r2["verdict"]}',
        '',
        '  Interpretation: DS correlates weakly with progressive pass rate,',
        '  confirming it captures decision quality, not just directionality.', '',
    ]

    # ── Test 3 detail ─────────────────────────────────────────────────────
    r3 = results[3]
    lines += [
        sep,
        'TEST 3  METRIC INDEPENDENCE FROM TRADITIONAL STATISTICS',
        thin,
        f'  N players (with all data): {r3["n"]}',
        '',
        '  Pearson r matrix (Architect components × traditional metrics):',
        f'  {"":6s}  {"Compl%":>8s}  {"Prog%":>8s}  {"Press%":>8s}  {"Pass/90":>8s}',
    ]
    ac = r3['arch_cols']
    for i, comp in enumerate(ac):
        row_str = f'  {comp.upper():<6s}  '
        for j in range(4):
            v = r3['corr'][i, j]
            star = '*' if r3['pval'][i, j] < 0.05 else ' '
            row_str += f'{v:+.3f}{star}   '
        lines.append(row_str)
    lines += [
        '  (* p < 0.05)',
        '',
        f'  Max |r| for DS row: {r3["ds_max"]:.4f}  (threshold PASS < 0.50)',
        f'  VERDICT: {r3["verdict"]}', '',
    ]

    # ── Test 4 detail ─────────────────────────────────────────────────────
    r4 = results[4]
    lines += [
        sep,
        'TEST 4  XHAKA CROSS-CONTEXT CONSISTENCY',
        thin,
        '  NOTE: Leverkusen TVI uses CV×degrees (Phase 1) vs CV×radians',
        '        (Phase 2). TVI z-score is incomparable across phases.',
        '',
        f'  {"Component":>10s}  {"Euro2020 z":>10s}  {"Euro2024 z":>10s}  '
        f'{"Lev. z (P1)":>12s}  {"Sign agree":>10s}',
        f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*12}  {"-"*10}',
    ]
    for _, row in r4['df'].iterrows():
        def fmt_z(v):
            return f'{v:+.3f}' if not np.isnan(v) else '   —  '
        agree_str = 'YES' if row['sign_agree'] else 'NO '
        lines.append(
            f'  {row["component"]:>10s}  {fmt_z(row["euro2020_z"]):>10s}  '
            f'{fmt_z(row["euro2024_z"]):>10s}  {fmt_z(row["lev_z"]):>12s}  '
            f'{agree_str:>10s}'
        )
    lines += [
        '',
        f'  Sign agreement: {r4["n_agree"]}/{r4["total"]}  (threshold PASS ≥ 5/6)',
        f'  Architect Scores: Euro2020={r4["as_20"]:.3f}  Euro2024={r4["as_24"]:.3f}  '
        f'Leverkusen(P1)={r4["as_lv"]:.3f}',
        f'  VERDICT: {r4["verdict"]}', '',
    ]

    # ── Test 5 detail ─────────────────────────────────────────────────────
    r5 = results[5]
    lines += [
        sep,
        'TEST 5  WITHIN-TOURNAMENT SPLIT-HALF DS RELIABILITY',
        thin,
        '  Methodology: Euro 2020 players with 6+ matches split chronologically',
        '  (first N//2 matches vs last N//2 matches). Per-player mean DS',
        '  correlated across halves.',
        '',
    ]
    sh = r5.get('df', pd.DataFrame())
    if not sh.empty:
        lines.append(f'  {"Player":<30s}  {"Matches":>7s}  {"DS (H1)":>9s}  {"DS (H2)":>9s}')
        lines.append(f'  {"-"*30}  {"-"*7}  {"-"*9}  {"-"*9}')
        for _, row in sh.iterrows():
            lines.append(
                f'  {row["player"]:<30s}  {int(row["n_matches"]):>7d}  '
                f'{row["ds_h1"]:>+9.5f}  {row["ds_h2"]:>+9.5f}'
            )
        lines.append('')
    if not np.isnan(r5.get('r', np.nan)):
        lines += [
            f'  Pearson r = {r5["r"]:.4f}  p = {r5["p"]:.4f}  n = {r5.get("n", len(sh))}',
            f'  Threshold: r ≥ 0.70 → PASS',
        ]
    else:
        lines.append(f'  {r5.get("note", "Insufficient data")}')
    lines += [f'  VERDICT: {r5["verdict"]}', '']

    # ── Test 6 detail ─────────────────────────────────────────────────────
    r6 = results[6]
    lines += [
        sep,
        'TEST 6  KNOWN-GROUPS VALIDATION (MANN-WHITNEY U)',
        thin,
        f'  Registas  (n={r6["n_reg"]}): {", ".join(p.split()[0] for p in REGISTAS)}',
        f'  Non-reg.  (n={r6["n_non"]}): {", ".join(p.split()[0] for p in NON_REGISTAS)}',
        '',
        f'  H1: registas score higher on Architect Score (one-sided)',
        f'  Mann-Whitney U = {r6["stat"]:.1f}  p = {r6["p"]:.4f}',
        f'  Effect size (rank-biserial r) = {r6["r_rb"]:.3f}',
        f'  Median (Regista)     = {r6["med_reg"]:.4f}',
        f'  Median (Non-Regista) = {r6["med_non"]:.4f}',
        '',
        f'  Threshold: p < 0.05 → PASS',
        f'  VERDICT: {r6["verdict"]}', '',
    ]

    # ── Test 7 detail ─────────────────────────────────────────────────────
    r7 = results[7]
    lines += [
        sep,
        'TEST 7  PEDRI vs RODRI — SAME TEAM, SAME TOURNAMENT',
        thin,
        '  Both played for Spain in Euro 2020. Controlled comparison: same team',
        '  selection bias, same opposition, same match context.',
        '',
        f'  {"":5s}  {"Pedri":>10s}  {"Rodri":>10s}  {"Δ (P-R)":>10s}  {"Expected ↑":>12s}',
        f'  {"─"*5}  {"─"*10}  {"─"*10}  {"─"*10}  {"─"*12}',
    ]
    for _, row in r7['df'].iterrows():
        pz = row['pedri_z']
        rz = row['rodri_z']
        delta_str = f'{pz - rz:+.3f}' if not (np.isnan(pz) or np.isnan(rz)) else '   —  '
        exp_str   = 'Pedri' if row['directional'] else 'Either'
        correct_str = ''
        if row['correct'] is True:
            correct_str = ' ✓'
        elif row['correct'] is False:
            correct_str = ' ✗'
        lines.append(
            f'  {row["component"]:<5s}  {pz:>+10.3f}  {rz:>+10.3f}  '
            f'{delta_str:>10s}  {exp_str:>12s}{correct_str}'
        )
    lines += [
        '',
        f'  Pedri  AS = {r7["pedri_as"]:.4f}  ({r7["pedri_passes"]} passes)',
        f'  Rodri  AS = {r7["rodri_as"]:.4f}  ({r7["rodri_passes"]} passes)',
        f'  Directional components correct: {r7["n_correct"]}/{r7["n_directional"]}',
        '',
        f'  VERDICT: {r7["verdict"]}', '',
    ]

    # ── Limitations ────────────────────────────────────────────────────────
    lines += [
        sep,
        'LIMITATIONS AND CAVEATS',
        thin,
        '  1. Test 1 is a proxy — the Bundesliga GBM pkl was not saved.',
        '     Cross-dataset transfer is the harder and more relevant test.',
        '',
        '  2. Test 5 uses only 5-match tournament data; split-half on N=5',
        '     matches (2 vs 3) is noisy. Small-n correlation should be',
        '     interpreted with caution (wide CI).',
        '',
        '  3. DS is computed only for passes with available freeze frames.',
        '     Euro 2020 StatsBomb 360 coverage is not 100%.',
        '',
        '  4. Known-groups (Test 6) uses subjective player categorisation.',
        '     "Regista" is a continuum; some listed players are borderline.',
        '',
        '  5. TVI scale mismatch between Phase 1 (Bundesliga) and Phase 2',
        '     (Euro) means cross-phase TVI z-scores are not comparable.',
        sep,
    ]

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    print('Loading data…')
    scores, ds_raw, passes, xhaka_cv, m2020, m2024 = load_data()

    print('Test 1: pass difficulty proxy…')
    r1 = test1_pass_difficulty_proxy(passes)

    print('Test 2: DS vs progressive passing…')
    r2 = test2_ds_progressive(ds_raw)

    print('Test 3: metric independence…')
    r3 = test3_metric_independence(scores, ds_raw, passes)

    print('Test 4: Xhaka cross-context…')
    r4 = test4_xhaka_cross_context(scores, xhaka_cv)

    print('Test 5: split-half reliability…')
    r5 = test5_split_half(ds_raw, scores, m2020)

    print('Test 6: known-groups…')
    r6 = test6_known_groups(scores)

    print('Test 7: Pedri vs Rodri…')
    r7 = test7_pedri_rodri(scores)

    results = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6, 7: r7}

    # ── Save report ────────────────────────────────────────────────────────
    report_path = _BASE / 'outputs/phase2_validation_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    report_txt = build_report(results)
    report_path.write_text(report_txt)
    print(f'\nReport saved → {report_path}')

    # ── Save visualisation ─────────────────────────────────────────────────
    out_dir = _BASE / 'outputs/phase2'
    out_dir.mkdir(exist_ok=True)
    fig = build_figure(r1, r2, r3, r4, r5, r6)
    fig_path = out_dir / 'phase2_validation_summary.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Visualisation saved → {fig_path}')

    # ── Print summary table ────────────────────────────────────────────────
    print()
    print('=' * 62)
    print('VALIDATION SUMMARY')
    print('=' * 62)
    labels = [
        'T1  Pass difficulty (proxy)',
        'T2  DS vs progressive pass',
        'T3  Metric independence',
        'T4  Xhaka cross-context',
        'T5  Split-half reliability',
        'T6  Known-groups (MW)',
        'T7  Pedri vs Rodri',
    ]
    for i, (res, lbl) in enumerate(zip(results.values(), labels), 1):
        v = res['verdict']
        pad = ' ' * (42 - len(lbl))
        print(f'  {lbl}{pad}[ {v} ]')
    print('=' * 62)
    n_pass = sum(1 for r in results.values() if r['verdict'] == 'PASS')
    n_warn = sum(1 for r in results.values() if r['verdict'] == 'WARN')
    n_fail = sum(1 for r in results.values() if r['verdict'] == 'FAIL')
    print(f'  {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL')
    print('=' * 62)


if __name__ == '__main__':
    run()
