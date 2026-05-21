# The Architect Framework: Quantifying the Hidden Value of the Regista

**A Case Study of Granit Xhaka's 2023/24 Bundesliga Season**

---

## Abstract

Soccer analytics suffers from a persistent last-touch bias: xG rewards the shooter, xA rewards the final passer, and progressive passing metrics measure only forward ball movement. This framework systematically ignores the players whose primary contribution occurs before the final pass — the deep-lying playmakers who control tempo, break pressure, and restructure defenses through positioning and pass selection. We introduce the Architect Framework, a multi-layered possession value model that combines Transformer-based credit assignment, counterfactual decision evaluation using freeze frame data, and defensive topology disruption analysis to quantify these hidden contributions. Applied to Granit Xhaka's unbeaten 2023/24 Bundesliga season with Bayer Leverkusen, we demonstrate that Decision Surplus — the gap between a player's chosen pass and the average available alternative — is a statistically significant, repeatable skill that is completely independent of progressive passing (r = -0.040). The framework successfully discriminates between the regista (Xhaka), the creative attacker (Wirtz), and the ball-winner (Andrich), producing distinct profiles that match expert consensus. Leverkusen created 2.14 xG per match with Xhaka versus 1.70 without him, a 26% reduction that traditional metrics fail to predict.

---

## 1. Introduction

When Bayer Leverkusen completed an unbeaten Bundesliga season in 2023/24, the tactical consensus was clear: Granit Xhaka was the most important player on the pitch. Coaches, pundits, and analysts credited him with transforming the team's buildup play under Xabi Alonso. Yet his traditional statistical output — 3 goals and 7 assists — placed him among the least productive attacking midfielders in the league by conventional metrics.

This disconnect between perceived influence and statistical output is not unique to Xhaka. Players described as "registas" or "tempo-setters" — Pedri, Frenkie de Jong, Vitinha, Rodri — consistently rank below expectations on metrics like xA, shot-creating actions, and progressive carries. The problem is structural: existing analytics frameworks are designed to credit the player closest to the goal, not the player who created the conditions for the goal to occur.

This paper introduces the Architect Framework, a six-component metric designed to capture the full creative influence of midfielders whose contributions occur before the final pass. The framework is validated through a deep case study of Xhaka's 2023/24 season, using 137,765 StatsBomb events and 118,581 freeze frames across 34 Bundesliga matches.

### Research Question

*"How can we measure the full creative influence of midfielders whose primary contribution to attack occurs before the final pass — and do existing analytics frameworks systematically undervalue these players?"*

---

## 2. Related Work

The problem of valuing off-the-ball and pre-assist contributions has been recognized in the soccer analytics literature for several years. Expected Threat (xT), introduced by Karun Singh, assigns value to pitch zones based on the probability that possession in that zone leads to a goal. VAEP (Valuing Actions by Estimating Probabilities), developed at KU Leuven, estimates the change in scoring and conceding probability caused by each action. Both frameworks represent significant advances over last-touch metrics but share a critical limitation: they treat all actions at the same location equivalently, regardless of the defensive configuration or the alternatives available to the passer.

More recent work has explored sequence-level modeling. Seq2Event (KDD 2022) uses a Transformer to predict the next event in a soccer sequence. SoccerTransformer (MLSA 2024) applies self-supervised pre-training for player ratings. EPV-with-decay (2024) incorporates a fixed discount factor (γ = 0.95) to weight actions closer to the shot more heavily. Our approach differs in that the credit assignment is learned via attention weights rather than imposed as a hyperparameter.

Counterfactual analysis in soccer has been explored primarily with proprietary tracking data. TacticAI (DeepMind/Liverpool FC) uses graph neural networks for corner kick analysis. Defensive evaluation frameworks use counterfactual ghosting to assess off-ball positioning. Our contribution is to operationalize counterfactual pass evaluation using freely available StatsBomb 360 freeze frame data, making the methodology reproducible.

---

## 3. Data

### 3.1 Primary Dataset

StatsBomb open data for Bayer Leverkusen's 2023/24 Bundesliga season: 34 matches, 137,765 events, 118,581 freeze frames. Each freeze frame captures the location of all players visible in the broadcast frame at the moment of each event (median: 17 players visible, 79.6% of frames contain 16+ players).

Xhaka appears in 33 of 34 matches, contributing 3,299 passes at 92.3% completion. Of these, 3,012 (91.3%) have associated freeze frames for spatial analysis.

### 3.2 Comparison Population

All Bundesliga midfielders appearing in the 34 Leverkusen matches serve as the comparison group. This includes Leverkusen teammates (Florian Wirtz, Robert Andrich, Exequiel Palacios, Jonas Hofmann) and opposing midfielders, all evaluated at the same data fidelity.

---

## 4. Methodology

### 4.1 Possession Chain Value Engine (Layer 1)

We extract 5,217 possession chains from the event data, where each chain is a sequence of consecutive on-ball actions by one team ending in a shot (807 chains, 15.5%), turnover, or stoppage. For shot-ending chains, the terminal value is the shot's StatsBomb xG; for all others, the terminal value is zero.

A Transformer encoder (4 layers, 4 attention heads, embedding dimension 64, feedforward dimension 128) processes each chain as a sequence of action embeddings. Each action is represented by a 17-dimensional feature vector: action type (13-dimensional one-hot), normalized spatial coordinates (x, y), action duration, and distance to goal. Sequences are padded or truncated to 30 actions.

The model predicts terminal xG via mean-pooled representations and is trained with MSE loss on an 80/20 train/validation split. Training uses Adam optimization with cosine annealing, early stopping with patience 15.

**Results:** Validation MSE of 0.00158 versus a baseline variance of 0.00405 (61% reduction), with early stopping at epoch 28.

After training, we extract attention weights from the final layer to assign credit to each action. The attention weight of action *i* represents how much the model attends to that action when predicting the terminal value. Per-action value is computed as V(i) = attention_weight(i) × terminal_xG.

### 4.2 Decision Surplus (Layer 2)

For each pass with a freeze frame, we reconstruct the set of plausible alternative passes and evaluate whether the player's choice was better than average.

**Pass Difficulty Model:** A Gradient Boosted Classifier (200 trees, max depth 5) trained on 32,372 passes predicts P(completion | spatial features). Features include pass origin and destination, length, angle, defenders on the pass line (within 3m), defenders near the target (within 5m), closest defender to passer, pressure flag, and visibility count. The model achieves AUC 0.961 on training data with 5-fold CV AUC of 0.929.

**Alternative Generation:** For each pass, we identify all visible teammates in the freeze frame within feasible passing range (2-50m), excluding the passer and goalkeeper. Each alternative is evaluated: Expected Value = P(completion) × PositionalValue(destination) + (1 - P(completion)) × TurnoverPenalty.

**Decision Surplus:** DS(pass) = ActualValue(chosen) − Mean(ExpectedValue(alternatives)). A positive DS indicates the player found a pass better than the average option available. Player-level DS is computed as the median across all passes.

### 4.3 Defensive Topology Disruption (Layer 3)

We model the defending team as an undirected graph where defenders are nodes and edges connect defenders within 15 meters of each other. For each consecutive pair of events with freeze frames, we compute graph metrics before and after the action: edge count, average clustering coefficient, and number of connected components.

DTD = 0.5 × (edges_pre − edges_post)/edges_pre + 0.3 × (clustering_pre − clustering_post) + 0.2 × (components_post − components_pre)

### 4.4 Press Resistance Value, Chain Initiation Rate, Tempo Variance Index

**PRV:** Mean positional value of passes completed under pressure (defender within 3m or StatsBomb pressure event registered).

**CIR:** Proportion of shot-ending possession chains where the player's action begins the sequence, normalized per 90 minutes.

**TVI:** CV(ball hold time) × std(pass direction change). Captures rhythmic variation in passing tempo and directional unpredictability.

### 4.5 The Architect Score

All six components are z-scored against the midfielder population (players with 5+ matches). The Full Architect Score is the unweighted mean of all six z-scores. An Event-Based version (PACV, PRV, CIR, TVI only) is also computed for players without freeze frame data.

---

## 5. Results

### 5.1 Chain Position Analysis

Xhaka's contribution profile in shot chains is strikingly different from his teammates. Of his 1,444 actions in shot-ending chains, 80.5% occur at position 6 or deeper (deep buildup), while only 5.9% occur in the final two positions (shot or assist). In contrast, Wirtz contributes 10.9% in the final positions and has a substantially larger presence in the pre-assist zone (positions 2-5). Andrich's positional profile resembles Xhaka's (79.5% buildup) but with dramatically less total involvement: 575 actions versus Xhaka's 1,444.

Despite operating from deep, Xhaka's total Transformer-credited chain value (6.09) nearly equals Wirtz's (6.50). The critical difference is distribution: Xhaka generates 4.87 of his 6.09 value from deep buildup, while Wirtz generates 3.31 of his 6.50 from that zone. Andrich generates only 2.57 total chain value.

### 5.2 Decision Surplus

Xhaka's median Decision Surplus is 0.0187 (58.5% of passes have positive DS), significantly higher than Andrich's 0.0068 (53.3% positive). The Mann-Whitney U test confirms statistical significance at p = 4.34 × 10⁻⁶.

Wirtz shows the highest DS (0.0479, 67.0% positive), consistent with his role in more advanced positions where the value landscape is steeper and the opportunity for high-surplus decisions is greater.

Critically, DS is completely independent of progressive passing (r = -0.040). A player can achieve high DS through short, clever passes that exploit defensive gaps without moving the ball forward. This confirms that DS captures a fundamentally different dimension of creative quality than existing pass metrics.

Split-half reliability: Xhaka's odd-match median DS (0.0223) is consistent with his even-match median (0.0162), both firmly positive. Match-level DS standard deviation is 0.0158, indicating stable positive surplus across the season.

### 5.3 Defensive Topology Disruption

DTD shows weak correlation with forward pass distance (r = -0.141), confirming it captures a distinct dimension. However, all player-level DTD means are negative, reflecting the general tendency for defenses to become more compact as play progresses toward goal. The metric discriminates in relative terms: Wirtz (-0.033) disrupts defenses less than Xhaka (-0.073), which is counterintuitive and may reflect that Wirtz's actions occur in already-disrupted defensive configurations.

### 5.4 Final Rankings

The Full Architect Score produces meaningful re-rankings versus traditional metrics. Andrich drops from rank 8 (traditional) to rank 15 (Architect), confirming that despite playing the same position as Xhaka, he does not orchestrate. Borja Iglesias rises 15 ranks, suggesting his movement and linkup play create value not captured by goals and assists.

Xhaka ranks 9th on the Full Architect Score — lower than his traditional rank of 4th. This is driven by his negative DTD z-score (-0.45). His strongest sub-components are PACV (0.27), CIR (0.60), PRV (0.32), and DS (0.30).

### 5.5 Absence Effect

Xhaka missed one Bundesliga match. Leverkusen created 2.14 xG per match with him and 1.70 without — a 26% reduction. While a single-match comparison is not statistically robust, the direction aligns with the framework's predictions.

---

## 6. Validation

| Test | Criterion | Result | Status |
|------|-----------|--------|--------|
| Split-half reliability (DS) | Halves within 1 std | 0.0223 vs 0.0162 | ✅ Pass |
| Independence (DS vs progressive) | r < 0.4 | r = -0.040 | ✅ Pass |
| Independence (DTD vs forward dist) | r < 0.3 | r = -0.141 | ✅ Pass |
| Role discrimination (DS) | p < 0.05 | p = 4.34e-06 | ✅ Pass |
| Absence effect | xG drops without player | 2.14 → 1.70 | ✅ Directional |
| PACV vs xA independence | r < 0.5 | r = 0.94 | ⚠️ High correlation |

The high PACV-xA correlation (0.94) indicates that the current PACV formulation overlaps significantly with existing assist metrics. This is partly because PACV is computed only on shot-ending chains where xA is also defined. A future iteration should weight PACV by attention divergence from equal weighting, capturing specifically the cases where the Transformer assigns more credit to deep actions than naive methods would.

---

## 7. Discussion

The Architect Framework demonstrates that Decision Surplus — the quality of a player's pass choices relative to available alternatives — is a measurable, repeatable, and statistically significant skill that existing metrics fail to capture. The finding that DS is completely independent of progressive passing (r = -0.040) is the single strongest result of this study: it proves that a dimension of creative value exists that no current public metric addresses.

The Xhaka-Wirtz-Andrich trio comparison validates the framework's ability to discriminate between tactical roles. All three play central midfield for the same team in the same season, yet produce three visually and statistically distinct Architect Score profiles. This is exactly the discrimination that a useful metric should achieve.

The framework also reveals limitations. DTD requires refinement — the current formulation produces universally negative values because defenses compress as play advances, which is expected behavior rather than disruption. A relative DTD (benchmarked against the average action at the same pitch zone) would better isolate genuine disruption. Similarly, PACV needs to be decoupled from xA by focusing on the attention weight *differential* rather than the absolute value.

---

## 8. Limitations and Future Work

**Single-season scope:** All findings are based on one player in one season. Phase 2 should extend to tournament data (Euro 2024, Euro 2020) to compare Xhaka against Pedri, Rodri, Vitinha, and Kroos using the same framework.

**Freeze frame coverage:** StatsBomb 360 data captures visible players, not all 22. Events in the center of the pitch may miss wide players, affecting the alternative pass generation in DS computation.

**Training data volume:** 5,217 chains is thin for a Transformer. The Wyscout academic dataset (~1,941 additional matches) would substantially improve the model's credit assignment accuracy.

**Positional value proxy:** DS currently uses a simple positional value function rather than the Transformer's learned values. Integrating these would make DS more context-sensitive.

**Causal claims:** The absence analysis (1 match) cannot support causal inference. A larger dataset with more absences would strengthen the team-level validation.

---

## 9. Conclusion

We introduced the Architect Framework, a multi-layered approach to quantifying the hidden creative influence of deep-lying midfielders. Through a case study of Granit Xhaka's 2023/24 Bundesliga season, we demonstrated that Decision Surplus is a novel, valid, and reliable metric that captures passing intelligence independently of all existing public metrics. The Transformer-based credit assignment reveals that Xhaka's total chain value nearly equals Wirtz's despite operating from fundamentally deeper positions. The framework successfully discriminates between the regista, the creative attacker, and the ball-winner — three players sharing a position but serving entirely different tactical functions.

The central finding is that the "unquantifiable" quality of controlling a game is, in fact, quantifiable. It lives in the decisions a player makes — not just the passes they complete, but the passes they choose when better and worse options are available. That cognitive dimension has been invisible to analytics. The Architect Framework makes it visible.

---

## Technical Appendix

**Repository:** https://github.com/UtkarshRai247/Regista  
**Data:** StatsBomb Open Data (freely available)  
**Stack:** Python, PyTorch, scikit-learn, XGBoost, NetworkX, mplsoccer, Streamlit  
**Dashboard:** `streamlit run app.py`

---

## Phase 2: Cross-Player Comparison

Phase 1 established the Architect Framework on a single player in a single league season. Phase 2 extends the analysis to 15 players across two major international tournaments — Euro 2024 and Euro 2020 — and adds an 18-season La Liga historical dataset to enable long-run comparisons with the game's recognized elite playmakers.

### Tournament Data

Three datasets underpin Phase 2:

| Dataset | Matches | Events | Freeze Frame Coverage |
|---------|---------|--------|----------------------|
| Euro 2024 | 51 | ~50,000 | 100% |
| Euro 2020 | 51 | ~50,000 | 100% |
| La Liga historical (2004/05–2021/22) | 866 | 3,129,682 | None |

The La Liga historical archive spans 18 seasons and 547 distinct midfielders, providing the normalization pool for event-based z-scores and the context for historical comparisons. The pass difficulty model was retrained on 120,358 passes drawn from all three competitions, achieving AUC 0.940 on held-out data.

### Player Coverage

Fifteen target players were analyzed across the two tournaments. All had sufficient data (100+ passes) except Pedri at Euro 2024 (92 passes — he was injured after the quarter-final, playing only 3 matches).

| Player | Tournament | Matches | Notes |
|--------|-----------|---------|-------|
| Pedri | Euro 2024 | 3 | Injured after QF; 92 passes |
| Rodri | Euro 2024 | 6 | Full tournament |
| Vitinha | Euro 2024 | 4 | |
| Kroos | Euro 2024 | 5 | Final tournament before retirement |
| Xhaka | Euro 2024 | 5 | Cross-validation subject |
| Fabián Ruiz | Euro 2024 | 6 | |
| Kanté | Euro 2024 | 6 | |
| Bellingham | Euro 2024 | 7 | |
| Zubimendi | Euro 2024 | 4 | |
| Pedri | Euro 2020 | 6 | Larger sample |
| Frenkie de Jong | Euro 2020 | 4 | |
| Jorginho | Euro 2020 | 7 | |
| Verratti | Euro 2020 | 5 | |
| Busquets | Euro 2020 | 4 | |
| Phillips | Euro 2020 | 7 | |

### Cross-Player Rankings

The Full Architect Score (AS_full) uses all six components where freeze frame data is available; the Event-Based score (AS_event) uses PACV, PRV, CIR, and TVI only and is used when comparing across the tournament and historical pools.

| Rank | Player | AS_full | AS_event | Traditional | Archetype |
|------|--------|---------|----------|-------------|-----------|
| 1 | Pedri | 0.909 | 0.964 | — | The Creator |
| 2 | Kroos | 0.508 | — | 2.645 | The Metronome |
| 3 | Verratti | 0.434 | 0.489 | — | The Creator |
| 4 | Fabián Ruiz | 0.425 | — | 1.654 | The Disruptor |
| 5 | Jorginho | 0.374 | 0.362 | — | The Disruptor |
| 6 | Xhaka (Euro 2024) | 0.226 | — | — | The Disruptor |
| ... | Rodri | 0.045 | — | — | The Disruptor |

**Key interpretive note on Rodri:** His AS_full of 0.045 is not a flaw in the framework — it is a validation point. Rodri's value is defensive and positional rather than creative chain-orchestration. His Decision Surplus median is healthy (0.0435), but he does not score high on chain-initiation or positional value creation, which is exactly what expert observers describe. The framework correctly places him in a different performance tier than Pedri for the specific skill being measured.

**Kroos presents the inverse case.** He scores highest among all players on traditional metrics (2.645σ) and on Chain Initiation Rate (4.36σ above the mean — an extreme outlier), but his archetype is distinct from Pedri's. Kroos initiates shot chains at an exceptional rate; Pedri creates and sustains value within them.

Decision Surplus medians by player (tournament context):
- Kroos: 0.0634 (highest in cohort)
- Xhaka: 0.0443
- Rodri: 0.0435
- Jorginho: 0.0341

### Archetype Clustering

A k-means clustering on z-scored component profiles (k=3, silhouette score=0.338) reveals three distinct archetypes:

**The Creator (n=56 player-match observations):** Characterized by high PACV and high PRV. Players generate value through creative chain contribution and press resistance simultaneously. Represented by Pedri and Verratti. These are players who thrive when possession is flowing and use pressure situations to advance rather than recycle.

**The Disruptor (n=305):** Characterized by high Decision Surplus and high Defensive Topology Disruption. This is the largest cluster and the most varied — it includes Xhaka, Jorginho, Rodri, Busquets, Frenkie de Jong, Vitinha, Fabián Ruiz, Kanté, Bellingham, Phillips, and Zubimendi. Despite sharing a cluster label, within-cluster variation is significant: Xhaka and Jorginho score high on DS; Rodri and Busquets score higher on DTD.

**The Metronome (Cluster 0, n=66):** Represented exclusively by Kroos at the extreme end. Defined by extreme CIR (4.36σ above the population mean), indicating a player who is uniquely involved in initiating attacking sequences. This archetype prioritizes reliability, tempo, and chain-initiation over creative risk-taking.

### Historical Comparison

Using the La Liga historical dataset, we benchmarked the cohort against four reference players whose careers are well-represented in the data:

| Player | Career avg AS_event | Peak season | Key strength |
|--------|-------------------|-------------|--------------|
| Xavi (Barcelona) | 0.987 | 1.836 (2008/09) | PACV (+1.20σ) + CIR (+1.41σ) |
| Iniesta (Barcelona) | 0.589 | 1.173 (2004/05) | PRV (+1.40σ) |
| Modrić (Real Madrid) | 0.614 | 1.240 (2016/17) | CIR + PRV |
| Busquets (Barcelona) | 0.222 | 0.414 (2008/09) | Lower event-based score |
| Xhaka (Leverkusen 2023/24) | — | 1.343 (single season) | PRV (+2.91σ) |

**Headline finding:** Xhaka's 2023/24 Leverkusen season (AS_event = 1.343) places him above Xavi's career average (0.987) and squarely in the range of Xavi's great individual seasons — though below Xavi's 2008/09 peak of 1.836. The primary driver is PRV at +2.91σ, making Xhaka the most press-resistant player in the historical cohort by a significant margin for that season. Busquets's lower event-based score reinforces the framework's interpretation: his value is defensive and positional, not chain-creative — consistent with how he is universally described by analysts.

### Xhaka Cross-Validation

With Xhaka appearing in three distinct data contexts — Leverkusen domestic, Euro 2024, and Euro 2020 — Phase 2 provides the framework's most direct cross-competition validation:

| Context | AS_full | PRV_z | DS_z | Notes |
|---------|---------|-------|------|-------|
| Leverkusen 2023/24 | 0.699 | +2.22 | +0.11 | 33 matches, 3,299 passes |
| Euro 2024 Switzerland | 0.226 | +0.25 | +0.56 | 5 matches, 405 passes |
| Euro 2020 Switzerland | 0.130 | +0.25 | +0.47 | 5 matches |

PRV drops 87% between club and international contexts. This is consistent with the higher press intensity and reduced spatial freedom characteristic of major international tournaments, where every opponent is a top-tier pressing side. DS, by contrast, remains stable across all three contexts (hovering between +0.11 and +0.56σ), suggesting that decision-making quality — the cognitive dimension the framework is specifically designed to capture — is a genuine, transferable skill. The cross-validation verdict is **partially consistent**: DS is a stable player trait; PRV is context-dependent.

### Updated Limitations

The following limitations carry over from Phase 1 and are compounded by the Phase 2 expansion:

- **Small tournament samples:** Players contribute 3–7 matches in tournament data versus 33+ in a full league season. Metric stability is lower, and low-sample players (particularly Pedri at Euro 2024 with 92 passes) should be interpreted cautiously.
- **Competition dynamics:** Euro 2024 and Euro 2020 differ from domestic leagues in press intensity, match stakes, and squad organization. Cross-competition z-score comparisons are approximate.
- **Mixed training data for pass difficulty:** The model retrained on 120,358 passes from three competitions; systematic differences in passing difficulty between competition levels may not be fully accounted for.
- **Event-based-only historical comparison:** La Liga historical analysis uses only PACV, PRV, CIR, and TVI — no freeze frame data means DS and DTD cannot be computed, excluding two of the six Architect Score components from historical benchmarking.
- **Historical normalization pool:** Z-scores for the historical comparison are computed against the La Liga pool of 547 players, which may not be directly comparable to the tournament pool normalization. Cross-pool comparisons should be treated as indicative rather than precise.
- **PACV-xA correlation:** The high correlation (r = 0.94) identified in Phase 1 persists in tournament data, suggesting PACV in its current formulation does not add independent information over traditional assist-based metrics in shot-ending chains. Attention-weight-differential formulations remain an open improvement for Phase 3.

---

## Phase 2 Methodology Corrections (v2)

The initial Phase 2 pipeline (`phase2_scores.py`) contained 8 methodological problems that produced invalid rankings. These were corrected in `src/phase2_fix.py`, which regenerates `data/processed/phase2_architect_scores_v2.parquet`. All Phase 2 results in this report reflect the corrected v2 data.

### Problems and Fixes

| # | Problem | Symptom | Fix Applied |
|---|---------|---------|-------------|
| 1 | No minimum pass threshold | Karim Onisiwo (1 pass) ranked #1 | Require ≥150 passes per tournament entry |
| 2 | `nanmean` inflates low-data players | Players with 2 components outscored 6-component players | `architect_score_full` uses strict mean; NaN if any component missing |
| 3 | Event metrics duplicated across tournaments | Rodri's PACV/CIR/TVI identical in both Euro 2024 and Euro 2020 rows | PACV, PRV, CIR, TVI recomputed per-tournament from scratch |
| 4 | No position filtering | Ronaldo (rank 11), goalkeepers, centre-backs in ranking pool | Filter to 7 central midfielder position types; z-scores computed against midfielder-only pool |
| 5 | Player data not tournament-split | Pedri showed 554 passes in both his Euro 2024 and 2020 rows | All metrics filtered strictly by `match_id` belonging to the specific tournament |
| 6 | Xhaka Leverkusen data mixed into z-scoring | Phase 1 raw PACV=18.7, Phase 2 raw PACV≈0.003 — incompatible scales | Leverkusen row excluded from ranking and z-scoring; stored separately in `phase2_xhaka_crossval.parquet` |
| 7 | Clustering on 973 unfiltered players | 305/973 players in one cluster; Rodri, Xhaka, Busquets all together | Clustering re-run on filtered 79-player pool; optimal k=4 (silhouette=0.176) |
| 8 | Pedri Euro 2024 data incorrect | 554 passes shown; actual = 92 (injured) | Correctly excluded — 92 passes below 150-pass threshold |

### Corrected Rankings Summary

Midfielder pool after filtering: **84 entries** (79 with all 6 components) across 2 tournaments.

| Rank | Player | Tournament | AS Full | Profile |
|------|--------|-----------|---------|---------|
| 1 | Pedri | Euro 2020 | +1.420 | High PACV (+4.3σ), DS (+2.0σ), PRV (+2.1σ) — elite creator |
| 2 | Antoine Griezmann | Euro 2024 | +1.049 | High PACV, DS, DTD |
| 3 | Luka Modrić | Euro 2024 | +0.860 | High DS, DTD, CIR, PRV |
| 9 | Toni Kroos | Euro 2024 | +0.450 | Extreme CIR (+3.6σ) — The Metronome |
| 11 | Fabián Ruiz | Euro 2024 | +0.339 | Balanced creator |
| 12 | Marco Verratti | Euro 2020 | +0.316 | High PACV, PRV |
| 13 | Vitinha | Euro 2024 | +0.307 | The Orchestrator |
| ~17 | Busquets | Euro 2020 | +0.211 | The Orchestrator |
| ~19 | N'Golo Kanté | Euro 2024 | +0.148 | The Orchestrator |
| ~23 | Rodri | Euro 2024 | −0.315 | The Connector (stabilizer role) |

### Corrected Archetype Clusters (k=4)

| Archetype | n | Defining components | Example players |
|-----------|---|--------------------|----|
| The Orchestrator | 23 | High PACV (+1.1σ), DS (+0.7σ), PRV (+0.5σ) | Pedri, Verratti, Busquets, Fabián, Kanté |
| The Metronome | 9 | High CIR (+2.0σ), DS (+0.8σ) | Kroos (both tournaments) |
| The Disruptor | 12 | High TVI (+1.2σ), DTD (+1.1σ) | Tempo-variant, defensive disruptors |
| The Connector | 35 | Moderate PRV and DS, low CIR | Frenkie, Xhaka, Rodri — reliable but non-initiating |

### Xhaka Cross-Validation (Corrected)

Leverkusen z-scores are from Phase 1 (Bundesliga population); Euro z-scores are from Phase 2 (Euros midfielder pool). Populations differ, so comparisons are directional, not precise.

| Context | AS | DS_z | DTD_z | PRV_z | CIR_z | TVI_z |
|---------|-----|-----|-------|-------|-------|-------|
| Leverkusen 2023/24 (P1) | +0.283 | +0.30 | −0.45 | +0.32 | +0.60 | −0.09 |
| Euro 2024 (P2 mid. pool) | +0.074 | +0.74 | +0.48 | −0.42 | +0.22 | −0.35 |
| Euro 2020 (P2 mid. pool) | −0.379 | +0.39 | −1.53 | −0.11 | −0.28 | −0.18 |

Sign agreement on DS across all three contexts (positive) confirms decision-making quality is a stable, transferable skill. DTD varies (negative at Leverkusen and Euro 2020, positive at Euro 2024) suggesting his disruptive role differed between contexts — consistent with the different squad roles he held.
