# Pitch Grading Model: A Decomposed Approach to Stuff+

A pitch-quality grading model built on 2025 Statcast data, using a decomposed,
per-pitch-type approach rather than a single end-to-end prediction. This
README documents the full methodology, including the initial approach that
didn't work and what that failure taught the project.

## Overview

The goal: grade individual MLB pitches on quality of "stuff" — the physical
characteristics of a pitch (velocity, movement, spin, release point) —
independent of count, batter, or location. This is the same category of
metric as publicly known models like Stuff+, PitchingBot, and PLV.

Rather than predicting run value directly, this project breaks pitch outcome
into three independently modeled components:

1. **Whiff+** — given a batter swings, how likely is a miss?
2. **Swing probability** — does pitch shape alone induce a swing?
3. **Contact quality** — given contact is made, how likely is it hard-hit?

Each component is modeled separately, per pitch type, using gradient-boosted
classifiers validated with cross-validation.

## Data

- **Source:** Statcast, via `pybaseball`
- **Scope:** Full 2025 MLB regular season (~March 18 – September 28, 2025)
- **Volume:** 742,080 total pitches, league-wide
- **Features used:** `release_speed`, `release_spin_rate`, `pfx_x`, `pfx_z`,
  `release_pos_x`, `release_pos_z`, `release_extension`

Deliberately excluded: pitch location (`plate_x`/`plate_z`), count, and
batter identity. This keeps the model a pure "shape only" Stuff+ metric,
consistent with how public models separate Stuff+ (shape) from Location+
and Pitching+ (shape plus command/context).

## Methodology

### Phase 1: The naive approach — and why it failed

The first version regressed `delta_run_exp` (the change in run expectancy a
pitch produced) directly against the seven physical features, using linear
regression.

| Sample | Rows | R² |
|---|---|---|
| 1 week | ~4,000 fastballs | -0.0025 |
| ~10 weeks | 83,880 fastballs | -0.0000 |

Going from ~4,000 to ~84,000 pitches — a 20x increase — produced no
meaningful change in R². That ruled out sample size as the explanation.

### Phase 2: Diagnosing the failure

Swapping to XGBoost (to capture non-linear relationships) made things
**worse**, not better: R² dropped to -0.0023, and feature importances came
back suspiciously flat across all seven features — a classic overfitting
signature. A regularized model with 5-fold cross-validation confirmed the
real ceiling:

```
R² across 5 folds: [0.0007, -0.0003, 0.0010, 0.0003, -0.0002]
Mean R²: 0.0003 (std: 0.0005)
```

A tight, consistent, repeatable zero. Not overfitting, not noise — a real
property of the target. `delta_run_exp` folds in everything that happens
*after* the pitch is thrown (contact quality, defense, park effects, luck),
and that swamps whatever signal pitch shape provides at the single-pitch
level.

### Phase 3: Reframing the problem

This is why public pitch models decompose the problem instead of regressing
raw run value directly. The project pivoted to predicting intermediate,
more tractable binary outcomes instead — starting with whiff probability,
the outcome most directly tied to pitch shape.

### Phase 4: Per-pitch-type modeling

An early pooled model (all pitch types combined, with pitch type as a
one-hot feature) scored well (AUC 0.6257) — but the pitch-type dummy
variables dominated feature importance, above every physical feature. That
meant the model was partly taking a shortcut: learning "what type of pitch
is this" rather than what makes a good example of that type. Every
individual per-type model scored lower than the pooled number, confirming
the shortcut. All subsequent modeling was done per pitch type.

### Phase 5: Scaling to a full season

All three sub-models were built and validated on a ~10-week sample, then
rerun unchanged against the full 742,080-pitch season to confirm results
held at scale. In most cases they did. Cutters (FC) were the one pitch type
that meaningfully improved with more data across all three sub-models — a
useful revision: the earlier weak cutter results were partly a small-sample
artifact, not purely a structural limitation of the pitch type.

## Results

All models: `XGBClassifier` (n_estimators=100, max_depth=3, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0), evaluated with 5-fold
cross-validation, ROC-AUC scoring. Full season, minimum 2,000-pitch threshold
per type.

### Whiff+ (given a swing, probability of a miss)

| Pitch | n | Whiff rate | AUC | Top feature |
|---|---|---|---|---|
| FF | 112,740 | 18.6% | 0.5849 | release_speed |
| SL | 51,413 | 31.6% | 0.5586 | release_spin_rate |
| SI | 51,111 | 12.1% | 0.5680 | pfx_z (vertical movement) |
| CH | 37,589 | 29.4% | 0.5596 | pfx_z |
| FC | 26,944 | 20.5% | 0.5411 | pfx_z |
| ST | 25,053 | 30.2% | 0.5671 | release_speed |
| CU | 19,978 | 29.5% | 0.5834 | release_speed |
| FS | 12,319 | 32.8% | 0.5779 | release_extension |
| KC | 5,732 | 34.6% | **0.6119** | release_speed |

### Swing probability (shape only, no location)

| Pitch | n | Swing rate | AUC | Top feature |
|---|---|---|---|---|
| FF | 230,987 | 48.8% | 0.5394 | release_speed |
| SI | 112,252 | 45.5% | 0.5282 | release_speed |
| SL | 104,510 | 49.2% | 0.5235 | release_speed |
| CH | 74,714 | 50.3% | 0.5333 | release_pos_x |
| ST | 55,448 | 45.2% | 0.5350 | release_speed |
| FC | 54,663 | 49.3% | 0.5285 | release_speed |
| CU | 48,356 | 41.3% | 0.5594 | release_speed |
| FS | 23,661 | 52.1% | 0.5319 | release_pos_x |
| KC | 12,815 | 44.7% | **0.5715** | release_speed |
| SV | 3,604 | 43.7% | 0.5334 | release_spin_rate |

Consistently lower than the whiff model across nearly every type — expected,
since pitch location (the dominant real driver of swing decisions) is
deliberately excluded to keep this a shape-only metric.

### Contact quality (given contact, probability of a hard-hit ball, ≥95 mph)

| Pitch | n | Hard-hit rate | AUC | Top feature |
|---|---|---|---|---|
| FF | 38,112 | 47.0% | 0.5342 | pfx_x (horizontal movement) |
| SI | 23,752 | 45.0% | 0.5430 | pfx_z |
| SL | 17,850 | 37.9% | 0.5307 | pfx_x |
| CH | 14,291 | 34.1% | **0.5659** | release_speed |
| FC | 9,997 | 39.0% | 0.5256 | release_pos_x |
| ST | 8,842 | 32.3% | 0.5475 | pfx_z |
| CU | 7,176 | 37.4% | 0.5581 | release_speed |
| FS | 4,223 | 35.6% | 0.5371 | release_speed |

## Composite Stuff+

The three sub-models above are combined into a single expected-value-per-pitch
score, the way real "+" pitch models work: predict the probability of each
outcome, weight each by its real run value, and sum. Two branch points aren't
modeled per-pitch (ball vs. called strike, foul vs. ball in play) since both
require pitch location, which this project deliberately excludes — those use
each pitch type's empirical rate instead. Everywhere a real per-pitch model
exists (swing, whiff, hard-hit probability), the composite uses that model's
actual prediction.

Outcome values are computed directly from this project's own data (the true
average `delta_run_exp` per outcome type, per pitch type) rather than
published linear weight constants, keeping the model self-consistent with the
season it's built on:

| Pitch | v_take | v_whiff | v_foul | v_hardhit | v_weakhit |
|---|---|---|---|---|---|
| FF | 0.0188 | -0.1156 | -0.0397 | 0.2275 | -0.0901 |
| SI | 0.0105 | -0.1024 | -0.0435 | 0.1550 | -0.0892 |
| SL | 0.0223 | -0.1112 | -0.0365 | 0.2764 | -0.0723 |
| CH | 0.0327 | -0.1104 | -0.0374 | 0.2470 | -0.0699 |
| ST | 0.0166 | -0.1181 | -0.0329 | 0.3240 | -0.0572 |
| FC | 0.0212 | -0.1009 | -0.0436 | 0.2585 | -0.0951 |
| CU | 0.0166 | -0.1206 | -0.0321 | 0.3034 | -0.0524 |
| FS | 0.0327 | -0.1222 | -0.0349 | 0.2775 | -0.0689 |
| KC | 0.0189 | -0.1246 | -0.0330 | 0.2963 | -0.0396 |
| SV | 0.0169 | -0.1261 | -0.0323 | 0.3253 | -0.0480 |

The sign of every value holds across all ten pitch types with no exceptions —
whiffs and weak contact are always good for the pitcher, hard-hit balls and
takes are always good for the hitter — a real check on the derivation, not
an assumption. Worth noting: sinkers have by far the lowest hard-hit value
(0.1550) of any pitch type, while sweepers and slurves have the highest
(0.3240, 0.3253) — consistent with real scouting logic, since even a
hard-hit sinker tends to stay on the ground, while a hard-hit breaking ball
is usually a mistake pitch that gets fully punished.

### Composite leaderboard (top 20, minimum 50 pitches)

| Pitcher | Pitch | n | Stuff+ |
|---|---|---|---|
| Chapman, Aroldis | SI | 313 | 143.5 |
| Hader, Josh | SI | 472 | 141.1 |
| Maton, Phil | CU | 388 | 133.5 |
| Rogers, Tyler | SI | 733 | 127.9 |
| Suarez, Robert | SI | 137 | 124.0 |
| Bautista, Félix | SI | 386 | 123.2 |
| Megill, Trevor | FF | 468 | 120.9 |
| Helsley, Ryan | FF | 458 | 120.8 |
| Chapman, Aroldis | FF | 367 | 120.7 |
| Stephenson, Robert | FC | 108 | 120.5 |
| Ashcraft, Graham | SL | 480 | 119.9 |
| Gilbert, Logan | FS | 420 | 119.8 |
| Montgomery, Mason | FF | 580 | 119.7 |
| Burns, Chase | SL | 258 | 119.6 |
| Scott, Tanner | FF | 484 | 119.5 |
| Leahy, Kyle | ST | 187 | 119.5 |
| Miller, Mason | FF | 522 | 118.2 |
| Estrada, Jeremiah | FF | 706 | 118.2 |
| Sánchez, Cristopher | CH | 1096 | 118.2 |
| McLean, Nolan | CU | 117 | 118.0 |

Validation notes: several names appear on both this leaderboard and the
earlier Whiff+-only leaderboard, built through a completely different
combination path — Cristopher Sánchez's changeup, Logan Gilbert's splitter,
Mason Montgomery's fastball, Félix Bautista's sinker. Two different models
independently converging on the same names is stronger evidence than either
leaderboard alone. Tyler Rogers' appearance at #4 is a particularly clean,
explainable result: his well-known submarine arm slot shows up directly in
`release_pos_z`, one of the model's actual features, without the model ever
being told anything about his delivery specifically.

## Turning probability into a grade

Whiff probabilities are converted to a scaled grade the same way real "+"
stats work (wRC+, ERA+): 100 = league average for that pitch type, roughly
10 points per standard deviation, higher is better.

```
whiff_plus = 100 + 10 * (whiff_prob - mean_prob) / std_prob
```

Scores use out-of-fold predictions (`cross_val_predict`) so no pitch is ever
graded by a model that trained on it.

## Does it pass the smell test?

The player-level Whiff+ leaderboard was checked against known scouting
consensus rather than just trusted on AUC alone. Several results matched
independently well-known reputations:

- Cristopher Sánchez's changeup — widely regarded as one of the best in
  baseball. Top of the CH leaderboard.
- Jesús Luzardo's and Griffin Jax's sweepers — both frequently singled out
  in analytics writing for elite sweepers. Both landed near the top of ST.
- Fernando Cruz's splitter — known specifically for a dominant splitter.
  Top of FS.
- Edwin Díaz's fastball — one of the most dominant strikeout closers in the
  game. Top of FF.
- Josh Hader, Aroldis Chapman, and Félix Bautista clustering at the very
  top of the sinker leaderboard — three of the most feared swing-and-miss
  relievers of this era.

Real-world validation like this matters more than any single AUC number —
it's the difference between a model that scores well and one that's
actually measuring something real.

## Key findings

- Directly predicting run value from pitch shape produces no usable signal
  (R² ≈ 0), even at large scale. This is a real property of the target, not
  a modeling failure — confirmed via cross-validation, not assumed.
- Decomposing into binary sub-outcomes (whiff, swing, hard-hit) recovers
  real, usable, cross-validated signal in every case.
- Modeling per pitch type instead of pooling avoids a specific failure
  mode: a pooled model can inflate its own accuracy by learning to
  distinguish pitch *types* rather than learning what makes a good example
  *within* a type.
- Different pitch types are driven by different physical characteristics:
  four-seamers and sinkers lean on vertical movement, splitters and
  changeups lean on release extension/deception, cutters lean on
  horizontal movement. This matches how these pitches are actually scouted.
- Cutters are the consistently hardest pitch type to grade from shape
  alone across all three sub-models, though full-season data closed part
  of that gap.
- The composite score and the standalone Whiff+ model, built through
  different combination paths, independently agree on several of the same
  top names (Sánchez's changeup, Gilbert's splitter, Bautista's sinker).
  Two different methods landing on the same answer is stronger evidence
  than either result alone.

## Honest limitations

- **Two branch points in the composite score still use pitch-type averages,
  not per-pitch predictions** — whether a take is a ball or called strike,
  and whether contact is a foul or ball in play. Both require pitch
  location, which this project deliberately excludes. Closing this gap is
  the called-strike model listed below.
- **Swing probability is missing its biggest real driver (location) by
  design**, which caps how predictive that component can be on its own.
- **Sample sizes for rarer pitch types remain modest** even at a full
  season (e.g., KC and SV), especially for the contact-quality model,
  where ball-in-play events are a smaller slice of total pitches.
- **Batter handedness and pitch sequencing** aren't modeled either — both
  real components of a full pitch value framework, and both natural
  extensions once a called-strike model closes the location gap.

## Tech stack

- **Data acquisition:** `pybaseball` (Statcast)
- **Modeling:** `scikit-learn`, `xgboost`
- **Data processing:** `pandas`

## Repository structure

```
.
├── README.md
├── pull_full_season.py             # Pulls full-season Statcast data via pybaseball
├── build_whiff_model_by_type.py    # Sub-model 1: whiff probability given a swing
├── build_swing_model.py            # Sub-model 2: swing probability given a pitch
├── build_contact_quality_model.py  # Sub-model 3: hard-hit probability given contact
├── build_composite_stuff_score.py  # Combines all three sub-models into one Stuff+ score
├── player_stuff_plus_grades.csv    # Composite leaderboard output - small enough to commit directly
└── .gitignore                      # Excludes sample_week.csv, season_chunks/, and scored_pitches_composite.csv
```

Note: `sample_week.csv` (the pulled Statcast data) and the `season_chunks/`
folder are intentionally not tracked in this repo — they're large, easily
regenerated, and not something you want bloating a git history. Run
`pull_full_season.py` locally to produce them.

## How to run

```bash
pip install pandas xgboost scikit-learn pybaseball

# 1. Pull the data (takes 20-40+ minutes, resumable if interrupted)
python pull_full_season.py

# 2. Run each sub-model independently - all read the same sample_week.csv
python build_whiff_model_by_type.py
python build_swing_model.py
python build_contact_quality_model.py

# 3. Combine all three into one composite score (run this last)
python build_composite_stuff_score.py
```

## Future work

- Add a called-strike probability model to complete the decomposition
- Extend to a Location+/Pitching+ version that incorporates pitch location
- Automate ingestion on a scheduled cadence and expose grades via an API
- Persist scored pitches to a proper database rather than flat CSVs

## About this project

Built as a self-directed portfolio project applying a Navy public affairs
and sports analytics background — including hands-on Statcast pitch-timing
validation work with Inside Edge Scouting — toward the kind of applied
modeling work done in MLB front offices. The project was deliberately built
to document failure and revision honestly rather than presenting only a
clean final result, since the diagnostic process is as much the point as
the final numbers.
