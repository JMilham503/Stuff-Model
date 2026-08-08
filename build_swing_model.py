"""
Stuff+ Project - Phase 8: Swing probability sub-model.

Why this step: Whiff+ only grades pitches the batter already decided to
swing at. A fuller Stuff+ needs a second component - does this pitch shape
induce a swing in the first place? This is deliberately built WITHOUT pitch
location (plate_x/plate_z), even though location is normally the dominant
driver of swing decisions. That's intentional: keeping location out is what
keeps this consistent with the rest of the pipeline as a pure "shape only"
Stuff+ metric, rather than sliding into a Location+ or Pitching+ style
metric that blends shape and command together.

Honest expectation: because location is being ignored on purpose, expect
this AUC to land lower than the whiff model's - possibly close to what we
saw with delta_run_exp. A weak-but-positive result means shape has some
real independent pull on swing decisions. A near-zero result means swing
decisions are close to entirely a location story - also a real, useful
finding, not a failed model.

Run from your terminal (venv activated):
    python build_swing_model.py
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('sample_week.csv')

# --- Define swings vs. clean takes (exclude ambiguous/rare events) ---
swing_descriptions = [
    'foul', 'foul_tip', 'hit_into_play',
    'swinging_strike', 'swinging_strike_blocked'
]
take_descriptions = ['ball', 'called_strike', 'blocked_ball']

pitches = df[df['description'].isin(swing_descriptions + take_descriptions)].copy()
pitches['swung'] = pitches['description'].isin(swing_descriptions).astype(int)

feature_cols = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'release_extension'
]
pitches = pitches.dropna(subset=feature_cols)

type_counts = pitches['pitch_type'].value_counts()
pitch_types_to_model = type_counts[type_counts >= 2000].index.tolist()
print(f"Modeling these pitch types: {pitch_types_to_model}\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for pt in pitch_types_to_model:
    subset = pitches[pitches['pitch_type'] == pt]
    X = subset[feature_cols]
    y = subset['swung']

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        eval_metric='logloss'
    )

    auc_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')

    print(f"--- {pt} (n={len(subset)}, swing rate={y.mean():.4f}) ---")
    print(f"Mean ROC-AUC: {auc_scores.mean():.4f} (std {auc_scores.std():.4f})")

    model.fit(X, y)
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    top3 = importances[:3]
    print("Top features:", ", ".join(f"{f} ({i:.3f})" for f, i in top3))
    print()
