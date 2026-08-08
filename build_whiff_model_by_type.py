"""
Stuff+ Project - Phase 6: Separate whiff models per pitch type.

Why this step: Phase 5's pooled model (all pitch types + pitch_type as a
feature) improved AUC from 0.58 to 0.6257 - but the pitch_type dummy
variables dominated the feature importances, above physical features like
pfx_z and spin rate. That means much of the improvement came from the model
learning "what type of pitch is this" (whiff rate already varies a lot by
type alone) rather than what makes one pitch of a given type better than
another of the same type.

That's also the wrong shape for the real goal - grading individual pitches
within a pitcher's arsenal. A slider and a changeup aren't compared on the
same scale. This trains one model per pitch type instead, which removes the
shortcut and shows what actually drives whiffs within each type.

Run from your terminal (venv activated):
    python build_whiff_model_by_type.py
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score

# --- Load and identify swings, same as before ---
df = pd.read_csv('sample_week.csv')

swing_descriptions = [
    'foul', 'foul_tip', 'hit_into_play',
    'swinging_strike', 'swinging_strike_blocked'
]
whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked']

swings = df[df['description'].isin(swing_descriptions)].copy()
swings['whiff'] = swings['description'].isin(whiff_descriptions).astype(int)

feature_cols = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'release_extension'
]
swings = swings.dropna(subset=feature_cols)

# --- Only pitch types with enough volume for reliable per-type CV ---
type_counts = swings['pitch_type'].value_counts()
pitch_types_to_model = type_counts[type_counts >= 2000].index.tolist()
print(f"Modeling these pitch types: {pitch_types_to_model}\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for pt in pitch_types_to_model:
    subset = swings[swings['pitch_type'] == pt]
    X = subset[feature_cols]
    y = subset['whiff']

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

    print(f"--- {pt} (n={len(subset)}, whiff rate={y.mean():.4f}) ---")
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
