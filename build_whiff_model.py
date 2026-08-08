"""
Stuff+ Project - Phase 4: Pivoting to a more tractable target - whiff probability.

Why this pivot: Phase 3b confirmed, via 5-fold cross-validation, that pitch
physical characteristics explain essentially none of the variance in a
single pitch's delta_run_exp (mean R^2 = 0.0003, std = 0.0005 - a tight,
consistent zero, not noise or overfitting). That's a real property of the
target, not a modeling mistake. delta_run_exp folds in everything that
happens AFTER the pitch is thrown - contact quality, defense, ballpark,
luck - and that swamps whatever signal pitch shape provides.

This is exactly why real public pitch models (PitchingBot, Stuff+, PLV)
decompose the problem instead of regressing raw run value directly. This
script builds the first, most well-established sub-model: given a swing,
does the batter miss? Pitch shape has a real, direct relationship with
whiff rate that's much less diluted by downstream events.

Uses the same sample_week.csv from before - the 'description' column
needed to identify swings and whiffs was already pulled, just unused
until now.

Run from your terminal (venv activated):
    python build_whiff_model.py
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score

# --- Load and filter to four-seam fastballs ---
df = pd.read_csv('sample_week.csv')
ff = df[df['pitch_type'] == 'FF'].copy()

# --- Isolate swings, using Statcast's pitch description field ---
swing_descriptions = [
    'foul', 'foul_tip', 'hit_into_play',
    'swinging_strike', 'swinging_strike_blocked'
]
whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked']

swings = ff[ff['description'].isin(swing_descriptions)].copy()
swings['whiff'] = swings['description'].isin(whiff_descriptions).astype(int)

print(f"Swings on four-seamers: {len(swings)}")
print(f"Whiff rate: {swings['whiff'].mean():.4f}")

# --- Same physical features as before ---
feature_cols = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'release_extension'
]

swings = swings.dropna(subset=feature_cols)
X = swings[feature_cols]
y = swings['whiff']

# --- Regularized classifier, same conservative settings as Phase 3b ---
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

# --- 5-fold cross-validation, scored on ROC-AUC (better fit for classification) ---
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')

print("\nROC-AUC across 5 folds:", [f"{s:.4f}" for s in auc_scores])
print(f"Mean ROC-AUC: {auc_scores.mean():.4f}")
print(f"Std ROC-AUC: {auc_scores.std():.4f}")

# --- Fit on full data for feature importances ---
model.fit(X, y)
print("\nFeature importances:")
importances = sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for feat, imp in importances:
    print(f"  {feat}: {imp:.4f}")
