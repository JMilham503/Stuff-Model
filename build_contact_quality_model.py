"""
Stuff+ Project - Phase 9: Contact quality sub-model (hard-hit probability).

Why this step, and why classification instead of regression: Phase 2-3
showed that regressing a noisy, downstream-contaminated target directly
(delta_run_exp) produced essentially no signal. Whiff and swing probability,
framed as binary classification, both worked well. So contact quality is
framed the same way here: hard-hit probability (exit velocity >= 95 mph,
the standard modern threshold), rather than raw exit velocity regression.

Honest expectation: this grades contact quality using ONLY pitch
characteristics - no batter skill, no bat speed. That's intentional, same
as every model so far, to keep this a pure "Stuff" metric (what the PITCH
contributes) rather than blending in batter quality. Expect a result
somewhere between the near-zero delta_run_exp finding and the stronger
whiff signal - contact quality is closer to the pitch than delta_run_exp
was (less contaminated by defense/park), but still heavily driven by the
batter's swing, which this deliberately doesn't model.

Run from your terminal (venv activated):
    python build_contact_quality_model.py
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('sample_week.csv')

# --- Isolate actual balls in play with a recorded exit velocity ---
contact = df[df['description'] == 'hit_into_play'].copy()
contact = contact.dropna(subset=['launch_speed'])
contact['hard_hit'] = (contact['launch_speed'] >= 95).astype(int)

print(f"Batted balls with exit velocity recorded: {len(contact)}")
print(f"Overall hard-hit rate: {contact['hard_hit'].mean():.4f}")

feature_cols = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'release_extension'
]
contact = contact.dropna(subset=feature_cols)

type_counts = contact['pitch_type'].value_counts()
pitch_types_to_model = type_counts[type_counts >= 2000].index.tolist()
print(f"\nModeling these pitch types: {pitch_types_to_model}\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for pt in pitch_types_to_model:
    subset = contact[contact['pitch_type'] == pt]
    X = subset[feature_cols]
    y = subset['hard_hit']

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

    print(f"--- {pt} (n={len(subset)}, hard-hit rate={y.mean():.4f}) ---")
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
