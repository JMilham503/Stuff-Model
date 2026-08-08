"""
Stuff+ Project - Phase 10: Pulling a full season of data.

Why this step: everything so far has been built and validated on ~10 weeks
of the 2025 season. Scaling to the full season means bigger, more reliable
per-pitcher and per-pitch-type sample sizes across every model - especially
useful for pitch types like FS and KC that were already too thin to model
reliably at 10 weeks.

This pulls the season in monthly chunks rather than one giant request.
pybaseball's statcast() already breaks a date range into daily requests
internally, but pulling ~180 days in a single call means one network
hiccup partway through loses everything. Chunking by month, saving each
chunk to its own file, and skipping months already saved makes this
resumable - if it fails partway through, rerunning the script picks up
where it left off instead of starting over.

Note: dates below are my best estimate of the 2025 regular season (roughly
mid-March through late September). Double check against the actual 2025
schedule before a long pull and adjust season_start/season_end if needed.

Run from your terminal (venv activated) - this will take a while, likely
20-40+ minutes depending on your connection:
    python pull_full_season.py
"""

import os
import time
import pandas as pd
from pybaseball import statcast

# --- 2025 regular season - double check these dates before running ---
season_start = pd.Timestamp("2025-03-18")
season_end = pd.Timestamp("2025-09-28")

# --- Build one (start, end) pair per calendar month within the season ---
periods = pd.period_range(start=season_start, end=season_end, freq='M')
chunks = []
for period in periods:
    month_start = max(period.start_time, season_start)
    month_end = min(period.end_time, season_end)
    chunks.append((month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))

os.makedirs('season_chunks', exist_ok=True)

for start, end in chunks:
    chunk_file = f'season_chunks/statcast_{start}_to_{end}.csv'
    if os.path.exists(chunk_file):
        print(f"Already have {chunk_file}, skipping.")
        continue

    print(f"Pulling {start} to {end}...")
    try:
        chunk_df = statcast(start_dt=start, end_dt=end)
        chunk_df.to_csv(chunk_file, index=False)
        print(f"  Saved {len(chunk_df)} rows.")
    except Exception as e:
        print(f"  FAILED on {start} to {end}: {e}")
        print("  Rerun the script later to retry - already-saved months will be skipped.")

    time.sleep(2)  # small pause between chunks

# --- Combine all monthly chunks into one file ---
chunk_files = sorted(
    f'season_chunks/{f}' for f in os.listdir('season_chunks') if f.endswith('.csv')
)
print(f"\nCombining {len(chunk_files)} monthly files...")

full_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)

# Keeping the filename as sample_week.csv (misleading name, but every
# script you've already built reads this exact filename - overwriting it
# means all of them work unchanged on the full season with no edits needed)
full_df.to_csv('sample_week.csv', index=False)

print(f"\nFull season saved to sample_week.csv: {len(full_df)} total rows")
print("(Filename kept as sample_week.csv so your existing scripts still work as-is.)")
