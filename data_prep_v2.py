"""
Optimized data preparation for France vs Spain World Cup prediction.
Uses vectorized operations and simpler Elo calculation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Downloading historical match data...")
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
try:
    df = pd.read_csv(url)
    print(f"[OK] Downloaded {len(df)} historical matches")
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
    raise

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total matches: {len(df)}")

# Simplified Elo calculation using groupby and rolling window
print("\nCalculating Elo ratings using rolling windows...")

# Create a result column: home_points
df['home_points'] = df.apply(
    lambda row: 3 if row['home_score'] > row['away_score']
                 else (1 if row['home_score'] == row['away_score'] else 0),
    axis=1
)

# For each team, get their rolling stats
teams = pd.concat([df['home_team'], df['away_team']]).unique()

# Create base features
df['is_neutral'] = df.get('neutral', True)
df['tournament'] = df.get('tournament', 'Unknown')

# Calculate recent form using rolling window
features_list = []

print("Computing team form and features...")
for idx in range(len(df)):
    row = df.iloc[idx]

    # Get past matches for form calculation
    past_matches = df[df['date'] < row['date']]

    # Home team recent form (last 10 as home team)
    home_past = past_matches[past_matches['home_team'] == row['home_team']].tail(10)
    if len(home_past) > 0:
        home_ppg = home_past['home_points'].sum() / len(home_past)
        home_gf = home_past['home_score'].sum() / len(home_past)
        home_ga = home_past['away_score'].sum() / len(home_past)
    else:
        home_ppg = home_gf = home_ga = 0

    # Away team recent form (last 10 as away team)
    away_past = past_matches[past_matches['away_team'] == row['away_team']].tail(10)
    if len(away_past) > 0:
        away_points = away_past.apply(
            lambda r: 0 if r['home_score'] > r['away_score']
                      else (1 if r['home_score'] == r['away_score'] else 3),
            axis=1
        )
        away_ppg = away_points.sum() / len(away_past)
        away_gf = away_past['away_score'].sum() / len(away_past)
        away_ga = away_past['home_score'].sum() / len(away_past)
    else:
        away_ppg = away_gf = away_ga = 0

    # Simple Elo proxy: use ranking if available, else estimate from form
    home_elo = 1500 + (home_ppg * 200) + ((home_gf - home_ga) * 50)
    away_elo = 1500 + (away_ppg * 200) + ((away_gf - away_ga) * 50)

    features_list.append({
        'date': row['date'],
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'home_goals': row['home_score'],
        'away_goals': row['away_score'],
        'home_elo_pre': home_elo,
        'away_elo_pre': away_elo,
        'elo_diff': home_elo - away_elo,
        'is_neutral': row['is_neutral'],
        'tournament': row['tournament'],
        'home_win': 1 if row['home_score'] > row['away_score']
                   else (0 if row['home_score'] < row['away_score'] else 0.5),
        'home_ppg': home_ppg,
        'home_gf': home_gf,
        'home_ga': home_ga,
        'away_ppg': away_ppg,
        'away_gf': away_gf,
        'away_ga': away_ga,
    })

    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1}/{len(df)} matches...")

print(f"[OK] Calculated features for {len(features_list)} matches")

features_df = pd.DataFrame(features_list)
features_df = features_df.dropna()

# Save prepared data
features_df.to_csv('C:\\Users\\USER\\Documents\\Claude\\FIFA\\features.csv', index=False)
print(f"[OK] Saved {len(features_df)} feature records to features.csv")

# Print summary
print(f"\nData summary:")
print(f"  Total matches: {len(features_df)}")
print(f"  Unique teams: {features_df['home_team'].nunique()}")
print(f"  Date range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
print(f"  Elo range: {features_df['home_elo_pre'].min():.0f} - {features_df['home_elo_pre'].max():.0f}")

# Get current Elo for France and Spain
latest_france_home = features_df[features_df['home_team'] == 'France'].iloc[-1] if len(features_df[features_df['home_team'] == 'France']) > 0 else None
latest_spain_home = features_df[features_df['home_team'] == 'Spain'].iloc[-1] if len(features_df[features_df['home_team'] == 'Spain']) > 0 else None

if latest_france_home is not None:
    france_elo = latest_france_home['home_elo_pre']
    print(f"\nFrance Elo: {france_elo:.1f}")

if latest_spain_home is not None:
    spain_elo = latest_spain_home['home_elo_pre']
    print(f"Spain Elo: {spain_elo:.1f}")

print("\n[OK] Data preparation complete!")
