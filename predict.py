"""
Predict France vs Spain outcome using the trained model.
"""

import pandas as pd
import pickle
import json
from datetime import datetime

# Load model and feature info
print("Loading trained model...")
with open('C:\\Users\\USER\\Documents\\Claude\\FIFA\\model.pkl', 'rb') as f:
    model_info = pickle.load(f)

model = model_info['model']
feature_cols = model_info['feature_cols']
norm_params = model_info['norm_params']

print("[OK] Model loaded")

# Load all features to compute current Elo and form
features_df = pd.read_csv('C:\\Users\\USER\\Documents\\Claude\\FIFA\\features.csv')
features_df['date'] = pd.to_datetime(features_df['date'])

# Compute latest Elo for both teams
def get_latest_elo(team_name, df):
    team_home = df[df['home_team'] == team_name].sort_values('date')
    team_away = df[df['away_team'] == team_name].sort_values('date')

    home_elo = team_home['home_elo_pre'].iloc[-1] if len(team_home) > 0 else 1500
    away_elo = team_away['away_elo_pre'].iloc[-1] if len(team_away) > 0 else 1500

    return max(home_elo, away_elo)

france_elo = get_latest_elo('France', features_df)
spain_elo = get_latest_elo('Spain', features_df)

print(f"\nCurrent Elo ratings:")
print(f"  France: {france_elo:.1f}")
print(f"  Spain: {spain_elo:.1f}")

# Compute recent form for both teams
def get_team_form(team_name, df, role='home'):
    if role == 'home':
        team_matches = df[df['home_team'] == team_name].sort_values('date').tail(10)
        ppg = (team_matches['home_win'] * 3).sum() / len(team_matches) if len(team_matches) > 0 else 0
        gf = team_matches['home_goals'].sum() / len(team_matches) if len(team_matches) > 0 else 0
        ga = team_matches['away_goals'].sum() / len(team_matches) if len(team_matches) > 0 else 0
    else:
        team_matches = df[df['away_team'] == team_name].sort_values('date').tail(10)
        points_earned = team_matches['home_win'].apply(lambda x: 0 if x == 1 else (3 if x == 0 else 1))
        ppg = points_earned.sum() / len(team_matches) if len(team_matches) > 0 else 0
        gf = team_matches['away_goals'].sum() / len(team_matches) if len(team_matches) > 0 else 0
        ga = team_matches['home_goals'].sum() / len(team_matches) if len(team_matches) > 0 else 0

    return ppg, gf, ga

france_home_ppg, france_home_gf, france_home_ga = get_team_form('France', features_df, 'home')
france_away_ppg, france_away_gf, france_away_ga = get_team_form('France', features_df, 'away')
spain_home_ppg, spain_home_gf, spain_home_ga = get_team_form('Spain', features_df, 'home')
spain_away_ppg, spain_away_gf, spain_away_ga = get_team_form('Spain', features_df, 'away')

print(f"\nRecent form (last 10 matches):")
print(f"  France (home): {france_home_ppg:.2f} PPG, {france_home_gf:.2f} GF, {france_home_ga:.2f} GA")
print(f"  France (away): {france_away_ppg:.2f} PPG, {france_away_gf:.2f} GF, {france_away_ga:.2f} GA")
print(f"  Spain (home): {spain_home_ppg:.2f} PPG, {spain_home_gf:.2f} GF, {spain_home_ga:.2f} GA")
print(f"  Spain (away): {spain_away_ppg:.2f} PPG, {spain_away_gf:.2f} GF, {spain_away_ga:.2f} GA")

# Build feature vector for prediction (France as "home" team for symmetry)
# In World Cup, both teams play neutral, so we use average form
france_ppg = (france_home_ppg + france_away_ppg) / 2
france_gf = (france_home_gf + france_away_gf) / 2
france_ga = (france_home_ga + france_away_ga) / 2

spain_ppg = (spain_home_ppg + spain_away_ppg) / 2
spain_gf = (spain_home_gf + spain_away_gf) / 2
spain_ga = (spain_home_ga + spain_away_ga) / 2

# Create feature vector (France as "home", Spain as "away")
elo_diff = france_elo - spain_elo
is_neutral = 1  # World Cup is neutral venue

features = {
    'elo_diff': elo_diff,
    'is_neutral': is_neutral,
    'home_ppg': france_ppg,
    'home_gf': france_gf,
    'home_ga': france_ga,
    'away_ppg': spain_ppg,
    'away_gf': spain_gf,
    'away_ga': spain_ga,
}

# Normalize features
feature_vector = []
for col in feature_cols:
    val = features[col]
    if col in norm_params:
        val = (val - norm_params[col]['mean']) / norm_params[col]['std']
    feature_vector.append(val)

print(f"\n--- PREDICTION INPUT ---")
print(f"Elo difference (France - Spain): {features['elo_diff']:.1f}")
print(f"Neutral venue: Yes")

# Make prediction
import numpy as np
X_pred = np.array([feature_vector])
proba = model.predict_proba(X_pred)[0]

away_win_prob = proba[0]  # Spain wins
draw_prob = proba[1]
home_win_prob = proba[2]  # France wins

print(f"\n--- PREDICTION RESULTS ---")
print(f"France Win: {home_win_prob:.3f} ({home_win_prob*100:.1f}%)")
print(f"Draw: {draw_prob:.3f} ({draw_prob*100:.1f}%)")
print(f"Spain Win: {away_win_prob:.3f} ({away_win_prob*100:.1f}%)")

# Get head-to-head record
france_spain_matches = features_df[
    ((features_df['home_team'] == 'France') & (features_df['away_team'] == 'Spain')) |
    ((features_df['home_team'] == 'Spain') & (features_df['away_team'] == 'France'))
].sort_values('date')

france_wins = 0
draws = 0
spain_wins = 0

for _, match in france_spain_matches.iterrows():
    if match['home_team'] == 'France':
        if match['home_win'] == 1:
            france_wins += 1
        elif match['home_win'] == 0:
            spain_wins += 1
        else:
            draws += 1
    else:
        if match['home_win'] == 1:
            spain_wins += 1
        elif match['home_win'] == 0:
            france_wins += 1
        else:
            draws += 1

print(f"\nHead-to-head record (France vs Spain):")
print(f"  France wins: {france_wins}")
print(f"  Draws: {draws}")
print(f"  Spain wins: {spain_wins}")
print(f"  Last match: {france_spain_matches.iloc[-1]['date'].date()} ({france_spain_matches.iloc[-1]['home_team']} {france_spain_matches.iloc[-1]['home_goals']}-{france_spain_matches.iloc[-1]['away_goals']})")

# Save results
results = {
    'match': 'France vs Spain (World Cup 2026)',
    'match_date': '2026-07-15 02:00 GMT+7',
    'predictions': {
        'france_win': float(home_win_prob),
        'draw': float(draw_prob),
        'spain_win': float(away_win_prob),
    },
    'france_elo': float(france_elo),
    'spain_elo': float(spain_elo),
    'france_recent_form': {
        'ppg': float(france_ppg),
        'gf': float(france_gf),
        'ga': float(france_ga),
    },
    'spain_recent_form': {
        'ppg': float(spain_ppg),
        'gf': float(spain_gf),
        'ga': float(spain_ga),
    },
    'h2h_record': {
        'france_wins': int(france_wins),
        'draws': int(draws),
        'spain_wins': int(spain_wins),
    },
    'model_accuracy': float(model_info['accuracy']),
    'model_logloss': float(model_info['logloss']),
}

with open('C:\\Users\\USER\\Documents\\Claude\\FIFA\\prediction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to prediction_results.json")
