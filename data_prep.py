"""
Data preparation for France vs Spain World Cup prediction.
Downloads historical international football matches, computes Elo ratings and features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download historical match data
print("Downloading historical match data...")
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
try:
    df = pd.read_csv(url)
    print(f"[OK] Downloaded {len(df)} historical matches")
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
    print("Attempting alternate source...")
    url = "https://raw.githubusercontent.com/datasets/international-football-results/master/data/matches.csv"
    df = pd.read_csv(url)
    print(f"[OK] Downloaded {len(df)} historical matches")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Initialize Elo ratings
INITIAL_ELO = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 65

team_elos = {}

def get_elo(team, date):
    """Get Elo rating for a team at a given date (or most recent before date)."""
    if team not in team_elos:
        return INITIAL_ELO

    history = team_elos[team]
    if not history:
        return INITIAL_ELO

    # Find the most recent Elo before this date
    recent = [elo for match_date, elo in history if match_date <= date]
    return recent[-1] if recent else INITIAL_ELO

def update_elo(team, date, new_elo):
    """Record an Elo update for a team."""
    if team not in team_elos:
        team_elos[team] = []
    team_elos[team].append((date, new_elo))

def expected_result(elo1, elo2):
    """Calculate expected probability of team 1 winning."""
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

def goal_multiplier(goal_diff):
    """Elo adjustment based on goal difference."""
    return min(1 + (abs(goal_diff) - 1) / 8, 1.75) if abs(goal_diff) > 0 else 1.0

def calculate_result(home_goals, away_goals):
    """Return 1 for home win, 0.5 for draw, 0 for away win."""
    if home_goals > away_goals:
        return 1
    elif home_goals < away_goals:
        return 0
    return 0.5

print("\nCalculating Elo ratings...")
features_list = []

for idx, row in df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    date = row['date']
    home_goals = row['home_score']
    away_goals = row['away_score']

    # Get pre-match Elos
    home_elo = get_elo(home_team, date)
    away_elo = get_elo(away_team, date)

    # Calculate expected results
    home_expected = expected_result(home_elo, away_elo)
    away_expected = 1 - home_expected

    # Get actual result
    home_actual = calculate_result(home_goals, away_goals)
    away_actual = 1 - home_actual

    # Determine tournament (for venue/stage context)
    tournament = row.get('tournament', 'Unknown') if 'tournament' in df.columns else 'Unknown'
    is_neutral = row.get('neutral', False) if 'neutral' in df.columns else ('World Cup' in str(tournament))

    # Update Elo ratings
    goal_mult = goal_multiplier(home_goals - away_goals)
    home_elo_new = home_elo + K_FACTOR * goal_mult * (home_actual - home_expected)
    away_elo_new = away_elo + K_FACTOR * goal_mult * (away_actual - away_expected)

    update_elo(home_team, date, home_elo_new)
    update_elo(away_team, date, away_elo_new)

    # Store feature record
    features_list.append({
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_elo_pre': home_elo,
        'away_elo_pre': away_elo,
        'elo_diff': home_elo - away_elo,
        'is_neutral': is_neutral,
        'tournament': tournament,
        'home_win': 1 if home_goals > away_goals else (0 if home_goals < away_goals else 0.5)
    })

    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1} matches...")

features_df = pd.DataFrame(features_list)

# Compute rolling recent form (last 10 matches for each team)
def compute_form(df, team_col, goals_for_col, goals_against_col, window=10):
    """Compute recent form (points per game, goals for/against) for a team."""
    form_data = {}

    for team in df[team_col].unique():
        team_matches = df[(df[team_col] == team)].copy()
        team_matches = team_matches.sort_values('date').reset_index(drop=True)

        form_data[team] = []
        for idx in range(len(team_matches)):
            if idx < window:
                window_matches = team_matches[:idx+1]
            else:
                window_matches = team_matches[idx-window:idx+1]

            if len(window_matches) > 0:
                goals_for = window_matches[goals_for_col].sum()
                goals_against = window_matches[goals_against_col].sum()
                points = (window_matches[team_col + '_win'] * 3).sum() if team_col + '_win' in window_matches.columns else 0

                form_data[team].append({
                    'points_per_game': points / len(window_matches) if len(window_matches) > 0 else 0,
                    'goals_for': goals_for / len(window_matches) if len(window_matches) > 0 else 0,
                    'goals_against': goals_against / len(window_matches) if len(window_matches) > 0 else 0,
                })
            else:
                form_data[team].append({
                    'points_per_game': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                })

    return form_data

print("\nComputing recent form metrics...")
# Compute form for home and away teams
home_form_list = []
away_form_list = []

for idx, row in features_df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']

    # Get all matches up to this date
    past_matches = features_df[features_df['date'] < row['date']].copy()

    # Home team form (last 10 matches as home)
    home_matches = past_matches[past_matches['home_team'] == home_team].tail(10)
    if len(home_matches) > 0:
        home_form_list.append({
            'home_ppg': (home_matches['home_win'] * 3).sum() / len(home_matches),
            'home_gf': home_matches['home_goals'].sum() / len(home_matches),
            'home_ga': home_matches['away_goals'].sum() / len(home_matches),
        })
    else:
        home_form_list.append({'home_ppg': 0, 'home_gf': 0, 'home_ga': 0})

    # Away team form (last 10 matches as away)
    away_matches = past_matches[past_matches['away_team'] == away_team].tail(10)
    if len(away_matches) > 0:
        away_form_list.append({
            'away_ppg': (away_matches['home_win'].apply(lambda x: 0 if x == 1 else (3 if x == 0 else 1))).sum() / len(away_matches),
            'away_gf': away_matches['away_goals'].sum() / len(away_matches),
            'away_ga': away_matches['home_goals'].sum() / len(away_matches),
        })
    else:
        away_form_list.append({'away_ppg': 0, 'away_gf': 0, 'away_ga': 0})

home_form_df = pd.DataFrame(home_form_list)
away_form_df = pd.DataFrame(away_form_list)

features_df = pd.concat([features_df, home_form_df, away_form_df], axis=1)

# Save prepared data
features_df.to_csv('C:\\Users\\USER\\Documents\\Claude\\FIFA\\features.csv', index=False)
print(f"\n[OK] Saved {len(features_df)} feature records to features.csv")

# Print sample stats
print(f"\nData summary:")
print(f"  Total matches: {len(features_df)}")
print(f"  Unique teams: {features_df['home_team'].nunique()}")
print(f"  Date range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
print(f"  Elo range: {features_df['home_elo_pre'].min():.0f} - {features_df['home_elo_pre'].max():.0f}")

# Get current Elo for France and Spain
print(f"\nCurrent Elo ratings (as of latest data):")
france_elo = get_elo('France', features_df['date'].max())
spain_elo = get_elo('Spain', features_df['date'].max())
print(f"  France: {france_elo:.1f}")
print(f"  Spain: {spain_elo:.1f}")
