"""
Train a gradient boosting classifier to predict match outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Loading feature data...")
features_df = pd.read_csv('C:\\Users\\USER\\Documents\\Claude\\FIFA\\features.csv')
features_df['date'] = pd.to_datetime(features_df['date'])

# Filter out matches with missing values
features_df = features_df.dropna()
print(f"[OK] Loaded {len(features_df)} matches with complete features")

# Prepare training data
feature_cols = [
    'elo_diff',
    'is_neutral',
    'home_ppg',
    'home_gf',
    'home_ga',
    'away_ppg',
    'away_gf',
    'away_ga'
]

# Create target: encode as 0 (away win), 1 (draw), 2 (home win)
def encode_result(home_win):
    if home_win == 1:
        return 2
    elif home_win == 0:
        return 0
    else:
        return 1

features_df['result'] = features_df['home_win'].apply(encode_result)

X = features_df[feature_cols].copy()
y = features_df['result'].copy()

# Normalize numeric features (except is_neutral which is 0/1)
numeric_cols = [col for col in feature_cols if col != 'is_neutral']
for col in numeric_cols:
    mean = X[col].mean()
    std = X[col].std() + 1e-8
    X[col] = (X[col] - mean) / std

print(f"\nFeature set: {feature_cols}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Time-based split: train on older matches, test on recent
split_date = features_df['date'].quantile(0.9)
train_idx = features_df['date'] < split_date
test_idx = ~train_idx

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print(f"\nTrain/test split (90/10 by date):")
print(f"  Train: {len(X_train)} matches (up to {features_df[train_idx]['date'].max().date()})")
print(f"  Test: {len(X_test)} matches (from {features_df[test_idx]['date'].min().date()})")

# Train model
print("\nTraining Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)

print(f"\n[OK] Model trained!")
print(f"  Validation accuracy: {accuracy:.3f}")
print(f"  Validation log-loss: {logloss:.3f}")

print(f"\nClassification report (0=away win, 1=draw, 2=home win):")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# Save model and scaler params
model_info = {
    'model': model,
    'feature_cols': feature_cols,
    'numeric_cols': numeric_cols,
    'accuracy': accuracy,
    'logloss': logloss,
}

# Compute normalization params for prediction time
norm_params = {}
for col in numeric_cols:
    norm_params[col] = {
        'mean': X[col].mean() if col in X.columns else 0,
        'std': X[col].std() + 1e-8 if col in X.columns else 1,
    }
model_info['norm_params'] = norm_params

with open('C:\\Users\\USER\\Documents\\Claude\\FIFA\\model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"\n[OK] Model saved to model.pkl")
