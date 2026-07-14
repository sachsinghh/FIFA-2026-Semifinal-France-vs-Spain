# FIFA 2026 Semifinal: France vs Spain — Match Prediction

Projecting the outcome of the France vs Spain World Cup 2026 semi-final with a gradient-boosted match model trained on historical results, Elo ratings, and recent form.

> ⚠️ Educational project only. Built on public data for fun — not a betting or professional forecasting tool.

## Overview

This project builds a machine learning pipeline that estimates win / draw / loss probabilities for international football matches, then applies it to the France vs Spain semi-final (July 15, 2026).

## Method

### 1. Data
A historical match dataset carrying each game's date, venue neutrality, and running Elo ratings is engineered into a features table for every fixture.

### 2. Features
- Elo rating difference between the two sides
- Neutral-venue flag
- Points-per-game over each side's last 10 matches
- Goals-for and goals-against over each side's last 10 matches
- Home/away form averaged out to represent a neutral-venue final

### 3. Model
A gradient-boosted tree classifier (100 estimators, max depth 5) trained to output a win / draw / loss probability distribution for the home-coded side.

### 4. Validation
A time-based 90/10 split — the model is trained on older matches and tested on the most recent 10% — scored on accuracy and log-loss before trusting any output.

## Inputs: France vs Spain, by the numbers

| Metric | France | Spain |
|---|---|---|
| FIFA ranking | #2 | #3 |
| Elo rating | 1820 | 1795 |
| Goals for (avg) | 2.1 / game | 1.9 / game |
| Goals against (avg) | 0.8 / game | 0.9 / game |
| Points per game | 2.4 | 2.3 |
| Head-to-head | 8 wins | 6 wins |

## Results

Predicted match outcome probabilities:

| Outcome | Probability |
|---|---|
| France win | 44.6% |
| Draw | 15.4% |
| Spain win | 40.0% |

France edges the model by 4.6 points over Spain — a lean, not a certainty. A draw is the least likely result, as is typical in a knockout final.

## Caveats

- **Single-match variance**: a 44.6% probability still means Spain wins 2 times in 5 — this reads as a close contest, not a safe pick.
- **Squad detail not modeled**: injuries, suspensions, and tactical changes are not quantitatively captured.
- **Ratings frozen as of early July 2026**, validated on a time-based 90/10 holdout of historical matches — any squad news after this date isn't reflected.

## Repository structure

| File | Description |
|---|---|
| `data_prep.py` | Initial data cleaning and feature engineering |
| `data_prep_v2.py` | Updated/refined data preparation pipeline |
| `train_model.py` | Trains the gradient-boosted classifier and validates it |
| `predict.py` | Generates win/draw/loss probabilities from the trained model |
| `quick_predict.py` | Lightweight script for fast one-off predictions |
| `prediction_results.json` | Output predictions for the France vs Spain fixture |
| `report.html` | Rendered summary report of the analysis |

## Disclaimer

This project is for educational purposes only, built using publicly available football data. It does not account for real-time squad news, injuries, or tactical decisions, and should not be used as a basis for betting or financial decisions.

---
*Sachi Singh · FIFA World Cup Semi-Final · July 15, 2026*
