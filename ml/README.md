# M&A announcement relevance model

Binary classifier: given 14 flights (coordinates, flight date, embedded location relevance) and company, predict whether the window is **relevant to an announcement** (last flight date within 14 days before an announce date).

## Setup

From repo root (use a virtualenv if your environment is externally managed):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r ml/requirements.txt
```

## Data

- **Labels**: Announce dates only from `filing-data/Regeneron M&A Filing Dates Analysis - Regeneron M&A Filing Dates Analysis.csv` (column "The News (Announcement)" → "Announce: DD Mon YYYY").
- **Flights**: `flight-radar/flight_radar_flights.csv` (source, date, from, to).
- **Location relevance**: `location-relevance/location_relevance.csv` (location, company, relevance text) → embedded with sentence-transformers and cached.

First run downloads OurAirports airport coordinates and builds sentence-transformers embeddings; these are cached under `ml/data/`.

## Train

From repo root:

```bash
python -m ml.train
# or
python ml/train.py
```

Options: `--epochs`, `--lr`, `--batch-size`, `--val-ratio`, `--save`, `--no-cache`.

Training uses **weighted BCE loss** (class weights for imbalance) and saves the best checkpoint to `ml/data/model.pt`.

## Plot confidence (pos vs neg)

After training, plot predicted confidence with points colored by actual label:

```bash
python -m ml.plot_confidence
# or: python ml/plot_confidence.py --model ml/data/model.pt --out ml/data/confidence_plot.png
```

Saves `ml/data/confidence_plot.png`: x-axis = actual label (0 = neg, 1 = pos), y-axis = confidence P(relevant); points jittered so they don’t stack.

## Dealing with confidence stuck at 0 or 1

The model often outputs probabilities very close to 0 or 1 (saturating). You can handle that without retraining:

1. **Temperature at inference (recommended)**  
   Use softened probabilities: `P = sigmoid(logits / T)` with `T > 1`. No retrain.
   - **Plot**: `python -m ml.plot_confidence --temperature 2` (or 1.5) so the graph shows more spread.
   - **Predictions**: When you load the model and need calibrated scores (e.g. for ranking or a threshold), apply the same: `sigmoid(logits / T)`. Tune T on a val set (e.g. T that minimizes cross-entropy or looks reasonable on the plot).

2. **Soften the decision boundary in the model**  
   In `ml/model.py`, reduce `NEG_BIAS` (e.g. from -2.0 to -1.0). The model will be less pushed to 0 for uncertain cases, so you may see more mid-range probabilities, at the cost of more false positives. Retrain after changing.

3. **Leave as-is**  
   Many trained classifiers output confident probabilities; decisions still use a 0.5 threshold. Use temperature scaling only when you need interpretable spread (e.g. for the plot or for ranking).
