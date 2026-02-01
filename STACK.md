# Stack: M&A Announcement Relevance Model

Binary classifier that predicts whether a **14-flight window** is **relevant to an M&A announcement** (Corresponding Event) or not (No Corresponding Event). A window is labeled relevant when its last flight date falls within 14 days before a known announcement date for that company.

---

## High-Level Flow

```
  RAW DATA                    PREPROCESSING                    MODEL / OUTPUT
  ---------                   ------------                     ---------------
  Flights CSV  ------------>  windows.py  -------+
  M&A CSV      ------------>  ma_dates.py -------+---> FlightWindowDataset ----> RelevanceModel ----> model.pt
  Flights CSV  ------------>  airports.py -------+                                    |
  iata-icao    ------------>  (airports)  -------+                                    +---------------> confidence_plot.png
  Location CSV ------------>  embeddings.py -----+
```

```
Raw data (flights, M&A filings, location relevance)
    → Preprocessing (windows, coordinates, embeddings)
    → FlightWindowDataset (14 × 773-dim vectors per sample)
    → RelevanceModel (GRU + company embedding + MLP)
    → P(relevant) → train / plot confidence
```

---

## Data Sources

| Source | Path | Role |
|--------|------|------|
| **Flights** | `flight-radar/flight_radar_flights.csv` | Rows: source (company), date, from, to (e.g. "City (CODE)"). Sorted by company and date. |
| **M&A labels** | `filing-data/Regeneron M&A Filing Dates Analysis - ... .csv` | Announcement dates per company; column "The News (Announcement)" parsed for "Announce: DD Mon YYYY". |
| **Location relevance** | `location-relevance/location_relevance.csv` | location, company, relevance (text). Used to embed “how relevant is this city to this company.” |
| **Airport coords** | `iata-icao.csv` (repo) or OurAirports (fallback) | IATA code → lat/lon. Cached in `ml/data/airport_coordinates.csv`. |

Companies: Regeneron, AbbVie, Eli Lilly (keys: `regeneron`, `abbvie`, `eli-lilly`).

---

## Preprocessing Pipeline

```
  M&A CSV          ma_dates.py  ──(company, date)──┐
                                                   │
  Flights CSV      windows.py   ──windows_cache.parquet──┐
                                                   │     │
  Flights CSV  ──> airports.py <── iata-icao.csv   │     │     FlightWindowDataset
                   airport_coordinates.csv ─────────┼─────┼──-> (14, 773), label, company_id
                                                   │     │
  location_relevance.csv  embeddings.py             │     │
                   relevance_embeddings.pt ─────────┴─────┘
```

1. **M&A dates** (`ml/data/ma_dates.py`)  
   Parses the M&A CSV → list of `(company_key, announce_date)`. Only “Announce: …” rows; company blocks normalized via `MA_COMPANY_TO_KEY`.

2. **Flights** (`ml/data/windows.py`)  
   Load flights CSV, parse dates, sort by (source, flight_date).  
   **Windows**: For each company, sliding windows of 14 consecutive flights. **Label**: 1 if the last flight date in the window is within 0–14 days before an announce date for that company; 0 otherwise.  
   Result cached in `ml/data/windows_cache.parquet` (columns: company, window_index, flight_indices, label).

3. **Airport coordinates** (`ml/data/airports.py`)  
   Extract 3-letter codes from flight “from”/“to” (e.g. `City (CODE)`). Prefer `iata-icao.csv`; fallback: fetch OurAirports CSV and filter by those codes. Cache → `ml/data/airport_coordinates.csv`.

4. **Location relevance embeddings** (`ml/data/embeddings.py`)  
   For each (location, company) in `location_relevance.csv`, embed the `relevance` text with **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim). Cache → `ml/data/relevance_embeddings.pt`.

5. **Dataset** (`ml/data/dataset.py`)  
   **FlightWindowDataset**: each sample = one 14-flight window.  
   Per flight: 4 coords (from_lat, from_lon, to_lat, to_lon) + 1 scalar (flight date as days since ref) + 384-dim “from” embedding + 384-dim “to” embedding → **773-dim vector**.  
   Sequence: 14 × 773 → `(14, 773)`. Plus company_id (0/1/2) and binary label.  
   Uses cached windows, coords, and relevance embeddings.

---

## Model Architecture (`ml/model.py`)

```
  Input                    GRU                    Embed + head                 Output
  -----                    ---                    ------------                 ------
  (B, 14, 773)  ──────>  2-layer GRU  ──────>  (B, 128) final hidden  ──┐
  flight sequence         hidden=128            concat with company_emb  │
  (B,) company_id  ─────────────────────────>  (16-dim) ────────────────┘
                                                                         │
                                                                         v
                                              Linear(144→128) → ReLU → Dropout → Linear(128→1)
                                                                         │
                                                                         v
                                              (B,) logits  ──>  sigmoid  ──>  P(relevant)
```

- **Input**: `(batch, 14, 773)` flight sequences, `(batch,)` company IDs.
- **GRU**: 2-layer, hidden 128, dropout 0.4; processes the 14 steps → final hidden state (128-dim).
- **Company embedding**: 16-dim per company, concatenated with GRU state → 144-dim.
- **MLP head**: Linear(144 → 128) → ReLU → Dropout(0.4) → Linear(128 → 1) → logits.  
  Final layer bias initialized to -1.0 (NEG_BIAS) to reduce overconfident positives.
- **Output**: Logits; `sigmoid(logits)` = P(relevant).

---

## Training (`ml/train.py`)

- **Loss**: Weighted binary cross-entropy (BCE) with `pos_weight = fn_weight * (n_neg / n_pos)` to handle class imbalance and penalize false negatives (default `--fn-weight 5.0`).
- **Optimizer**: AdamW, default lr 1e-3, weight decay 0.02.
- **Evaluation**: 5-fold stratified (or plain) cross-validation; confusion matrix (TN, FP, FN, TP) per fold.
- **Checkpoint**: Last fold’s model state and `pos_weight` saved to `ml/data/model.pt`.

CLI: `--epochs`, `--lr`, `--batch-size`, `--fn-weight`, `--weight-decay`, `--val-ratio`, `--matrix-out`, etc.

---

## Confidence Plot (`ml/plot_confidence.py`)

Loads `ml/data/model.pt` and a validation split of the dataset. For each sample: `confidence = sigmoid(logits / temperature)`, then **scaled to [-1, 1]** as `confidence * 2 - 1` for display.  
Scatter: x = actual label (0 = No Corresponding Event, 1 = Corresponding Event), y = confidence; points jittered.  
Saves `ml/data/confidence_plot.png`. Optional `--temperature` to soften probabilities.

---

## Config (`ml/config.py`)

- Paths: flights CSV, M&A CSV, location relevance CSV, `iata-icao.csv`, and `ml/data/` cache paths.
- Constants: `WINDOW_SIZE=14`, `DAYS_BEFORE_ANNOUNCE=14`, `EMBED_DIM=384`, `FLIGHT_DIM=773`, company id mapping and M&A company name normalization.

---

## File Layout

```
ml/
  config.py           # Paths and constants
  model.py            # RelevanceModel (GRU + company embed + MLP)
  train.py            # Training loop, k-fold, weighted BCE
  plot_confidence.py  # Confidence vs relevance scatter plot
  requirements.txt    # torch, pandas, numpy, sentence-transformers, scikit-learn, matplotlib
  data/
    ma_dates.py       # Parse M&A CSV → announce dates
    windows.py        # Build 14-flight windows and labels, cache parquet
    airports.py       # Airport code → lat/lon, cache CSV
    embeddings.py     # Relevance text → 384-dim, cache .pt
    dataset.py        # FlightWindowDataset
    __init__.py       # Exposes FlightWindowDataset, load/build helpers
    airport_coordinates.csv   # Cached
    windows_cache.parquet     # Cached
    relevance_embeddings.pt   # Cached
    model.pt                  # Trained checkpoint
    confidence_plot.png       # Output of plot_confidence
```

---

## Dependencies

- **Python 3.10+** (for `|` union types if used).
- **pip**: `pip install -r ml/requirements.txt`  
  (torch, pandas, numpy, sentence-transformers, scikit-learn, matplotlib).

Run from repo root, e.g. `python -m ml.train`, `python -m ml.plot_confidence`.
