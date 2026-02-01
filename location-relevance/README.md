# Location relevance (OpenAI)

Generate a six-word description of why a location is relevant to a company (e.g., why a company plane might visit), using the [OpenAI API](https://platform.openai.com/docs/api-reference). Results are cached by `(location, company)` so repeated queries reuse the stored value.

## Setup

1. **API key**: Create a `.env` file in this directory with your OpenAI API key (create one at [OpenAI](https://platform.openai.com/api-keys)):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Install dependencies** (use a virtual environment if your system Python is externally managed):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

**From Python:**
```python
from relevance import get_relevance

# First call uses OpenAI and caches the result
print(get_relevance("Teterboro (TEB)", "regeneron"))

# Same (location, company) returns cached value
print(get_relevance("Teterboro (TEB)", "regeneron"))
```

**Batch from flight CSV:**
```bash
python batch_relevance.py
```
Reads `../flight-radar/flight_radar_flights.csv`, collects unique location+company pairs, calls `get_relevance` for each (cache is used automatically), and writes `location_relevance.csv`. Ensure `.env` with `OPENAI_API_KEY` is present so OpenAI is called; otherwise rows will show an error (and that response is cached until you clear `relevance_cache.json`).

## Cache

- Cache file: `relevance_cache.json` (created in this directory at runtime).
- Add `relevance_cache.json` to `.gitignore` if you do not want to commit cached results.
