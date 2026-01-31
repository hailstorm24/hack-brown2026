# Flight history scraper

Fetches **historical flight data** for an aircraft from the [OpenSky Network API](https://openskynetwork.github.io/opensky-api/) (research / non-commercial use). Uses the [official OpenSky Python API](https://github.com/openskynetwork/opensky-api) (vendored in `opensky_api.py`).

## Usage

By **registration** (e.g. N310EL):

```bash
pip install -r requirements.txt
# For N-numbers like N310EL, also install:
pip install icao_nnumber_converter_us

python fetch_flight_history.py N310EL
```

By **ICAO24** (6-char hex):

```bash
python fetch_flight_history.py a0c3f2 --icao24
```

### Options

| Option | Description |
|--------|-------------|
| `--days N` | Query last N days (default: 730 ≈ 2 years) |
| `-o FILE` | Output JSON file (default: `<registration>_flights.json`) |
| `--no-track` | Do not fetch trajectory for each flight (faster) |
| `--icao24` | Treat input as ICAO24 hex instead of registration |

## Output

JSON with:

- `icao24`, `registration_or_icao`
- `query`: time range used
- `flights`: list of flights (firstSeen, lastSeen, estDepartureAirport, estArrivalAirport, callSign, etc.)
- Optionally `track` per flight (trajectory points) if `--no-track` is not used

## Authentication (optional)

Some environments get **403** (e.g. cloud IPs). You can use **OAuth2 client credentials** or username/password.

**OAuth2 client credentials** (recommended):

```bash
export CLIENT_ID=your_client_id
export CLIENT_SECRET=your_client_secret
python fetch_flight_history.py N310EL
# or
python fetch_flight_history.py N310EL --client-id your_client_id --client-secret your_client_secret
```

To obtain a token manually (e.g. for debugging):

```bash
export TOKEN=$(curl -s -X POST "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET" | jq -r .access_token)
```

The script sends this token in the `Authorization` header on every API request. To test the token directly:

```bash
curl -H "Authorization: Bearer $TOKEN" https://opensky-network.org/api/states/all | jq .
```

**Username/password** (legacy):

```bash
export OPENSKY_USERNAME=your_username OPENSKY_PASSWORD=your_password
python fetch_flight_history.py N310EL
# or
python fetch_flight_history.py N310EL --username your_username --password your_password
```

## Notes

- OpenSky is rate-limited; the script adds a short delay between requests.
- The API allows at most **2 partitions (days)** per request; the script chunks into 1-day windows to stay within this limit.
- For **N-numbers** like `N310EL` (alphanumeric), install `icao_nnumber_converter_us`. Numeric-only N-numbers (e.g. `N12345`) work without it.
- Data is from ADS-B; coverage and history depend on OpenSky’s network.
