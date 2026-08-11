# NOAA LCD API Extractor (Standalone)

A dedicated micro-tool to securely fetch, parse, and debug Local Climatological Data (LCD) from the NOAA CDO v2 API. Built to handle NOAA's high-latency endpoints using exponential backoff and localized caching.

## Local Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

*Note: You will need a valid NOAA CDO Web Services Token to query data.*
