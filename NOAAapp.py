from datetime import datetime, timedelta
from io import StringIO
import logging
from math import asin, cos, radians, sin, sqrt
import os
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
from timezonefinder import TimezoneFinder

# Configure logger for backend processing tracking
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class NOAA_WBGT_Fetcher:
  """Standalone collector for NOAA Global Hourly (ISD) data.

  Features auto-station discovery, multi-station fallback on 404 errors,
  local caching, RH calculation, spatial timezone conversion, working-hour
  filtering, and live execution logging.
  """

  def __init__(self, cache_dir="./noaa_cache"):
    self.cache_dir = cache_dir
    self.noaa_bulk_url = "https://www.ncei.noaa.gov/data/global-hourly/access/"
    self.station_history_url = (
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    )
    self.session = requests.Session()
    self.session.headers.update({"User-Agent": "OSHA-WBGT-Tool"})
    self.tz_finder = TimezoneFinder()
    os.makedirs(self.cache_dir, exist_ok=True)

  def _haversine_distance(self, lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth in miles."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * asin(sqrt(a)) * 6371 * 0.621371

  def get_candidate_stations(
      self, target_lat, target_lon, target_year, logs, max_candidates=15
  ):
    """Retrieves and ranks the closest NOAA candidate stations by Haversine distance."""
    logs.append(
        "Fetching NOAA master station history list (isd-history.csv)..."
    )
    try:
      response = self.session.get(self.station_history_url, timeout=30)
      response.raise_for_status()
      df = pd.read_csv(
          StringIO(response.text), dtype={"USAF": str, "WBAN": str}
      )
      logs.append(
          f"Successfully retrieved master station list ({len(df)} entries)."
      )
    except Exception as e:
      err = f"Failed to fetch station list from NOAA: {e}"
      logs.append(f"ERROR: {err}")
      return None, err

    # Clean numeric coordinate and date columns
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["BEGIN"] = pd.to_numeric(df["BEGIN"], errors="coerce")
    df["END"] = pd.to_numeric(df["END"], errors="coerce")

    df = df.dropna(subset=["LAT", "LON"])

    target_start = int(f"{target_year}0101")
    target_end = int(f"{target_year}1231")

    # Filter for active stations within target year window (allowing a 2-year index lag)
    active_mask = (df["BEGIN"] <= target_end) & (
        df["END"] >= (target_year - 2) * 10000 + 101
    )
    active_stations = df[active_mask].copy()

    if active_stations.empty:
      logs.append(
          "WARNING: No strictly active stations found for target window."
          f" Falling back to all historical stations prior to {target_year}."
      )
      active_stations = df[df["BEGIN"] <= target_end].copy()

    if active_stations.empty:
      err = (
          "No valid station records found in database prior to year"
          f" {target_year}."
      )
      logs.append(f"ERROR: {err}")
      return None, err

    logs.append(
        "Evaluating distance across"
        f" {len(active_stations)} candidate stations..."
    )
    active_stations["DIST_MILES"] = active_stations.apply(
        lambda row: self._haversine_distance(
            target_lat, target_lon, row["LAT"], row["LON"]
        ),
        axis=1,
    )

    # Sort stations strictly by distance ascending
    sorted_candidates = active_stations.sort_values(
        by="DIST_MILES"
    ).head(max_candidates)
    return sorted_candidates, None

  def get_hourly_data(self, target_lat, target_lon, target_date_str):
    """Retrieves, parses, and cleans hourly weather data.

    Executes an iterative download loop over nearby candidate stations to
    bypass missing files (404s).
    Adjusts output to local time based on coordinates and filters for
    08:00-17:00.
    Returns (DataFrame or None, message string, list of execution logs).
    """
    logs = []
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    target_year = target_date.year
    days_ago = (datetime.now().date() - target_date).days

    logs.append(
        f"Target Parameters -> Lat: {target_lat:.4f}, Lon:"
        f" {target_lon:.4f}, Date: {target_date_str} ({days_ago} days ago)"
    )

    candidates_df, err = self.get_candidate_stations(
        target_lat, target_lon, target_year, logs
    )
    if candidates_df is None or candidates_df.empty:
      return None, err, logs

    selected_station_id = None
    selected_station_name = None
    selected_local_path = None

    # Fallback Loop: Iterate through candidates in order of proximity until a valid CSV is downloaded
    for _, row in candidates_df.iterrows():
      raw_usaf = str(row["USAF"]).strip().split(".")[0]
      raw_wban = str(row["WBAN"]).strip().split(".")[0]

      usaf = raw_usaf.zfill(6)
      wban = "99999" if raw_wban in ["", "nan", "99999"] else raw_wban.zfill(5)

      station_id = f"{usaf}{wban}"
      station_name = str(row.get("STATION NAME", "UNKNOWN")).strip()
      dist_miles = row["DIST_MILES"]

      file_name = f"{station_id}.csv"
      local_path = os.path.join(self.cache_dir, f"{target_year}_{file_name}")

      # Validate cache freshness
      if os.path.exists(local_path):
        file_mod_time = datetime.fromtimestamp(
            os.path.getmtime(local_path)
        ).date()
        if target_date > file_mod_time:
          logs.append(
              f"Cache Notice ({station_id}): Target date is newer than local"
              " cache modification time. Clearing local cache..."
          )
          os.remove(local_path)
        else:
          logs.append(
              f"Cache Hit: Found valid cached dataset for '{station_name}'"
              f" (ID: {station_id}) [{dist_miles:.2f} mi away]."
          )
          selected_station_id = station_id
          selected_station_name = station_name
          selected_local_path = local_path
          break

      # Attempt remote download if not cached
      download_url = f"{self.noaa_bulk_url}{target_year}/{file_name}"
      logs.append(
          f"Testing station '{station_name}' (ID: {station_id}) [{dist_miles:.2f} mi away] -> {download_url}"
      )

      try:
        response = self.session.get(download_url, timeout=20)
        if response.status_code == 200:
          with open(local_path, "wb") as out_file:
            out_file.write(response.content)
          logs.append(
              f"SUCCESS: Connected and saved {len(response.content)} bytes from"
              f" station '{station_name}' (ID: {station_id})."
          )
          selected_station_id = station_id
          selected_station_name = station_name
          selected_local_path = local_path
          break
        elif response.status_code == 404:
          logs.append(
              f"WARNING: Station {station_id} ({station_name}) returned 404"
              " Not Found on NOAA server. Bypassing to next candidate..."
          )
        else:
          logs.append(
              f"WARNING: Station {station_id} returned HTTP status"
              f" {response.status_code}. Trying next candidate..."
          )
      except Exception as e:
        logs.append(
            f"WARNING: Network error connecting to station {station_id}: {e}."
            " Retrying next candidate..."
        )

    if not selected_local_path or not os.path.exists(selected_local_path):
      err_msg = (
          "Failed to retrieve valid NOAA data from all nearby candidate"
          " stations."
      )
      logs.append(f"ERROR: {err_msg}")
      return None, err_msg, logs

    try:
      df = pd.read_csv(selected_local_path, low_memory=False)
      logs.append(
          f"Parsed CSV file for station {selected_station_id} successfully"
          f" ({len(df)} total hourly rows)."
      )
    except Exception as e:
      err_msg = f"Error parsing CSV data for station {selected_station_id}: {e}"
      logs.append(f"ERROR: {err_msg}")
      return None, err_msg, logs

    df["DATE"] = pd.to_datetime(df["DATE"])

    window_start = pd.to_datetime(target_date_str) - timedelta(days=1)
    window_end = pd.to_datetime(target_date_str) + timedelta(days=2)
    window_df = df[
        (df["DATE"] >= window_start) & (df["DATE"] < window_end)
    ].copy()

    logs.append(
        f"Filtered UTC time window ({window_start.strftime('%Y-%m-%d')} to"
        f" {window_end.strftime('%Y-%m-%d')}): {len(window_df)} matching"
        " records."
    )

    if window_df.empty:
      warning = ""
      if days_ago < 3:
        warning = (
            " Note: NOAA usually requires 48+ hours to quality-control and post"
            " new data."
        )
      err_msg = (
          f"No records found around {target_date_str} for station"
          f" {selected_station_id}.{warning}"
      )
      logs.append(f"WARNING: {err_msg}")
      return None, err_msg, logs

    def _parse(val, scale=10.0):
      try:
        v = float(str(val).split(",")[0])
        return None if v == 9999 else v / scale
      except:
        return None

    output = []
    for _, row in window_df.iterrows():
      db = _parse(row.get("TMP"))
      dp = _parse(row.get("DEW"))
      ws = _parse(row.get("WND"))

      rh = None
      if db is not None and dp is not None:
        rh = round(
            100
            * (
                10 ** ((7.5 * dp) / (237.3 + dp))
                / 10 ** ((7.5 * db) / (237.3 + db))
            ),
            1,
        )

      output.append({
          "Timestamp": row["DATE"],
          "Dry_Bulb_C": db,
          "Relative_Humidity_Pct": rh,
          "Wind_Speed_m_s": ws,
          "Station_Pressure_hPa": _parse(row.get("SLP")),
      })

    final_df = pd.DataFrame(output)

    tz_str = self.tz_finder.timezone_at(lng=target_lon, lat=target_lat)
    if not tz_str:
      tz_str = "UTC"
      logs.append(
          "Timezone warning: Could not resolve spatial timezone. Defaulting to"
          " UTC."
      )
    else:
      logs.append(f"Spatial Timezone Resolved: '{tz_str}'")

    local_tz = pytz.timezone(tz_str)

    final_df["Timestamp"] = pd.to_datetime(final_df["Timestamp"])
    final_df.set_index("Timestamp", inplace=True)
    final_df.index = final_df.index.tz_localize("UTC").tz_convert(local_tz)

    target_date_local = target_date.strftime("%Y-%m-%d")
    try:
      final_df = final_df.loc[target_date_local]
    except KeyError:
      err_msg = (
          "Timezone shifting pushed all available records out of the requested"
          " target date window."
      )
      logs.append(f"ERROR: {err_msg}")
      return None, err_msg, logs

    logs.append(
        "Filtering for local occupational daylight hours (08:00 - 17:00)..."
    )
    final_df = final_df.between_time("08:00", "17:00")

    if final_df.empty:
      err_msg = (
          "No data available during daylight working hours (08:00 - 17:00)"
          " for this specific date."
      )
      logs.append(f"WARNING: {err_msg}")
      return None, err_msg, logs

    logs.append(
        f"SUCCESS: Extracted {len(final_df)} hourly exposure records from station"
        f" '{selected_station_name}' ({selected_station_id}) for"
        f" {target_date_local}."
    )
    return (
        final_df,
        f"Successfully localized and extracted data from station"
        f" '{selected_station_name}' (ID: {selected_station_id}, Timezone:"
        f" {tz_str}).",
        logs,
    )


# ==========================================
# STREAMLIT USER INTERFACE (Root Execution)
# ==========================================

st.set_page_config(
    page_title="OSHA-WBGT NOAA Fetcher", page_icon="🌤️", layout="wide"
)

st.title("OSHA-WBGT Localized Weather Data Collector")
st.markdown(
    "Retrieves, caches, and formats occupational weather data directly from"
    " NOAA ISD feeds to be applied to compliance reports."
)

st.sidebar.header("Exposure Parameters")

target_lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=38.5500,
    step=0.0001,
    format="%.4f",
)
target_lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-77.5700,
    step=0.0001,
    format="%.4f",
)

default_date = datetime.now() - timedelta(days=5)
target_date = st.sidebar.date_input("Target Date", value=default_date)

if st.sidebar.button("Fetch NOAA Data", type="primary"):
  with st.spinner(
      "Connecting to NOAA feeds & evaluating nearest active stations..."
  ):
    fetcher = NOAA_WBGT_Fetcher()
    wbgt_data, message, logs = fetcher.get_hourly_data(
        target_lat, target_lon, target_date.strftime("%Y-%m-%d")
    )

    # Render Troubleshooting & Diagnostic Log Console
    with st.expander("🛠️ Execution Debug & Processing Log", expanded=True):
      for log in logs:
        if log.startswith("ERROR"):
          st.error(log)
        elif log.startswith("WARNING"):
          st.warning(log)
        elif log.startswith("SUCCESS"):
          st.success(log)
        else:
          st.text(f"• {log}")

    if wbgt_data is not None:
      st.success(message)
      st.dataframe(wbgt_data, use_container_width=True)

      csv = wbgt_data.to_csv()
      st.download_button(
          label="Download Exposure Data (CSV)",
          data=csv,
          file_name=f"NOAA_WBGT_Extract_{target_date}.csv",
          mime="text/csv",
      )
    else:
      st.error(message)
else:
  st.info(
      "Adjust the GPS coordinates and target date in the sidebar, then click"
      " 'Fetch NOAA Data' to pull the weather exposure window."
  )
