from datetime import datetime, timedelta
from io import StringIO
import logging
import math
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
  """Optimized multi-step collector for NOAA Global Hourly (ISD) data using

  NOAA Open Data Dissemination (NODD) AWS S3 buckets. Automatically fetches GPS
  elevation, computes station pressure from sea level pressure, converts dry
  bulb to Fahrenheit, and calculates relative humidity from dew point.
  """

  def __init__(self, cache_dir="./noaa_cache"):
    self.cache_dir = cache_dir
    self.s3_base_url = "https://noaa-global-hourly-pds.s3.amazonaws.com/"
    self.station_history_url = (
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    )
    self.elevation_api_url = "https://api.open-meteo.com/v1/elevation"
    self.session = requests.Session()
    self.session.headers.update({"User-Agent": "OSHA-WBGT-Tool"})
    self.tz_finder = TimezoneFinder()
    os.makedirs(self.cache_dir, exist_ok=True)

  def _haversine_distance(self, lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth in miles."""
    R = 3959.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

  def get_elevation(self, lat, lon, logs):
    """Fetches terrain elevation in meters for the target coordinates."""
    try:
      response = self.session.get(
          self.elevation_api_url,
          params={"latitude": lat, "longitude": lon},
          timeout=10,
      )
      if response.status_code == 200:
        data = response.json()
        elevation = data.get("elevation", [0.0])[0]
        logs.append(
            f"Fetched GPS Elevation: {elevation:.1f} meters"
            f" ({elevation * 3.28084:.1f} ft)."
        )
        return float(elevation)
    except Exception as e:
      logs.append(
          f"WARNING: Could not fetch elevation from API ({e}). Defaulting to"
          " 0.0m."
      )
    return 0.0

  def get_filtered_candidate_stations(
      self, target_lat, target_lon, target_date_str, logs, max_radius_miles=50.0
  ):
    """Fetches master station list, filters by radius, and prioritizes ICAO airport stations."""
    logs.append(
        "Step 1: Fetching NOAA master station history list (isd-history.csv)..."
    )
    try:
      response = self.session.get(self.station_history_url, timeout=30)
      response.raise_for_status()
      df = pd.read_csv(
          StringIO(response.text),
          dtype={"USAF": str, "WBAN": str},
          low_memory=False,
      )
      logs.append(
          f"Successfully retrieved master station list ({len(df)} entries)."
      )
    except Exception as e:
      err = f"Failed to fetch station list from NOAA: {e}"
      logs.append(f"ERROR: {err}")
      return None, err

    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])
    df = df[(df["LAT"] != 0.0) | (df["LON"] != 0.0)].copy()

    df["USAF_CLEAN"] = (
        df["USAF"].astype(str).str.strip().str.split(".").str[0].str.zfill(6)
    )
    df = df[df["USAF_CLEAN"] != "999999"].copy()

    logs.append(
        f"Step 2: Calculating distances and filtering stations within"
        f" {max_radius_miles} miles..."
    )
    df["DIST_MILES"] = df.apply(
        lambda row: self._haversine_distance(
            target_lat, target_lon, row["LAT"], row["LON"]
        ),
        axis=1,
    )

    radius_df = df[df["DIST_MILES"] <= max_radius_miles].copy()
    if radius_df.empty:
      logs.append(
          "WARNING: No stations found within strict radius. Expanding search"
          " to 15 closest global stations..."
      )
      radius_df = df.sort_values(by="DIST_MILES").head(15).copy()

    radius_df["ICAO"] = radius_df["ICAO"].astype(str).str.strip()
    radius_df["HAS_ICAO"] = radius_df["ICAO"].apply(
        lambda x: 1 if x and x.upper() != "NAN" and len(x) >= 3 else 0
    )

    sorted_candidates = radius_df.sort_values(
        by=["HAS_ICAO", "DIST_MILES"], ascending=[False, True]
    ).reset_index(drop=True)

    logs.append(
        f"Shortlisted {len(sorted_candidates)} viable stations near target"
        " coordinates (ICAO airport stations prioritized)."
    )
    return sorted_candidates, None

  def get_hourly_data(self, target_lat, target_lon, target_date_str):
    """Iteratively queries NOAA S3 storage, extracts exact variables, and applies local conversions."""
    logs = []
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    target_year = target_date.year
    days_ago = (datetime.now().date() - target_date).days

    logs.append(
        f"Target Parameters -> Lat: {target_lat:.4f}, Lon:"
        f" {target_lon:.4f}, Date: {target_date_str} ({days_ago} days ago)"
    )

    elevation_m = self.get_elevation(target_lat, target_lon, logs)

    candidates_df, err = self.get_filtered_candidate_stations(
        target_lat, target_lon, target_date_str, logs
    )
    if candidates_df is None or candidates_df.empty:
      return None, err, logs

    selected_station_id = None
    selected_station_name = None
    selected_local_path = None

    logs.append(
        "Step 3: Iterating through prioritized station shortlist to test NOAA"
        " S3 archive availability..."
    )

    for _, row in candidates_df.iterrows():
      usaf = row["USAF_CLEAN"]
      raw_wban = str(row["WBAN"]).strip().split(".")[0]
      wban = "99999" if raw_wban in ["", "nan", "99999"] else raw_wban.zfill(5)

      station_id = f"{usaf}{wban}"
      station_name = str(row.get("STATION NAME", "UNKNOWN")).strip()
      icao_code = str(row.get("ICAO", ""))
      dist_miles = row["DIST_MILES"]

      file_name = f"{station_id}.csv"
      local_path = os.path.join(self.cache_dir, f"{target_year}_{file_name}")

      if os.path.exists(local_path):
        file_mod_time = datetime.fromtimestamp(
            os.path.getmtime(local_path)
        ).date()
        if target_date > file_mod_time:
          os.remove(local_path)
        else:
          logs.append(
              f"Cache Hit: Found valid cached dataset for '{station_name}'"
              f" (ID: {station_id}, ICAO: {icao_code}) [{dist_miles:.2f} mi"
              " away]."
          )
          selected_station_id = station_id
          selected_station_name = station_name
          selected_local_path = local_path
          break

      download_url = f"{self.s3_base_url}{target_year}/{file_name}"
      logs.append(
          f"Testing station '{station_name}' (ID: {station_id}, ICAO:"
          f" {icao_code}) [{dist_miles:.2f} mi away] -> {download_url}"
      )

      try:
        response = self.session.get(download_url, timeout=20)
        if response.status_code == 200 and len(response.content) > 100:
          sample_df = pd.read_csv(
              StringIO(response.text), nrows=5, low_memory=False
          )
          if (
              "TMP" in sample_df.columns
              and "DEW" in sample_df.columns
              and "WND" in sample_df.columns
          ):
            with open(local_path, "wb") as out_file:
              out_file.write(response.content)
            logs.append(
                f"SUCCESS: Validated and downloaded {len(response.content)}"
                f" bytes for station '{station_name}' (ID: {station_id})."
            )
            selected_station_id = station_id
            selected_station_name = station_name
            selected_local_path = local_path
            break
          else:
            logs.append(
                f"WARNING: Station {station_id} missing core weather columns"
                " (TMP/DEW/WND). Bypassing..."
            )
        elif response.status_code == 404:
          logs.append(
              f"WARNING: Station {station_id} not found in NOAA S3 archive for"
              f" {target_year}. Bypassing..."
          )
        else:
          logs.append(
              f"WARNING: Station {station_id} returned HTTP status"
              f" {response.status_code}. Bypassing..."
          )
      except Exception as e:
        logs.append(
            f"WARNING: Network error connecting to station {station_id}: {e}."
        )

    if not selected_local_path or not os.path.exists(selected_local_path):
      err_msg = (
          "CRITICAL ERROR: Unable to source dataset from NOAA S3 across all"
          " shortlisted stations."
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

    if window_df.empty:
      err_msg = (
          f"No records found around {target_date_str} for station"
          f" {selected_station_id}."
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
      db_c = _parse(row.get("TMP"))
      dp_c = _parse(row.get("DEW"))
      ws_ms = _parse(row.get("WND"))
      slp_hpa = _parse(row.get("SLP"))

      # 1. Convert Dry Bulb Temp from Celsius to Fahrenheit
      db_f = (
          round((db_c * 9.0 / 5.0) + 32.0, 1) if db_c is not None else None
      )

      # 2. Calculate Relative Humidity (%) from Dry Bulb and Dew Point fallback
      rh = None
      if db_c is not None and dp_c is not None:
        rh = round(
            100
            * (
                10 ** ((7.5 * dp_c) / (237.3 + dp_c))
                / 10 ** ((7.5 * db_c) / (237.3 + db_c))
            ),
            1,
        )

      # 3. Derive Station Pressure (hPa) from Sea Level Pressure (SLP) using elevation
      station_pressure = None
      if slp_hpa is not None:
        # Standard barometric reduction formula
        station_pressure = round(
            slp_hpa * (1 - (0.0065 * elevation_m) / 288.15) ** 5.255, 1
        )

      output.append({
          "Timestamp": row["DATE"],
          "Dry_Bulb_F": db_f,
          "Relative_Humidity_Pct": rh,
          "Wind_Speed_10m_ms": ws_ms,
          "Station_Pressure_hPa": station_pressure,
      })

    final_df = pd.DataFrame(output)

    tz_str = self.tz_finder.timezone_at(lng=target_lon, lat=target_lat)
    if not tz_str:
      tz_str = "UTC"
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
        f"Successfully extracted variables from station '{selected_station_name}'"
        f" (ID: {selected_station_id}, Elevation: {elevation_m}m).",
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
    "Extracts Dry Bulb (°F), Relative Humidity (%), Wind Speed 10m (m/s), and"
    " Station Pressure (hPa) directly from NOAA S3 feeds."
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
  with st.spinner("Querying elevation, stations, & fetching NOAA S3 archive..."):
    fetcher = NOAA_WBGT_Fetcher()
    wbgt_data, message, logs = fetcher.get_hourly_data(
        target_lat, target_lon, target_date.strftime("%Y-%m-%d")
    )

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
      "Adjust GPS coordinates and target date in the sidebar, then click 'Fetch"
      " NOAA Data'."
  )
