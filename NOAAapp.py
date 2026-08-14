from datetime import datetime, timedelta
from io import StringIO
import logging
import math
import os
import pandas as pd
import pytz
import requests
import streamlit as st
from timezonefinder import TimezoneFinder

# Configure logger for backend processing tracking
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class NOAA_WBGT_Fetcher:
  """Optimized multi-step collector for NOAA Global Hourly (ISD) data using

  NOAA Open Data Dissemination (NODD) AWS S3 buckets. Includes robust station
  finding, data validation, and local timezone-aware filtering.
  """

  def __init__(self, cache_dir="./noaa_cache"):
    self.cache_dir = cache_dir
    self.s3_base_url = "https://noaa-global-hourly-pds.s3.amazonaws.com/"
    self.station_history_url = (
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    )
    self.elevation_api_url = "https://api.open-meteo.com/v1/elevation"
    self.session = requests.Session()
    self.session.headers.update({"User-Agent": "OSHA-WBGT-Tool/1.1"})
    self.tz_finder = TimezoneFinder()
    os.makedirs(self.cache_dir, exist_ok=True)

  def _haversine_distance(self, lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth in miles."""
    R = 3959.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
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

  def get_candidate_stations(
      self, target_lat, target_lon, logs, max_radius_miles=50.0
  ):
    """Fetches and filters the master station list, prioritizing ICAO airport stations."""
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
      err = f"CRITICAL: Failed to fetch station list from NOAA: {e}"
      logs.append(f"ERROR: {err}")
      return None, err

    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df.dropna(subset=["LAT", "LON"], inplace=True)
    df = df[(df["LAT"] != 0.0) | (df["LON"] != 0.0)].copy()

    df["DIST_MILES"] = df.apply(
        lambda row: self._haversine_distance(
            target_lat, target_lon, row["LAT"], row["LON"]
        ),
        axis=1,
    )

    candidates = df[df["DIST_MILES"] <= max_radius_miles].copy()
    if len(candidates) < 10:
      logs.append(
          f"INFO: Found only {len(candidates)} stations within radius. Expanding"
          " search to 15 closest globally."
      )
      candidates = df.sort_values(by="DIST_MILES").head(15).copy()

    candidates["HAS_ICAO"] = candidates["ICAO"].apply(
        lambda x: 1 if pd.notna(x) and len(str(x).strip()) >= 3 else 0
    )
    sorted_candidates = candidates.sort_values(
        by=["HAS_ICAO", "DIST_MILES"], ascending=[False, True]
    ).reset_index(drop=True)

    logs.append(
        f"Step 2: Shortlisted {len(sorted_candidates)} viable stations (ICAO"
        " airport stations prioritized)."
    )
    return sorted_candidates, None

  def find_and_download_data(self, candidates_df, target_date, logs):
    """Iteratively tests candidate stations across target and previous year to find and download a valid dataset."""
    logs.append(
        "Step 3: Iterating through station shortlist to find a valid dataset in"
        " NOAA's S3 archive..."
    )
    target_year = target_date.year

    for _, row in candidates_df.iterrows():
      usaf = str(row.get("USAF", "")).strip().zfill(6)
      raw_wban = str(row.get("WBAN", "")).strip()

      if usaf == "999999" or raw_wban in ["", "nan", "99999"]:
        wban = "99999"
      else:
        wban = raw_wban.zfill(5)

      station_id = f"{usaf}-{wban}"
      station_name = str(row.get("STATION NAME", "UNKNOWN")).strip()
      dist_miles = row["DIST_MILES"]
      
      # NOAA S3 bucket concatenates USAF and WBAN with no hyphen separator
      file_name = f"{usaf}{wban}.csv"

      # Implement year fallback check (Target Year and Target Year - 1)
      for search_year in [target_year, target_year - 1]:
        local_path = os.path.join(self.cache_dir, f"{search_year}_{file_name}")

        if os.path.exists(local_path):
          logs.append(
              f"Cache Hit: Using existing local data for station"
              f" '{station_name}' [{dist_miles:.1f} mi]."
          )
          return local_path, station_name, station_id

        download_url = f"{self.s3_base_url}{search_year}/{file_name}"
        logs.append(
            f"Testing: '{station_name}' [{dist_miles:.1f} mi] for"
            f" {search_year} -> {download_url}"
        )

        try:
          response = self.session.head(download_url, timeout=10)
          if (
              response.status_code == 200
              and int(response.headers.get("Content-Length", 0)) > 1024
          ):
            logs.append(
                f"SUCCESS: Station '{station_name}' has a valid data file for"
                f" {search_year}. Downloading..."
            )
            full_response = self.session.get(download_url, timeout=30)
            full_response.raise_for_status()
            with open(local_path, "wb") as out_file:
              out_file.write(full_response.content)
            return local_path, station_name, station_id
          else:
            logs.append(
                f"INFO: No valid data file found for station '{station_name}'"
                f" for {search_year}."
            )
        except Exception as e:
          logs.append(
              f"WARNING: Network error checking station {station_id}: {e}"
          )

    return None, None, None

  def process_datafile(
      self,
      local_path,
      station_name,
      station_id,
      target_date,
      target_lat,
      target_lon,
      elevation_m,
      logs,
  ):
    """Parses the downloaded CSV, applies conversions, and filters to the target date and time."""
    logs.append(f"Step 4: Parsing and cleaning data from station '{station_name}'...")
    try:
      df = pd.read_csv(local_path, low_memory=False)
      df["DATE"] = pd.to_datetime(df["DATE"])
    except Exception as e:
      return None, f"Error parsing CSV file for station {station_id}: {e}"

    tz_str = self.tz_finder.timezone_at(lng=target_lon, lat=target_lat) or "UTC"
    local_tz = pytz.timezone(tz_str)
    logs.append(f"INFO: Target coordinates fall in timezone: {tz_str}")

    df.set_index("DATE", inplace=True)
    df.index = df.index.tz_localize("UTC").tz_convert(local_tz)

    local_date_str = target_date.strftime("%Y-%m-%d")
    day_df = df[df.index.strftime("%Y-%m-%d") == local_date_str].copy()

    if day_df.empty:
      return (
          None,
          f"No records found for the local date {local_date_str} at this"
          " station.",
      )

    def _parse(val, scale=10.0):
      try:
        v = float(str(val).split(",")[0])
        return None if v > 9990 else v / scale
      except (ValueError, TypeError):
        return None

    output = []
    for ts, row in day_df.iterrows():
      db_c = _parse(row.get("TMP"))
      dp_c = _parse(row.get("DEW"))
      ws_ms = _parse(row.get("WND"))
      slp_hpa = _parse(row.get("SLP"))

      rh, station_pressure = None, None
      db_f = (db_c * 9 / 5 + 32) if db_c is not None else None
      if db_c is not None and dp_c is not None:
        es = 6.112 * math.exp((17.67 * dp_c) / (dp_c + 243.5))
        e = 6.112 * math.exp((17.67 * db_c) / (db_c + 243.5))
        rh = (es / e) * 100
      if slp_hpa is not None and elevation_m is not None:
        station_pressure = slp_hpa * math.pow(
            1 - (0.0065 * elevation_m) / (slp_hpa / 3.48 + 273.15), 5.255
        )

      output.append({
          "Timestamp_Local": ts,
          "Dry_Bulb_F": round(db_f, 1) if db_f else None,
          "Relative_Humidity_Pct": round(rh, 1) if rh else None,
          "Wind_Speed_10m_ms": ws_ms,
          "Station_Pressure_hPa": (
              round(station_pressure, 1) if station_pressure else None
          ),
      })

    final_df = pd.DataFrame(output).set_index("Timestamp_Local")

    final_df = final_df.between_time("08:00", "17:00")
    if final_df.empty:
      return (
          None,
          "No data available during daylight working hours (08:00 - 17:00).",
      )

    logs.append(
        f"SUCCESS: Extracted {len(final_df)} hourly records for"
        f" {local_date_str}."
    )
    return final_df, f"Successfully processed data from station '{station_name}'."

  def get_hourly_data(self, target_lat, target_lon, target_date_str):
    """Main entrypoint to orchestrate the data fetching and processing pipeline."""
    logs = []
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    logs.append(
        f"Target -> Lat: {target_lat:.4f}, Lon: {target_lon:.4f}, Date:"
        f" {target_date_str}"
    )

    elevation_m = self.get_elevation(target_lat, target_lon, logs)

    candidates_df, err = self.get_candidate_stations(
        target_lat, target_lon, logs
    )
    if err:
      return None, err, logs

    local_path, station_name, station_id = self.find_and_download_data(
        candidates_df, target_date, logs
    )
    if not local_path:
      err_msg = (
          "CRITICAL: Unable to find a usable dataset from any nearby station"
          " in the NOAA S3 archive."
      )
      logs.append(f"ERROR: {err_msg}")
      return None, err_msg, logs

    final_df, msg = self.process_datafile(
        local_path,
        station_name,
        station_id,
        target_date,
        target_lat,
        target_lon,
        elevation_m,
        logs,
    )

    return final_df, msg, logs


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
