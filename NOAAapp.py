import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from timezonefinder import TimezoneFinder
import pytz
import streamlit as st
from io import StringIO

# Configure logger for backend processing tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NOAA_WBGT_Fetcher:
    """
    Standalone collector for NOAA Global Hourly (ISD) data.
    Features auto-station discovery, local caching, RH calculation, 
    local timezone conversion, working-hour filtering, and live execution logging.
    """
    def __init__(self, cache_dir="./noaa_cache"):
        self.cache_dir = cache_dir
        self.noaa_bulk_url = "https://www.ncei.noaa.gov/data/global-hourly/access/"
        self.station_history_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'OSHA-WBGT-Tool'})
        self.tz_finder = TimezoneFinder()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates the great-circle distance between two points on the Earth surface in miles."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        a = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2
        return 2 * asin(sqrt(a)) * 6371 * 0.621371

    def get_nearest_station(self, target_lat, target_lon, target_year, logs):
        """Locates the closest active NOAA station to the provided coordinates with fallback logic."""
        logs.append("Fetching NOAA master station history list (isd-history.csv)...")
        try:
            response = self.session.get(self.station_history_url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), dtype={'USAF': str, 'WBAN': str})
            logs.append(f"Successfully retrieved master station list ({len(df)} entries).")
        except Exception as e:
            err = f"Failed to fetch station list from NOAA: {e}"
            logs.append(f"ERROR: {err}")
            return None, err

        # Clean numeric data columns
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df['BEGIN'] = pd.to_numeric(df['BEGIN'], errors='coerce')
        df['END'] = pd.to_numeric(df['END'], errors='coerce')
        
        df = df.dropna(subset=['LAT', 'LON'])
        
        target_start = int(f"{target_year}0101")
        target_end = int(f"{target_year}1231")
        
        # Look for stations active within target year or active within 2 years prior (accounting for metadata index lag)
        active_mask = (df['BEGIN'] <= target_end) & (df['END'] >= (target_year - 2) * 10000 + 101)
        active_stations = df[active_mask].copy()
        
        if active_stations.empty: 
            logs.append(f"WARNING: No strictly active stations found for metadata window. Falling back to all historical stations active prior to {target_year}.")
            active_stations = df[df['BEGIN'] <= target_end].copy()

        if active_stations.empty:
            err = f"No valid station records found in database prior to year {target_year}."
            logs.append(f"ERROR: {err}")
            return None, err

        logs.append(f"Filtering complete. Evaluating Haversine distance across {len(active_stations)} candidate stations...")
        active_stations['DIST_MILES'] = active_stations.apply(
            lambda row: self._haversine_distance(target_lat, target_lon, row['LAT'], row['LON']), axis=1
        )
        
        closest = active_stations.loc[active_stations['DIST_MILES'].idxmin()]
        
        raw_wban = str(closest['WBAN']).strip() if pd.notna(closest['WBAN']) else '99999'
        wban = '99999' if raw_wban in ['', 'nan', '99999'] else raw_wban.zfill(5)
        usaf = str(closest['USAF']).strip().zfill(6)
        
        station_id = f"{usaf}{wban}"
        station_name = closest.get('STATION NAME', 'UNKNOWN')
        dist = closest['DIST_MILES']
        
        logs.append(f"SUCCESS: Nearest station identified -> '{station_name}' (ID: {station_id}) [{dist:.2f} miles away].")
        return station_id, None

    def get_hourly_data(self, target_lat, target_lon, target_date_str):
        """
        Retrieves, parses, and cleans hourly weather data. 
        Adjusts to local time based on coordinates and filters for 08:00-17:00.
        Returns (DataFrame or None, message string, list of execution logs).
        """
        logs = []
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        target_year = target_date.year
        days_ago = (datetime.now().date() - target_date).days
        
        logs.append(f"Target Parameters -> Lat: {target_lat:.4f}, Lon: {target_lon:.4f}, Date: {target_date_str} ({days_ago} days ago)")

        station_id, err = self.get_nearest_station(target_lat, target_lon, target_year, logs)
        if not station_id: 
            return None, err, logs

        file_name = f"{station_id}.csv"
        local_path = os.path.join(self.cache_dir, f"{target_year}_{file_name}")
        
        if os.path.exists(local_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path)).date()
            if target_date > file_mod_time:
                logs.append("Cache Notice: Target date is newer than local cache modification time. Clearing local cache...")
                os.remove(local_path)
            else:
                logs.append(f"Cache Notice: Using existing cached dataset at '{local_path}'.")
        
        if not os.path.exists(local_path):
            download_url = f"{self.noaa_bulk_url}{target_year}/{file_name}"
            logs.append(f"Downloading station data from NOAA endpoint: {download_url}")
            try:
                response = self.session.get(download_url, timeout=30)
                response.raise_for_status()
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.content)
                logs.append(f"Successfully stored {len(response.content)} bytes to cache.")
            except Exception as e:
                err_msg = f"Failed to download NOAA file for station {station_id}: {e}"
                logs.append(f"ERROR: {err_msg}")
                return None, err_msg, logs

        try:
            df = pd.read_csv(local_path, low_memory=False)
            logs.append(f"Parsed CSV file successfully ({len(df)} total hourly rows).")
        except Exception as e:
            err_msg = f"Error parsing downloaded CSV data: {e}"
            logs.append(f"ERROR: {err_msg}")
            return None, err_msg, logs

        df['DATE'] = pd.to_datetime(df['DATE'])
        
        window_start = pd.to_datetime(target_date_str) - timedelta(days=1)
        window_end = pd.to_datetime(target_date_str) + timedelta(days=2)
        window_df = df[(df['DATE'] >= window_start) & (df['DATE'] < window_end)].copy()
        
        logs.append(f"Filtered UTC time window ({window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}): {len(window_df)} matching records.")

        if window_df.empty:
            warning = ""
            if days_ago < 3:
                warning = " Note: NOAA usually requires 48+ hours to quality-control and post new data."
            err_msg = f"No records found around {target_date_str}.{warning}"
            logs.append(f"WARNING: {err_msg}")
            return None, err_msg, logs

        def _parse(val, scale=10.0):
            try:
                v = float(str(val).split(',')[0])
                return None if v == 9999 else v / scale
            except: 
                return None

        output = []
        for _, row in window_df.iterrows():
            db = _parse(row.get('TMP'))
            dp = _parse(row.get('DEW'))
            ws = _parse(row.get('WND')) 
            
            rh = None
            if db is not None and dp is not None:
                rh = round(100 * (10 ** ((7.5 * dp) / (237.3 + dp)) / 10 ** ((7.5 * db) / (237.3 + db))), 1)

            output.append({
                'Timestamp': row['DATE'],
                'Dry_Bulb_C': db,
                'Relative_Humidity_Pct': rh,
                'Wind_Speed_m_s': ws,
                'Station_Pressure_hPa': _parse(row.get('SLP'))
            })

        final_df = pd.DataFrame(output)
        
        tz_str = self.tz_finder.timezone_at(lng=target_lon, lat=target_lat)
        if not tz_str:
            tz_str = "UTC"
            logs.append("Timezone warning: Could not resolve spatial timezone. Defaulting to UTC.")
        else:
            logs.append(f"Spatial Timezone Resolved: '{tz_str}'")

        local_tz = pytz.timezone(tz_str)
        
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
        final_df.set_index('Timestamp', inplace=True)
        final_df.index = final_df.index.tz_localize('UTC').tz_convert(local_tz)

        target_date_local = target_date.strftime('%Y-%m-%d')
        try:
            final_df = final_df.loc[target_date_local]
        except KeyError:
            err_msg = "Timezone shifting pushed all available records out of the requested target date window."
            logs.append(f"ERROR: {err_msg}")
            return None, err_msg, logs

        logs.append(f"Filtering for local occupational daylight hours (08:00 - 17:00)...")
        final_df = final_df.between_time('08:00', '17:00')

        if final_df.empty:
            err_msg = "No data available during daylight working hours (08:00 - 17:00) for this specific date."
            logs.append(f"WARNING: {err_msg}")
            return None, err_msg, logs

        logs.append(f"SUCCESS: Extracted {len(final_df)} hourly exposure records for {target_date_local}.")
        return final_df, f"Successfully localized and extracted data (Timezone: {tz_str}).", logs


# ==========================================
# STREAMLIT USER INTERFACE (Root Execution)
# ==========================================

st.set_page_config(page_title="OSHA-WBGT NOAA Fetcher", page_icon="🌤️", layout="wide")

st.title("OSHA-WBGT Localized Weather Data Collector")
st.markdown("Retrieves, caches, and formats occupational weather data directly from NOAA ISD feeds to be applied to compliance reports.")

st.sidebar.header("Exposure Parameters")

target_lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=38.5500,
    step=0.0001,
    format="%.4f"
)
target_lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-77.5700,
    step=0.0001,
    format="%.4f"
)

default_date = datetime.now() - timedelta(days=5)
target_date = st.sidebar.date_input("Target Date", value=default_date)

if st.sidebar.button("Fetch NOAA Data", type="primary"):
    with st.spinner("Connecting to NOAA feeds & calculating nearest active station..."):
        fetcher = NOAA_WBGT_Fetcher()
        wbgt_data, message, logs = fetcher.get_hourly_data(
            target_lat, 
            target_lon, 
            target_date.strftime("%Y-%m-%d")
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
    st.info("Adjust the GPS coordinates and target date in the sidebar, then click 'Fetch NOAA Data' to pull the weather exposure window.")
