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

# Configure logger for backend processing tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NOAA_WBGT_Fetcher:
    """
    Standalone collector for NOAA Global Hourly (ISD) data.
    Features auto-station discovery, local caching, RH calculation, 
    local timezone conversion, and working-hour filtering.
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
        """Calculates the great-circle distance between two points on the Earth surface."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        a = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2
        return 2 * asin(sqrt(a)) * 6371 * 0.621371

    def get_nearest_station(self, target_lat, target_lon, target_year):
        """Locates the closest active NOAA station to the provided coordinates."""
        try:
            response = self.session.get(self.station_history_url, timeout=30)
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), dtype={'USAF': str, 'WBAN': str})
        except Exception as e:
            logging.error(f"Failed to fetch station list: {e}")
            return None

        df = df.dropna(subset=['LAT', 'LON'])
        df['BEGIN'] = pd.to_numeric(df['BEGIN'], errors='coerce')
        df['END'] = pd.to_numeric(df['END'], errors='coerce')
        
        target_start = int(f"{target_year}0101")
        target_end = int(f"{target_year}1231")
        
        active_stations = df[(df['BEGIN'] <= target_end) & (df['END'] >= target_start)].copy()
        
        if active_stations.empty: 
            return None

        active_stations['DIST_MILES'] = active_stations.apply(
            lambda row: self._haversine_distance(target_lat, target_lon, row['LAT'], row['LON']), axis=1
        )
        
        closest = active_stations.loc[active_stations['DIST_MILES'].idxmin()]
        wban = '99999' if pd.isna(closest['WBAN']) or closest['WBAN'] == '' else closest['WBAN']
        
        return f"{closest['USAF']}{wban}"

    def get_hourly_data(self, target_lat, target_lon, target_date_str):
        """
        Retrieves, parses, and cleans hourly weather data. 
        Adjusts to local time based on coordinates and filters for 08:00-17:00.
        """
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        target_year = target_date.year
        days_ago = (datetime.now().date() - target_date).days
        
        station_id = self.get_nearest_station(target_lat, target_lon, target_year)
        if not station_id: 
            return None, "No active weather station found for this year and location."

        file_name = f"{station_id}.csv"
        local_path = os.path.join(self.cache_dir, f"{target_year}_{file_name}")
        
        if os.path.exists(local_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path)).date()
            if target_date > file_mod_time:
                os.remove(local_path)
        
        if not os.path.exists(local_path):
            download_url = f"{self.noaa_bulk_url}{target_year}/{file_name}"
            try:
                response = self.session.get(download_url, timeout=30)
                response.raise_for_status()
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.content)
            except Exception as e:
                return None, f"Failed to download NOAA file for station {station_id}: {e}"

        try:
            df = pd.read_csv(local_path, low_memory=False)
        except Exception as e:
            return None, f"Error parsing downloaded CSV data: {e}"

        df['DATE'] = pd.to_datetime(df['DATE'])
        
        window_start = pd.to_datetime(target_date_str) - timedelta(days=1)
        window_end = pd.to_datetime(target_date_str) + timedelta(days=2)
        window_df = df[(df['DATE'] >= window_start) & (df['DATE'] < window_end)].copy()
        
        if window_df.empty:
            warning = ""
            if days_ago < 3:
                warning = " Note: NOAA usually requires 48+ hours to quality-control and post new data."
            return None, f"No records found around {target_date_str}.{warning}"

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

        local_tz = pytz.timezone(tz_str)
        
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
        final_df.set_index('Timestamp', inplace=True)
        final_df.index = final_df.index.tz_localize('UTC').tz_convert(local_tz)

        target_date_local = target_date.strftime('%Y-%m-%d')
        try:
            final_df = final_df.loc[target_date_local]
            final_df = final_df.between_time('08:00', '17:00')
        except KeyError:
             return None, "Timezone shifting pushed all records out of the requested target date window."

        if final_df.empty:
             return None, "No data available during daylight working hours (08:00 - 17:00) for this specific date."

        return final_df, f"Success! Data mapped and localized to timezone: {tz_str}."


# ==========================================
# STREAMLIT USER INTERFACE (Root Execution)
# ==========================================

st.set_page_config(page_title="OSHA-WBGT NOAA Fetcher", page_icon="🌤️", layout="wide")

st.title("OSHA-WBGT Localized Weather Data Collector")
st.markdown("Retrieves, caches, and formats occupational weather data directly from NOAA ISD feeds to be applied to compliance reports.")

st.sidebar.header("Exposure Parameters")

target_lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=36.718, step=0.001)
target_lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-76.246, step=0.001)

# Defaults to 5 days ago to accommodate for NOAA's data processing lag period
default_date = datetime.now() - timedelta(days=5)
target_date = st.sidebar.date_input("Target Date", value=default_date)

if st.sidebar.button("Fetch NOAA Data", type="primary"):
    with st.spinner("Connecting to NOAA feeds & calculating nearest active station..."):
        fetcher = NOAA_WBGT_Fetcher()
        wbgt_data, message = fetcher.get_hourly_data(target_lat, target_lon, target_date.strftime("%Y-%m-%d"))
        
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
