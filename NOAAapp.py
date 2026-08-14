import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from timezonefinder import TimezoneFinder
import pytz

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
        return 2 * asin(sqrt(a)) * 6371 * 0.621371 # Returns distance in miles

    def get_nearest_station(self, target_lat, target_lon, target_year):
        """Locates the closest active NOAA station to the provided coordinates."""
        logging.info("Fetching NOAA master station history list...")
        try:
            response = self.session.get(self.station_history_url, timeout=30)
            response.raise_for_status()
            # Read CSV from string response
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
            logging.error("No active stations found for the target year.")
            return None

        active_stations['DIST_MILES'] = active_stations.apply(
            lambda row: self._haversine_distance(target_lat, target_lon, row['LAT'], row['LON']), axis=1
        )
        
        closest = active_stations.loc[active_stations['DIST_MILES'].idxmin()]
        wban = '99999' if pd.isna(closest['WBAN']) or closest['WBAN'] == '' else closest['WBAN']
        
        logging.info(f"Nearest Active Station: {closest['STATION NAME']} ({closest['DIST_MILES']:.1f} miles away)")
        return f"{closest['USAF']}{wban}"

    def get_hourly_data(self, target_lat, target_lon, target_date_str):
        """
        Retrieves, parses, and cleans hourly weather data. 
        Adjusts to local time based on coordinates and filters for 08:00-17:00.
        """
        # Safety Check for 48-Hour Lag
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        target_year = target_date.year
        days_ago = (datetime.now().date() - target_date).days
        
        if days_ago < 3:
            logging.warning(f"⚠️ DATE TOO RECENT: {target_date_str} is only {days_ago} days ago.")
            logging.warning("NOAA takes 48+ hours to quality-control and post hourly data. This pull may return empty.")

        # Find Nearest Station
        station_id = self.get_nearest_station(target_lat, target_lon, target_year)
        if not station_id: 
            return None

        # Download/Cache File
        file_name = f"{station_id}.csv"
        local_path = os.path.join(self.cache_dir, f"{target_year}_{file_name}")
        
        if os.path.exists(local_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path)).date()
            if target_date > file_mod_time:
                logging.info("Cached file is older than target date. Redownloading updated NOAA file...")
                os.remove(local_path)
            else:
                logging.info("Using valid cached NOAA data.")
        
        if not os.path.exists(local_path):
            download_url = f"{self.noaa_bulk_url}{target_year}/{file_name}"
            try:
                response = self.session.get(download_url, timeout=30)
                response.raise_for_status()
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.content)
            except Exception as e:
                logging.error(f"Failed to download data for station {station_id}: {e}")
                return None

        # Extract and Clean Variables
        df = pd.read_csv(local_path, low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # We cannot filter by exact date yet because NOAA is in UTC. 
        # We must pull a slightly wider window, convert timezones, then filter.
        window_start = pd.to_datetime(target_date_str) - timedelta(days=1)
        window_end = pd.to_datetime(target_date_str) + timedelta(days=2)
        window_df = df[(df['DATE'] >= window_start) & (df['DATE'] < window_end)].copy()
        
        if window_df.empty:
            logging.error(f"No records found around {target_date_str}. NOAA may still be processing.")
            return None

        def _parse(val, scale=10.0):
            """Parses the NOAA scaled string formats, returning None for missing data (9999)."""
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
                # Clausius-Clapeyron approx for Relative Humidity
                rh = round(100 * (10 ** ((7.5 * dp) / (237.3 + dp)) / 10 ** ((7.5 * db) / (237.3 + db))), 1)

            output.append({
                'Timestamp': row['DATE'],
                'Dry_Bulb_C': db,
                'Relative_Humidity_Pct': rh,
                'Wind_Speed_m_s': ws,
                'Station_Pressure_hPa': _parse(row.get('SLP'))
            })

        final_df = pd.DataFrame(output)
        
        # Determine Local Timezone from Coordinates
        tz_str = self.tz_finder.timezone_at(lng=target_lon, lat=target_lat)
        if not tz_str:
            logging.warning("Could not determine local timezone. Defaulting to UTC.")
            tz_str = "UTC"
        else:
            logging.info(f"Localized coordinates to timezone: {tz_str}")

        local_tz = pytz.timezone(tz_str)
        
        # Apply Timezone Conversions
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
        final_df.set_index('Timestamp', inplace=True)
        final_df.index = final_df.index.tz_localize('UTC').tz_convert(local_tz)

        # Filter strictly to the target date requested, post-timezone conversion
        target_date_local = target_date.strftime('%Y-%m-%d')
        final_df = final_df.loc[target_date_local]
        
        # Filter strictly for occupational daylight working hours (08:00 - 17:00)
        final_df = final_df.between_time('08:00', '17:00')

        return final_df

if __name__ == "__main__":
    fetcher = NOAA_WBGT_Fetcher()
    
    # Test execution for a regional location (Chesapeake, VA)
    TEST_LAT = 36.718
    TEST_LON = -76.246
    
    # Utilizing a date far enough in the past to ensure NOAA has processed the 48-hour lag
    test_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    print(f"Starting NOAA Data Pull for GPS({TEST_LAT}, {TEST_LON}) on {test_date}...\n")
    
    wbgt_data = fetcher.get_hourly_data(TEST_LAT, TEST_LON, test_date)
    
    if wbgt_data is not None and not wbgt_data.empty:
        print("\nSUCCESS! Parsed, localized, and filtered dataset ready for the OSHA Calculator:")
        print(wbgt_data)
    else:
        print("\nFailed to retrieve or filter valid data for this period.")
