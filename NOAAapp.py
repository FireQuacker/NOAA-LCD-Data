import pandas as pd
import requests
import numpy as np
from datetime import datetime
import io
import sys
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in miles."""
    R = 3959.0 # Radius of Earth in miles
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_isd_history():
    """Fetches the master station list from NOAA NCEI."""
    url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    print("• Fetching NOAA master station history list (isd-history.csv)...")
    try:
        df = pd.read_csv(url, dtype={'USAF': str, 'WBAN': str}, low_memory=False)
        print(f"• Successfully retrieved master station list ({len(df)} entries).")
        return df
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to retrieve isd-history.csv: {e}")
        sys.exit(1)

def get_active_candidate_stations(df, target_lat, target_lon, target_date_str):
    """Filters stations by date activity and calculates distance."""
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    target_year = target_date.year
    
    # Drop rows without coordinate data
    df = df.dropna(subset=['LAT', 'LON']).copy()
    
    # Clean and parse the END date column (Format: YYYYMMDD)
    df['END_CLEAN'] = pd.to_numeric(df['END'].astype(str).str[:8], errors='coerce')
    target_date_int = int(target_date.strftime("%Y%m%d"))
    
    # Strictly filter for stations that have data extending past our target date
    active_df = df[df['END_CLEAN'] >= target_date_int].copy()
    print(f"• Evaluating distance across {len(active_df)} candidate stations...")
    
    # Calculate geographical distance
    active_df['DISTANCE_MI'] = active_df.apply(
        lambda row: haversine_distance(target_lat, target_lon, row['LAT'], row['LON']),
        axis=1
    )
    
    # Sort by closest distance
    active_df = active_df.sort_values(by='DISTANCE_MI')
    return active_df, target_year

def fetch_noaa_s3_data(station_id, station_name, distance, year):
    """Attempts to download the hourly CSV dataset from NOAA's S3 bucket."""
    url = f"https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station_id}.csv"
    print(f"• Testing station '{station_name}' (ID: {station_id}) [{distance:.2f} mi away] -> {url}")
    
    response = requests.get(url)
    if response.status_code == 200:
        print(f"\nSUCCESS: Successfully retrieved hourly weather dataset via NOAA S3 API.")
        print(f"Extracted occupational weather exposure data for station {station_name}.")
        return pd.read_csv(io.StringIO(response.text), low_memory=False)
    else:
        print(f"  WARNING: Station {station_id} ({station_name}) not found in NOAA S3 archive for {year}. Bypassing...\n")
        return None

def main():
    # Target Parameters
    target_lat = 38.5500
    target_lon = -77.5700
    target_date_str = "2026-06-08"
    
    # Calculate days ago for logging
    target_datetime = datetime.strptime(target_date_str, "%Y-%m-%d")
    days_ago = (datetime.now() - target_datetime).days
    
    print(f"• Target Parameters -> Lat: {target_lat}, Lon: {target_lon}, Date: {target_date_str} ({days_ago} days ago)\n")
    
    # Retrieve and process station list
    history_df = get_isd_history()
    active_stations, target_year = get_active_candidate_stations(history_df, target_lat, target_lon, target_date_str)
    
    weather_data = None
    
    # Iterate through the top closest active stations
    for index, row in active_stations.head(20).iterrows():
        # Clean and pad USAF and WBAN identifiers
        usaf = str(row['USAF']).zfill(6)
        wban = str(row['WBAN']).zfill(5)
        
        # Skip placeholder identifiers
        if usaf == '999999' and wban == '99999':
            continue
            
        station_id = f"{usaf}{wban}"
        station_name = str(row['STATION NAME']).strip()
        distance = row['DISTANCE_MI']
        
        # Attempt to fetch data
        weather_data = fetch_noaa_s3_data(station_id, station_name, distance, target_year)
        
        # Break loop on first successful retrieval
        if weather_data is not None:
            break
            
    if weather_data is None:
        print("CRITICAL ERROR: Unable to source dataset from NOAA S3 across all nearby active stations.")
        sys.exit(1)
        
    # weather_data DataFrame is now populated with historic NOAA records.
    # Ready for local standard time offset conversion and daylight hour filtering.
    return weather_data

if __name__ == "__main__":
    df_weather = main()
