import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import math
from datetime import datetime, date

st.set_page_config(page_title="NOAA LCD Extractor", layout="wide")

# =====================================================================
# HARDENED API SESSION SETUP
# =====================================================================
def get_robust_session():
    """Configures a requests session with exponential backoff to handle NOAA dropouts."""
    session = requests.Session()
    # Retry 5 times, increasing wait time between each (1s, 2s, 4s, 8s...)
    retries = Retry(
        total=5,
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)

# =====================================================================
# CACHED STATION LOOKUP
# =====================================================================
@st.cache_data(ttl=86400, show_spinner=False) # Cache for 24 hours
def find_closest_noaa_station(lat: float, lon: float, token: str):
    """Finds the closest station and caches it so we don't query it repeatedly."""
    session = get_robust_session()
    headers = {"token": token}
    extent = f"{lat-1.0},{lon-1.0},{lat+1.0},{lon+1.0}"
    
    url = "https://www.ncdc.noaa.gov/cdo-api/v2/stations"
    params = {"datasetid": "LCD", "extent": extent, "limit": 100}
    
    res = session.get(url, headers=headers, params=params, timeout=15)
    res.raise_for_status()
    
    stations = res.json().get("results", [])
    if not stations:
        return None
        
    valid_stations = []
    for stn in stations:
        if "GHCND" in stn.get("id", "").upper():
            continue
        stn["computed_dist"] = haversine_distance(lat, lon, stn["latitude"], stn["longitude"])
        valid_stations.append(stn)
        
    valid_stations.sort(key=lambda x: x["computed_dist"])
    return valid_stations[0] if valid_stations else None

# =====================================================================
# DATA FETCHING PIPELINE
# =====================================================================
def fetch_station_data(station_id: str, target_date: date, token: str):
    session = get_robust_session()
    headers = {"token": token}
    date_str = target_date.strftime("%Y-%m-%d")
    
    url = "https://www.ncdc.noaa.gov/cdo-api/v2/data"
    params = {
        "datasetid": "LCD",
        "stationid": station_id,
        "startdate": f"{date_str}T00:00:00",
        "enddate": f"{date_str}T23:59:59",
        "limit": 1000,
        "units": "standard"
    }
    
    res = session.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    return res.json().get("results", [])

# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("📡 NOAA LCD Diagnostic Extractor")
st.markdown("Isolated microservice for testing and extracting raw NOAA Local Climatological Data.")

with st.sidebar:
    st.header("Query Parameters")
    noaa_token = st.text_input("NOAA CDO API Token", type="password")
    target_lat = st.number_input("Latitude", value=29.6129, format="%.4f")
    target_lon = st.number_input("Longitude", value=-98.3244, format="%.4f")
    target_date = st.date_input("Target Date", value=date.today())
    
    execute = st.button("Fetch NOAA Data", type="primary", use_container_width=True)

if execute:
    if not noaa_token:
        st.error("Please provide a NOAA API Token.")
        st.stop()
        
    with st.status("Executing NOAA API Pipeline...", expanded=True) as status:
        try:
            st.write("🔍 Searching for closest valid LCD station...")
            closest_stn = find_closest_noaa_station(target_lat, target_lon, noaa_token)
            
            if not closest_stn:
                status.update(label="No stations found in range.", state="error")
                st.stop()
                
            st.write(f"✅ Found station: **{closest_stn.get('name')}** ({closest_stn.get('computed_dist')} miles away)")
            st.write("⏳ Downloading hourly data matrix (this may take up to 30 seconds)...")
            
            raw_data = fetch_station_data(closest_stn["id"], target_date, noaa_token)
            
            if not raw_data:
                status.update(label="Station found, but no data available for this date.", state="error")
                st.stop()
                
            status.update(label="Data successfully extracted!", state="complete")
            
            st.subheader("Raw Data Preview")
            df = pd.DataFrame(raw_data)
            st.dataframe(df, use_container_width=True)
            
            with st.expander("View Raw JSON Payload"):
                st.json(raw_data)
                
        except requests.exceptions.HTTPError as e:
            status.update(label="HTTP Error Occurred", state="error")
            st.error(f"Server returned an error: {e}")
        except requests.exceptions.ConnectionError:
            status.update(label="Connection Dropped", state="error")
            st.error("NOAA server forcibly closed the connection despite retries.")
        except requests.exceptions.Timeout:
            status.update(label="Timeout Exhausted", state="error")
            st.error("Request timed out entirely. The exponential backoff sequence maxed out.")
        except Exception as e:
            status.update(label="Pipeline Failure", state="error")
            st.error(f"Unexpected error: {str(e)}")
