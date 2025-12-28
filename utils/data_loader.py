# data_loader.py

import os
import requests
import rasterio
import numpy as np
import streamlit as st

CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_user_location():
    """Attempts to get user location via IP Geolocation."""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=2)
        if response.status_code == 200:
            data = response.json()
            return float(data.get('latitude')), float(data.get('longitude'))
    except Exception:
        pass # Fail silently
    return None

def load_api_key(filename="OpenTopography_API_key.txt"):
    # Check Streamlit Cloud Secrets first
    # Check Streamlit Cloud Secrets first (Handle missing secrets file gracefully)
    try:
        if hasattr(st, "secrets") and "OPENTOPOGRAPHY_API_KEY" in st.secrets:
            return st.secrets["OPENTOPOGRAPHY_API_KEY"]
    except Exception:
        pass # Secrets not found, fall back to file

    
    # Fallback to local file
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def get_cache_filename(lat, lon, width, height):
    return f"{CACHE_DIR}/dem_{lat:.4f}_{lon:.4f}_{width}_{height}.tif"

@st.cache_data
def fetch_elevation_data(lat, lon, width_km, height_km, api_key, force_refresh=False):
    filename = get_cache_filename(lat, lon, width_km, height_km)
    if force_refresh or not os.path.exists(filename):
        lat_deg = height_km / 111.0
        lon_deg = width_km / (111.0 * np.cos(np.radians(lat)))
        bounds = {'south': lat - (lat_deg / 2), 'north': lat + (lat_deg / 2), 'west': lon - (lon_deg / 2), 'east': lon + (lon_deg / 2)}
        url = "https://portal.opentopography.org/API/globaldem"
        params = {'demtype': 'COP30', 'south': bounds['south'], 'north': bounds['north'], 'west': bounds['west'], 'east': bounds['east'], 'outputFormat': 'GTiff', 'API_Key': api_key}
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 400:
                raise Exception(f"OpenTopography API returned 400 Bad Request.\n\nLikely Cause: The selected area is too small (<1.0 km).\n\nPlease increase Width/Height of the map.")
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request Failed: {e}")
    try:
        with rasterio.open(filename) as src:
            return src.read(1), src.bounds
    except rasterio.errors.RasterioIOError as e:
        raise Exception(f"Failed to read cached file: {filename}. Try a force refresh. Error: {e}")