"""
Helper functions to fetch and process real datasets for training.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Dataset cache to avoid reloading files
_data_cache: Dict[str, pd.DataFrame] = {}


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent.parent


def load_pv_data() -> pd.DataFrame:
    """
    Load PV Generation data from CSV.
    Data is in 15-minute intervals.
    
    Returns:
        DataFrame with columns: Timestamp, PV Power Generation (W)
    """
    if 'pv_data' not in _data_cache:
        root = get_project_root()
        filepath = root / 'data' / 'raw' / 'PV Generation House 1.csv'
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
        df.set_index('Timestamp', inplace=True)
        _data_cache['pv_data'] = df
        logger.info(f"Loaded PV data from {filepath}")
    return _data_cache['pv_data']


def load_price_data() -> pd.DataFrame:
    """
    Load electricity price data from CSV.
    Data is in 1-hour intervals.
    
    Returns:
        DataFrame with columns: Timestamp, Price (EUR/MWhe)
    """
    if 'price_data' not in _data_cache:
        root = get_project_root()
        filepath = root / 'data' / 'raw' / 'Price Portugal.csv'
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
        df.set_index('Timestamp', inplace=True)
        _data_cache['price_data'] = df
        logger.info(f"Loaded price data from {filepath}")
    return _data_cache['price_data']


def load_weather_data() -> pd.DataFrame:
    """
    Load weather data from CSV.
    Data is in 15-minute intervals.
    
    Returns:
        DataFrame with columns: Timestamp, Temperature, and other weather features
    """
    if 'weather_data' not in _data_cache:
        root = get_project_root()
        filepath = root / 'data' / 'raw' / 'Weather House 1.csv'
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
        df.set_index('Timestamp', inplace=True)
        _data_cache['weather_data'] = df
        logger.info(f"Loaded weather data from {filepath}")
    return _data_cache['weather_data']


def load_load_data() -> pd.DataFrame:
    """
    Load household load (consumption) data from CSV.
    Data is in 15-minute intervals.
    
    Returns:
        DataFrame with columns: DateTime, Consumption (kW)
    """
    if 'load_data' not in _data_cache:
        root = get_project_root()
        filepath = root / 'data' / 'raw' / 'Load House 1.csv'
        df = pd.read_csv(filepath)
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')
        df.set_index('DateTime', inplace=True)
        _data_cache['load_data'] = df
        logger.info(f"Loaded load data from {filepath}")
    return _data_cache['load_data']



# PV data fetching through index
def _get_Pv_value( data , idx):
    day_data = data.iloc[idx]
    
    if day_data.empty:
        logger.warning(f"No PV data found for index {idx}")
        return 0.0
    
    return float(day_data['PV Power Generation (W)'])


def pv_profile(idx : int ) -> float:

    try:
        pv_data = load_pv_data()
        # PV data is in Watts, average across all 15-min intervals in that hour
        return _get_Pv_value(pv_data, idx)
    except Exception as e:
        logger.error(f"Error loading PV profile for index {idx}: {e}")
        return 0.0

# price settings
def _get_Price_value( data , datetime):
    datetime = datetime.replace(minute=0) 
    day_data = data[data.index == datetime]
    if day_data.empty:
        logger.warning(f"No price data found for hour {datetime}")
        return 0.0
    return float(day_data['Price (EUR/MWhe)'].iloc[0])

def price_profile(idx) -> float:
    """
    Get electricity price profile for a given hour.
    
    Args:
        hour: Hour of day (0-23)
    
    Returns:
        Electricity price in EUR/MWhe -> EUR/Kwh
    """
    try:
        price_data = load_price_data()
        # use load data to get the timestamp for the given hour index
        load_data = load_load_data()
        datetime = load_data.iloc[idx].name  # Get the timestamp for the given index

        return _get_Price_value(price_data,datetime)/1000.0
    except Exception as e:
        logger.error(f"Error loading price profile for hour {idx}: {e}")
        return 0.0

def _get_Temp_value( data , datetime): 
    day_data = data[data.index == datetime]
    if day_data.empty:
        logger.warning(f"No temperature data found for hour {datetime}")
        return 10.0
    return float(day_data['Temperature'].iloc[0])

def outdoor_temperature(idx) -> float:
    """
    Get outdoor temperature for a given hour.
    
    Args:
        hour: Hour of day (0-23)
    
    Returns:
        Temperature in Celsius
    """
    try:
        weather_data = load_weather_data()
        # use load data to get the timestamp for the given hour index
        load_data = load_load_data()
        datetime = load_data.iloc[idx].name  # Get the timestamp for the given index

        return _get_Temp_value(weather_data,datetime)
    except Exception as e:
        logger.error(f"Error loading outdoor temperature for hour {idx}: {e}")
        return 10.0  # Default fallback

def _get_load_value(data, idx):
    day_data = data.iloc[idx]

    if day_data.empty:
        logger.warning(f"No load data found for index {idx}")
        return 0.0

    return float(day_data['Consumption (kW)'])


def load_profile(idx) -> float:
    try:
        load_data = load_load_data()
        return _get_load_value(load_data, idx)
    except Exception as e:
        logger.error(f"Error loading non-shiftable load profile for idx {idx}: {e}")
        return 0.3  # Default fallback


# Alternative: Get full daily profiles at once for efficiency
def get_daily_profiles(day_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Get complete daily profiles for all hours (0-23).
    More efficient if you need data for multiple hours.
    
    Args:
        day_data: Optional specific day data to use. If None, uses all available data.
    
    Returns:
        Dictionary with keys: 'pv', 'price', 'temperature', 'load'
        Each value is a numpy array of length 24 (hours)
    """
    profiles = {
        'pv': np.array([pv_profile(h) for h in range(24)]),
        'price': np.array([price_profile(h) for h in range(24)]),
        'temperature': np.array([outdoor_temperature(h) for h in range(24)]),
        'load': np.array([load_profile(h) for h in range(24)])
    }
    return profiles


def normalize_pv(pv_value: float, max_value: Optional[float] = None) -> float:
    """
    Normalize PV output to [0, 1] range.
    
    Args:
        pv_value: PV power in Watts
        max_value: Maximum expected PV output. If None, uses 90th percentile.
    
    Returns:
        Normalized PV value
    """
    try:
        if max_value is None:
            pv_data = load_pv_data()
            max_value = pv_data['PV Power Generation (W)'].quantile(0.9)
        return min(float(pv_value) / max_value, 1.0)
    except Exception as e:
        logger.error(f"Error normalizing PV: {e}")
        return 0.0

def normalize_price(price_value: float, max_value: Optional[float] = None) -> float:
    """
    Normalize price to [0, 1] range.
    
    Args:
        price_value: Price in EUR/MWhe
        max_value: Maximum expected price. If None, uses 90th percentile.
    
    Returns:
        Normalized price value
    """
    try:
        if max_value is None:
            price_data = load_price_data()
            max_value = price_data['Price (EUR/MWhe)'].quantile(0.9)/1000.0  # Convert to EUR/Kwh
        return min(float(price_value) / max_value, 1.0)
    except Exception as e:
        logger.error(f"Error normalizing price: {e}")
        return 0.0

def normalize_temperature(temp_value: float, min_temp: float = -10.0, max_temp: float = 40.0) -> float:
    """
    Normalize temperature to [0, 1] range.
    
    Args:
        temp_value: Temperature in Celsius
        min_temp: Minimum expected temperature for normalization
        max_temp: Maximum expected temperature for normalization
    
    Returns:
        Normalized temperature value
    """
    try:
        return min(max((temp_value - min_temp) / (max_temp - min_temp), 0.0), 1.0)
    except Exception as e:
        logger.error(f"Error normalizing temperature: {e}")
        return 0.5  # Default fallback
def normalize_load(load_value: float, max_value: Optional[float] = None) -> float:
    """
    Normalize load to [0, 1] range.
    
    Args:
        load_value: Load in kW
        max_value: Maximum expected load. If None, uses 90th percentile.
    
    Returns:
        Normalized load value
    """
    try:
        if max_value is None:
            load_data = load_load_data()
            max_value = load_data['Consumption (kW)'].quantile(0.9)
        return min(float(load_value) / max_value, 1.0)
    except Exception as e:
        logger.error(f"Error normalizing load: {e}")
        return 0.0

def get_day_details(idx):
    load_data = load_load_data()
    data = load_data.iloc[idx]
    date = data.name
    date = date.hour + date.minute/60.0
    return {'season':data['Season'], 'day_type': data['Day of the Week'], 'hour': date}

def clear_cache():
    """Clear the data cache. Useful for memory management or reloading data."""
    global _data_cache
    _data_cache.clear()
    logger.info("Data cache cleared")

if __name__=="__main__":
    # Example usage
    print(get_day_details(1))
