"""
This module handles the multi-timeframe data fetching that wraps data_util.get_alpaca_data
It handles technical indicators
Handles the multi-timeframe data fetching that wraps data_util.get_alpaca_data

The main difference from autotrader lite is that the database operations won't be baked
into the feature engineering process. This fully focuses on the data pre processing, not the storage.

Usage:
from core.feature_engineer import (
get_all_timeframes,
add_indicators,
TIMEFRAME_CONFIG,
SUPPORTED_TIMEFRAMES,
)

data = get_all_timeframes("SPY", start="2024-01-01", end="2024-12-31")

df = get_alpaca_data("SPY", "2024-01-01", "2024-12-31", timescale="Hour")
df = add_indicators(df, timeframe="Hour")
"""


import pandas as pd
import numpy as np
from typing import Dict, Optional


from core.data_util import (
    get_alpaca_data,
    calculate_rsi,
    get_rvol
)

TIMEFRAME_CONFIG: Dict[str, dict] = {
    "5min":{
        "timescale": "5Min",
        "lookback_window": 60,
        "zigzag_threshold": 0.0025,
        "lookahead_window": 6,
        "bar_duration_min": 5,
        "description": "5-Minute",
    },

    "15min":{
        "timescale": "15Min",
        "lookback_window": 40,
        "zigzag_threshold": 0.004,
        "lookahead_window": 8,
        "bar_duration_min": 15,
        "description": "15-Minute",
    },

    "Hour":{
        "timescale": "Hour",
        "lookback_window": 30,
        "zigzag_threshold": 0.008,
        "lookahead_window": 6,
        "bar_duration_min": 60,
        "description": "1-Hour",
    },

    "Day":{
        "timescale": "Day",
        "lookback_window": 20,
        "zigzag_threshold": 0.015,
        "lookahead_window": 5,
        "bar_duration_min": 390,
        "description": "Daily",
    },

}

SUPPORTED_TIMEFRAMES = ["5min", "15min", "Hour", "Day"]


def add_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame based on the specified timeframe.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing price data.
    timeframe (str): The timeframe for which to calculate indicators (e.g., "5min", "15min", "Hour", "Day").

    Returns:
    pd.DataFrame: The DataFrame with added technical indicators.
    """

    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported timeframes are: {list(TIMEFRAME_CONFIG.keys())}")
    
    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns for indicator calculation: {missing}")
    
    data = df.copy()


    # RSI CALCULATION:
    data["RSI"] = calculate_rsi(data["Close"], period=14)

    # EMA CALCULATION - used for bollinger bands middle band
    data["EMAF"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands Calculation
    bb_window = 20
    bb_mean = data["Close"].rolling(window=bb_window).mean()
    bb_std = data["Close"].rolling(window=bb_window).std()

    data["BB_upper"] = bb_mean + (bb_std * 2)
    data["BB_middle"] = bb_mean
    data["BB_lower"] = bb_mean - (bb_std * 2)
    data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / data["BB_middle"]

    # %B: 0 = at lower band, 0.5 is middle, 1 = upper
    #avoid outliers:
    band_range = (data["BB_upper"] - data["BB_lower"]).replace(0, np.nan)
    data["BB_pct"]= ((data["Close"] - data["BB_lower"]) / band_range).clip(-0.5, 1.5)

    # CALCULATE MACD
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean
    data["MACD"] = ema_12 - ema_26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    # ATR CALCULATION
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(window=14).mean()
    data["ATR_pct"] = data["ATR"] / data["Close"]


    # OBV CALCULATION
    data["OBV"] = (
        (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()
    )

    # RVOL CALCULATION
    data["RVOL"] = get_rvol(data)

    #historical volatility
    data["hist_volatility"] = (
        data["Close"].pct_change().rolling(window=20).std() * np.sqrt(252)
    )

    data.dropna(inplace=True)

    return data


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "EMAF",
    "BB_upper", "BB_middle", "BB_lower", "BB_width", "BB_pct",
    "MACD", "MACD_signal", "MACD_hist",
    "ATR", "ATR_pct",
    "OBV",
    "RVOL",
    "hist_volatility",
]

def fetch_timeframe(
        ticker: str,
        timeframe: str,
        start: str,
        end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch ohlcv data for a single timeframe and compute indicators.

    wrapper aroudn get_alpaca_data and add_indicators
    The database layer won't be called here.

    Parameters:
    ticker (str): The stock ticker symbol.
    timeframe (str): The timeframe to fetch (SUPPORTED_TIMEFRAMES)
    start (str): The start date for the data range.
    end (Optional[str]): The end date for the data range. Defaults to None.

    Returns:
    Optional[pd.DataFrame]: The DataFrame with fetched OHLCV data and computed indicators, or None if an error occurs.

    """

    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported timeframes are: {SUPPORTED_TIMEFRAMES}")
    
    config = TIMEFRAME_CONFIG[timeframe]
    print(f"[feature_engineer] Fetching {ticker} @ {config['description']} from {start} to {end}")

    df = get_alpaca_data(
        ticker=ticker,
        start_date=start,
        end_date=end,
        timescale=config["timescale"]
    )

    if df is None or df.empty:
        print(f"[feature_engineer] No data returned for {ticker} @ {config['description']} from {start} to {end}")
        return None
    
    df = add_indicators(df, timeframe)

    print(
        f"[feature_engineer] {ticker} @ {config['description']}: "
        f"{len(df)} rows | columns: {len(df.columns)}"
    )

    return df

def get_all_timeframes(
        ticker: str,
        start: str,
        end: Optional[str] = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch and process data for all timeframes.
    
    Returns a dictionary keyed by timeframe.
    
    Parameteres:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data range.
    end (Optional[str]): The end date for the data range. Defaults to None.
    """

    print(f"\n[feature_engineer] === Fetching all timeframes for {ticker} from {start} to {end} ===\n")

    result: Dict[str, Optional[pd.DataFrame]] = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        try:
            result[timeframe] = fetch_timeframe(ticker, timeframe, start, end)
        except Exception as e:
            print(f"[feature_engineer] Error fetching {ticker} @ {timeframe}: {e}")
            result[timeframe] = None

    print("\n[feature_engineer] === Fetch summary ===")
    for tf, df in result.items():
        if df is not None:
            print(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s}: {len(df)} rows | columns: {len(df.columns)}"
                f"{df.index.min()} to {df.index.max()}"
            )
        else:
            print(f"{TIMEFRAME_CONFIG[tf]['description']:12s}: No data")

    return result