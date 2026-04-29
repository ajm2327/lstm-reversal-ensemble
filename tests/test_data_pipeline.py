"""
tests/test_data_pipeline.py
============================
Phase 2 integration test for the full data pipeline.

Tests the following in order:
    1. Database initialisation — tables created from models
    2. Alpaca data fetch — all four timeframes return data
    3. Indicator computation — all expected columns present, no NaNs
    4. Database write — OHLCVData and FeatureData rows inserted correctly
    5. Database read — rows round-trip correctly out of SQLite
    6. Summary report — shape, date range, and column check per timeframe

Run from the project root:
    python -m pytest tests/test_data_pipeline.py -v

Or run directly:
    python tests/test_data_pipeline.py
"""

import sys
import os
import traceback
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure project root is on the path when running directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import init_db, get_db, OHLCVData, FeatureData
from core.feature_engineer import (
    get_all_timeframes,
    add_indicators,
    FEATURE_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_CONFIG,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
TICKER     = "SPY"
START_DATE = "2024-01-01"
END_DATE   = "2024-03-31"   # 3-month window — enough rows, fast enough to run

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def result(label: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


# ---------------------------------------------------------------------------
# Test 1: Database initialisation
# ---------------------------------------------------------------------------

def test_db_init() -> bool:
    section("TEST 1 — Database Initialisation")
    try:
        init_db()
        passed = result("init_db() completed without error", True)
        return passed
    except Exception as e:
        result("init_db() raised an exception", False, str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 2: Alpaca data fetch
# ---------------------------------------------------------------------------

def test_alpaca_fetch() -> dict:
    section("TEST 2 — Alpaca Data Fetch (all timeframes)")

    data = get_all_timeframes(TICKER, start=START_DATE, end=END_DATE)
    all_passed = True

    for tf in SUPPORTED_TIMEFRAMES:
        df = data.get(tf)
        if df is not None and not df.empty:
            result(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s} fetch",
                True,
                f"{len(df):,} rows | {df.index.min().date()} → {df.index.max().date()}"
            )
        else:
            result(f"{TIMEFRAME_CONFIG[tf]['description']:12s} fetch", False, "returned None or empty")
            all_passed = False

    if not all_passed:
        print(f"\n  {WARN} Some timeframes returned no data.")
        print("  This may be normal for 5min/15min if the date range is outside")
        print("  Alpaca IEX free-tier history limits. Try a more recent START_DATE.")

    return data


# ---------------------------------------------------------------------------
# Test 3: Indicator computation
# ---------------------------------------------------------------------------

def test_indicators(data: dict) -> bool:
    section("TEST 3 — Indicator Computation")

    # Indicator columns produced by add_indicators (excludes raw OHLCV)
    expected_indicator_cols = [
        "RSI", "EMAF",
        "BB_upper", "BB_middle", "BB_lower", "BB_width", "BB_pct",
        "MACD", "MACD_signal", "MACD_hist",
        "ATR", "ATR_pct",
        "OBV", "RVOL",
        "hist_volatility",
    ]

    all_passed = True

    for tf in SUPPORTED_TIMEFRAMES:
        df = data.get(tf)
        if df is None or df.empty:
            result(f"{TIMEFRAME_CONFIG[tf]['description']:12s} indicators", False, "no data to test")
            all_passed = False
            continue

        # Check all expected columns are present
        missing_cols = [c for c in expected_indicator_cols if c not in df.columns]
        if missing_cols:
            result(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s} columns",
                False,
                f"missing: {missing_cols}"
            )
            all_passed = False
            continue

        # Check for NaN values in indicator columns
        nan_counts = df[expected_indicator_cols].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]

        if cols_with_nans.empty:
            result(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s} indicators",
                True,
                f"{len(expected_indicator_cols)} columns, 0 NaNs, {len(df):,} rows"
            )
        else:
            # NaNs after dropna in add_indicators would be unexpected
            result(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s} NaN check",
                False,
                f"NaNs found: {cols_with_nans.to_dict()}"
            )
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Test 4: Database write
# ---------------------------------------------------------------------------

def test_db_write(data: dict) -> bool:
    section("TEST 4 — Database Write (OHLCVData + FeatureData)")

    all_passed = True

    for tf in SUPPORTED_TIMEFRAMES:
        df = data.get(tf)
        if df is None or df.empty:
            result(f"{TIMEFRAME_CONFIG[tf]['description']:12s} write", False, "skipped — no data")
            all_passed = False
            continue

        # Write only the first 10 rows per timeframe to keep the test fast
        sample = df.head(10)
        rows_written = 0

        try:
            with get_db() as db:
                for ts, row in sample.iterrows():
                    # Check for existing row to avoid unique constraint errors on re-runs
                    existing = (
                        db.query(OHLCVData)
                        .filter_by(ticker=TICKER, timeframe=tf, timestamp=ts)
                        .first()
                    )
                    if existing:
                        ohlcv_row = existing
                    else:
                        ohlcv_row = OHLCVData(
                            ticker      = TICKER,
                            timeframe   = tf,
                            timestamp   = ts,
                            date        = ts.date(),
                            open        = float(row["Open"]),
                            high        = float(row["High"]),
                            low         = float(row["Low"]),
                            close       = float(row["Close"]),
                            volume      = int(row["Volume"]),
                            vwap        = float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
                            trade_count = int(row["trade_count"]) if "trade_count" in row and pd.notna(row["trade_count"]) else None,
                        )
                        db.add(ohlcv_row)
                        db.flush()  # flush to get the auto-generated id

                    # Only write FeatureData if ohlcv_row is new
                    if not existing:
                        feature_row = FeatureData(
                            ohlcv_id        = ohlcv_row.id,
                            rsi             = float(row["RSI"])             if pd.notna(row.get("RSI"))             else None,
                            emaf            = float(row["EMAF"])            if pd.notna(row.get("EMAF"))            else None,
                            bb_upper        = float(row["BB_upper"])        if pd.notna(row.get("BB_upper"))        else None,
                            bb_middle       = float(row["BB_middle"])       if pd.notna(row.get("BB_middle"))       else None,
                            bb_lower        = float(row["BB_lower"])        if pd.notna(row.get("BB_lower"))        else None,
                            bb_width        = float(row["BB_width"])        if pd.notna(row.get("BB_width"))        else None,
                            bb_pct          = float(row["BB_pct"])          if pd.notna(row.get("BB_pct"))          else None,
                            macd            = float(row["MACD"])            if pd.notna(row.get("MACD"))            else None,
                            macd_signal     = float(row["MACD_signal"])     if pd.notna(row.get("MACD_signal"))     else None,
                            macd_hist       = float(row["MACD_hist"])       if pd.notna(row.get("MACD_hist"))       else None,
                            atr             = float(row["ATR"])             if pd.notna(row.get("ATR"))             else None,
                            atr_pct         = float(row["ATR_pct"])         if pd.notna(row.get("ATR_pct"))         else None,
                            hist_volatility = float(row["hist_volatility"]) if pd.notna(row.get("hist_volatility")) else None,
                            obv             = float(row["OBV"])             if pd.notna(row.get("OBV"))             else None,
                            rvol            = float(row["RVOL"])            if pd.notna(row.get("RVOL"))            else None,
                        )
                        db.add(feature_row)
                        rows_written += 1

            result(
                f"{TIMEFRAME_CONFIG[tf]['description']:12s} write",
                True,
                f"{rows_written} new OHLCV+Feature rows written"
            )

        except Exception as e:
            result(f"{TIMEFRAME_CONFIG[tf]['description']:12s} write", False, str(e))
            traceback.print_exc()
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Test 5: Database read
# ---------------------------------------------------------------------------

def test_db_read() -> bool:
    section("TEST 5 — Database Read (round-trip check)")

    all_passed = True

    for tf in SUPPORTED_TIMEFRAMES:
        try:
            with get_db() as db:
                rows = (
                    db.query(OHLCVData)
                    .filter_by(ticker=TICKER, timeframe=tf)
                    .order_by(OHLCVData.timestamp)
                    .limit(5)
                    .all()
                )

            if rows:
                first = rows[0]
                result(
                    f"{TIMEFRAME_CONFIG[tf]['description']:12s} read",
                    True,
                    f"{len(rows)} rows fetched | first close={first.close:.2f} @ {first.timestamp}"
                )

                # Verify linked FeatureData exists
                with get_db() as db:
                    feat = db.query(FeatureData).filter_by(ohlcv_id=first.id).first()

                if feat is not None:
                    result(
                        f"{TIMEFRAME_CONFIG[tf]['description']:12s} FeatureData link",
                        True,
                        f"RSI={feat.rsi:.2f}, MACD={feat.macd:.4f}"
                    )
                else:
                    result(
                        f"{TIMEFRAME_CONFIG[tf]['description']:12s} FeatureData link",
                        False,
                        "no FeatureData row found for first OHLCVData row"
                    )
                    all_passed = False
            else:
                result(
                    f"{TIMEFRAME_CONFIG[tf]['description']:12s} read",
                    False,
                    "no rows found — did Test 4 pass?"
                )
                all_passed = False

        except Exception as e:
            result(f"{TIMEFRAME_CONFIG[tf]['description']:12s} read", False, str(e))
            traceback.print_exc()
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(data: dict):
    section("SUMMARY — Data Shape Per Timeframe")

    for tf in SUPPORTED_TIMEFRAMES:
        df = data.get(tf)
        cfg = TIMEFRAME_CONFIG[tf]
        if df is not None and not df.empty:
            print(f"\n  {cfg['description']} ({tf})")
            print(f"    Rows        : {len(df):,}")
            print(f"    Date range  : {df.index.min()} → {df.index.max()}")
            print(f"    Columns     : {len(df.columns)}")
            print(f"    Close range : ${df['Close'].min():.2f} – ${df['Close'].max():.2f}")
            print(f"    RSI range   : {df['RSI'].min():.1f} – {df['RSI'].max():.1f}")
            print(f"    ATR mean    : {df['ATR'].mean():.4f}")
        else:
            print(f"\n  {cfg['description']} ({tf})")
            print(f"    NO DATA")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_tests():
    print(f"\n{'#'*60}")
    print(f"  LSTM Reversal Ensemble — Phase 2 Pipeline Test")
    print(f"  Ticker: {TICKER} | {START_DATE} → {END_DATE}")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    results = {}

    # Test 1 — DB init
    results["db_init"] = test_db_init()
    if not results["db_init"]:
        print("\n  Database failed to initialise. Stopping.")
        return

    # Test 2 — Fetch
    data = test_alpaca_fetch()
    results["fetch"] = any(df is not None and not df.empty for df in data.values())

    # Test 3 — Indicators
    results["indicators"] = test_indicators(data)

    # Test 4 — Write
    results["write"] = test_db_write(data)

    # Test 5 — Read
    results["read"] = test_db_read()

    # Summary
    print_summary(data)

    # Final scorecard
    section("SCORECARD")
    total  = len(results)
    passed = sum(results.values())

    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  Phase 2 complete. Ready to move to Phase 3 — Reversal Labeler.\n")
    else:
        print("\n  Fix failures above before proceeding to Phase 3.\n")


if __name__ == "__main__":
    run_all_tests()