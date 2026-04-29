"""
database/models.py
==================
SQLAlchemy ORM models for the LSTM Reversal Ensemble system.

All models inherit from Base (defined in database/db.py). Tables are created
by calling init_db() at startup, which runs Base.metadata.create_all().

Table overview:
    OHLCVData       — raw price bars per ticker and timeframe
    FeatureData     — computed indicator values linked to an OHLCV row
    ModelVersion    — trained model metadata (one row per saved model)
    Prediction      — per-timeframe model outputs + ensemble ERS score
    SignalOutcome   — retroactive outcome tracking for live signal accuracy
    BacktestResult  — summary metrics from a completed backtest run
    User            — registered user accounts (stub, used in Phase 12)
    Subscription    — Stripe subscription records (stub, used in Phase 12)

Relationships:
    OHLCVData      1──* FeatureData
    OHLCVData      1──* Prediction
    ModelVersion   1──* Prediction
    Prediction     1──1 SignalOutcome
    User           1──* Subscription
"""

import uuid
import json
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Date, BigInteger, Text, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.db import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string. Used as primary key default."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# OHLCVData
# ---------------------------------------------------------------------------

class OHLCVData(Base):
    """
    Stores raw price bars fetched from Alpaca, one row per candle.

    The (ticker, timeframe, timestamp) combination is unique — this prevents
    duplicate rows when data is re-fetched for an overlapping date range.

    Columns:
        timeframe   — one of: "5min", "15min", "Hour", "Day"
        vwap        — included if Alpaca returns it (may be None for IEX feed)
        trade_count — number of trades in the bar (may be None for IEX feed)
    """
    __tablename__ = "ohlcv_data"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(10), nullable=False)
    timeframe   = Column(String(10), nullable=False)   # "5min" | "15min" | "Hour" | "Day"
    timestamp   = Column(DateTime(timezone=True), nullable=False)
    date        = Column(Date, nullable=False)
    open        = Column(Float, nullable=False)
    high        = Column(Float, nullable=False)
    low         = Column(Float, nullable=False)
    close       = Column(Float, nullable=False)
    volume      = Column(BigInteger, nullable=False)
    vwap        = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    fetched_at  = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationships
    features    = relationship("FeatureData",  back_populates="ohlcv", cascade="all, delete-orphan")
    predictions = relationship("Prediction",   back_populates="ohlcv", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<OHLCVData(ticker='{self.ticker}', timeframe='{self.timeframe}', "
            f"timestamp='{self.timestamp}', close={self.close})>"
        )


# Composite unique constraint + indexes for fast lookups
Index("ix_ohlcv_ticker_tf_ts", OHLCVData.ticker, OHLCVData.timeframe, OHLCVData.timestamp, unique=True)
Index("ix_ohlcv_ticker_tf",    OHLCVData.ticker, OHLCVData.timeframe)


# ---------------------------------------------------------------------------
# FeatureData
# ---------------------------------------------------------------------------

class FeatureData(Base):
    """
    Stores computed technical indicator values for a single OHLCV bar.

    Linked 1:1 to an OHLCVData row via ohlcv_id. Separating raw OHLCV from
    indicators means we can recompute indicators without touching raw data,
    and we can store multiple indicator versions if the feature set changes.

    All indicator values are nullable — if a bar is too early in the series
    for a rolling window to compute (e.g. the first 26 bars for MACD),
    the row exists but the value is NULL.
    """
    __tablename__ = "feature_data"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    ohlcv_id        = Column(Integer, ForeignKey("ohlcv_data.id", ondelete="CASCADE"), nullable=False)

    # RSI
    rsi             = Column(Float)

    # EMA
    emaf            = Column(Float)   # 20-period fast EMA

    # Bollinger Bands
    bb_upper        = Column(Float)
    bb_middle       = Column(Float)
    bb_lower        = Column(Float)
    bb_width        = Column(Float)
    bb_pct          = Column(Float)   # %B: position within bands

    # MACD
    macd            = Column(Float)
    macd_signal     = Column(Float)
    macd_hist       = Column(Float)

    # Volatility
    atr             = Column(Float)
    atr_pct         = Column(Float)   # ATR as % of close
    hist_volatility = Column(Float)   # 20-period annualised return std

    # Volume
    obv             = Column(Float)   # On-Balance Volume
    rvol            = Column(Float)   # Relative Volume vs 20-day avg

    calculated_at   = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationship
    ohlcv           = relationship("OHLCVData", back_populates="features")

    def __repr__(self):
        return f"<FeatureData(ohlcv_id={self.ohlcv_id}, rsi={self.rsi}, macd={self.macd})>"


Index("ix_feature_ohlcv_id", FeatureData.ohlcv_id)


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------

class ModelVersion(Base):
    """
    Tracks every trained model — one row per training run.

    Each model is scoped to a specific ticker and timeframe. The `is_active`
    flag marks the currently deployed model for that scope; only one model
    per (ticker, timeframe) pair should be active at a time.

    `parameters` and `metrics` store JSON blobs so we can record arbitrary
    hyperparameters and evaluation metrics without schema changes.

    Columns:
        model_id    — UUID string, used as the filename stem for saved weights
        timeframe   — which timeframe this model was trained on
        model_path  — path to the saved .keras model file
        scaler_path — path to the saved MinMaxScaler .pkl file
        parameters  — JSON: hyperparameters (units, dropout, lr, epochs, etc.)
        metrics     — JSON: val_loss, val_accuracy, f1, auc, etc.
        trained_on  — date training data ended (useful for staleness checks)
    """
    __tablename__ = "model_versions"

    model_id    = Column(String(36), primary_key=True, default=_new_uuid)
    ticker      = Column(String(10), nullable=False)
    timeframe   = Column(String(10), nullable=False)
    version     = Column(String(50), nullable=False)        # e.g. "1.0.0"
    model_path  = Column(String(255), nullable=True)        # path to .keras file
    scaler_path = Column(String(255), nullable=True)        # path to .pkl file
    parameters  = Column(Text, nullable=False, default="{}")
    metrics     = Column(Text, nullable=True)
    is_active   = Column(Boolean, default=False)
    trained_on  = Column(Date, nullable=True)               # last date in training set
    created_at  = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationship
    predictions = relationship("Prediction", back_populates="model")

    # JSON helpers — avoids json.loads/dumps at every call site
    def set_parameters(self, params: dict):
        self.parameters = json.dumps(params)

    def get_parameters(self) -> dict:
        return json.loads(self.parameters) if self.parameters else {}

    def set_metrics(self, metrics: dict):
        self.metrics = json.dumps(metrics)

    def get_metrics(self) -> dict:
        return json.loads(self.metrics) if self.metrics else {}

    def __repr__(self):
        return (
            f"<ModelVersion(ticker='{self.ticker}', timeframe='{self.timeframe}', "
            f"version='{self.version}', is_active={self.is_active})>"
        )


Index("ix_model_ticker_tf", ModelVersion.ticker, ModelVersion.timeframe)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Prediction(Base):
    """
    Stores the output of every model inference, live or backtest.

    Each row captures both the per-timeframe model outputs and the final
    ensemble Reversal Score (ERS) at the time of prediction.

    Columns:
        ohlcv_id            — the bar this prediction was made on
        model_id            — which ModelVersion produced this prediction
        timeframe           — redundant with model, but useful for fast queries
        reversal_prob       — sigmoid output of the per-TF model (0.0 – 1.0)
        reversal_direction  — "bullish", "bearish", or None if not classified
        ers_score           — Ensemble Reversal Score (0 – 100)
        is_live             — True = live prediction, False = backtest
        predicted_at        — wall-clock time the prediction was generated
    """
    __tablename__ = "predictions"

    prediction_id       = Column(String(36), primary_key=True, default=_new_uuid)
    ohlcv_id            = Column(Integer, ForeignKey("ohlcv_data.id", ondelete="CASCADE"), nullable=False)
    model_id            = Column(String(36), ForeignKey("model_versions.model_id"), nullable=False)
    ticker              = Column(String(10), nullable=False)
    timeframe           = Column(String(10), nullable=False)
    reversal_prob       = Column(Float, nullable=False)     # per-TF model output
    reversal_direction  = Column(String(10), nullable=True) # "bullish" | "bearish" | None
    ers_score           = Column(Float, nullable=True)      # ensemble score 0–100
    is_live             = Column(Boolean, default=True)
    predicted_at        = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationships
    ohlcv   = relationship("OHLCVData",     back_populates="predictions")
    model   = relationship("ModelVersion",  back_populates="predictions")
    outcome = relationship("SignalOutcome",  back_populates="prediction", uselist=False)

    def __repr__(self):
        return (
            f"<Prediction(ticker='{self.ticker}', timeframe='{self.timeframe}', "
            f"reversal_prob={self.reversal_prob:.3f}, ers={self.ers_score})>"
        )


Index("ix_prediction_ticker_tf",    Prediction.ticker, Prediction.timeframe)
Index("ix_prediction_predicted_at", Prediction.predicted_at)


# ---------------------------------------------------------------------------
# SignalOutcome
# ---------------------------------------------------------------------------

class SignalOutcome(Base):
    """
    Records the retrospective outcome of a live prediction signal.

    Filled in after the lookahead window has elapsed. Used to compute
    rolling live accuracy metrics shown on the dashboard and to trigger
    performance-based retraining.

    Columns:
        was_correct     — True if the predicted reversal occurred
        actual_move_pct — actual % price move over the lookahead window
        evaluated_at    — when the outcome was determined
        notes           — optional context (e.g. "earnings day", "halted")
    """
    __tablename__ = "signal_outcomes"

    outcome_id      = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id   = Column(String(36), ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=False, unique=True)
    ticker          = Column(String(10), nullable=False)
    timeframe       = Column(String(10), nullable=False)
    was_correct     = Column(Boolean, nullable=False)
    actual_move_pct = Column(Float, nullable=True)
    evaluated_at    = Column(DateTime(timezone=True), default=func.current_timestamp())
    notes           = Column(Text, nullable=True)

    # Relationship
    prediction      = relationship("Prediction", back_populates="outcome")

    def __repr__(self):
        return (
            f"<SignalOutcome(ticker='{self.ticker}', timeframe='{self.timeframe}', "
            f"was_correct={self.was_correct}, move={self.actual_move_pct:.2f}%)>"
        )


Index("ix_outcome_ticker_tf", SignalOutcome.ticker, SignalOutcome.timeframe)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class BacktestResult(Base):
    """
    Stores summary metrics from a completed backtest run.

    Each row represents one full backtest execution. Detailed trade-by-trade
    results are stored as a JSON blob in `trades` to avoid needing a separate
    trades table at this stage.

    Columns:
        ticker          — ticker backtested (SPY for now)
        timeframe       — timeframe tested ("all" for ensemble backtests)
        start_date      — start of the backtest period
        end_date        — end of the backtest period
        total_trades    — number of signals fired
        win_rate        — fraction of profitable trades
        sharpe_ratio    — annualised Sharpe ratio
        max_drawdown    — maximum peak-to-trough drawdown (as a negative float)
        total_return    — total strategy return over the period
        benchmark_return— SPY buy-and-hold return over the same period
        metrics         — JSON: any additional metrics (Calmar, Sortino, etc.)
        trades          — JSON: list of individual trade records
        report_path     — path to the generated HTML/PDF report file
    """
    __tablename__ = "backtest_results"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(10), nullable=False)
    timeframe        = Column(String(10), nullable=False)
    start_date       = Column(Date, nullable=False)
    end_date         = Column(Date, nullable=False)
    total_trades     = Column(Integer, nullable=True)
    win_rate         = Column(Float, nullable=True)
    sharpe_ratio     = Column(Float, nullable=True)
    max_drawdown     = Column(Float, nullable=True)
    total_return     = Column(Float, nullable=True)
    benchmark_return = Column(Float, nullable=True)
    metrics          = Column(Text, nullable=True)   # JSON blob
    trades           = Column(Text, nullable=True)   # JSON blob
    report_path      = Column(String(255), nullable=True)
    created_at       = Column(DateTime(timezone=True), default=func.current_timestamp())

    def set_metrics(self, metrics: dict):
        self.metrics = json.dumps(metrics)

    def get_metrics(self) -> dict:
        return json.loads(self.metrics) if self.metrics else {}

    def set_trades(self, trades: list):
        self.trades = json.dumps(trades)

    def get_trades(self) -> list:
        return json.loads(self.trades) if self.trades else []

    def __repr__(self):
        return (
            f"<BacktestResult(ticker='{self.ticker}', timeframe='{self.timeframe}', "
            f"return={self.total_return:.2%}, sharpe={self.sharpe_ratio:.2f})>"
        )


Index("ix_backtest_ticker_tf", BacktestResult.ticker, BacktestResult.timeframe)


# ---------------------------------------------------------------------------
# User  (Phase 12 stub)
# ---------------------------------------------------------------------------

class User(Base):
    """
    Registered user accounts.

    Stub table — populated and used in Phase 12 (Auth + Stripe).
    Created now so the schema migration history is clean from the start.
    """
    __tablename__ = "users"

    id           = Column(String(36), primary_key=True, default=_new_uuid)
    email        = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    is_active    = Column(Boolean, default=True)
    is_premium   = Column(Boolean, default=False)
    created_at   = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationship
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(email='{self.email}', is_premium={self.is_premium})>"


# ---------------------------------------------------------------------------
# Subscription  (Phase 12 stub)
# ---------------------------------------------------------------------------

class Subscription(Base):
    """
    Stripe subscription records linked to a user.

    Stub table — populated and used in Phase 12 (Auth + Stripe).
    `stripe_subscription_id` is the Stripe object ID used for webhooks.
    """
    __tablename__ = "subscriptions"

    id                      = Column(String(36), primary_key=True, default=_new_uuid)
    user_id                 = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stripe_subscription_id  = Column(String(255), nullable=True, unique=True)
    stripe_customer_id      = Column(String(255), nullable=True)
    plan                    = Column(String(50), nullable=True)    # "premium", "pro", etc.
    status                  = Column(String(50), nullable=True)    # "active", "cancelled", etc.
    current_period_end      = Column(DateTime(timezone=True), nullable=True)
    created_at              = Column(DateTime(timezone=True), default=func.current_timestamp())

    # Relationship
    user = relationship("User", back_populates="subscriptions")

    def __repr__(self):
        return (
            f"<Subscription(user_id='{self.user_id}', plan='{self.plan}', "
            f"status='{self.status}')>"
        )