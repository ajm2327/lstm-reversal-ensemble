"""
database/db.py
==============
Database engine, session management, and base class for the
LSTM Reversal Ensemble system.

SQLite is used for development. Switching to PostgreSQL later requires
only two changes:
  1. Update DATABASE_URL in config.py (or .env) to a postgres:// string
  2. Remove the `connect_args` block (SQLite-specific) from create_engine()

Everything else — models, queries, migrations — stays the same.

Usage:
    from database.db import get_db, engine, Base

    # In a script or trainer:
    with get_db() as db:
        db.add(some_model_instance)
        db.commit()

    # To create all tables (run once on first launch):
    from database.db import init_db
    init_db()
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
# Reads from environment variable if set, otherwise defaults to a local
# SQLite file. This makes the swap to Postgres a one-line .env change.
#
# SQLite  : "sqlite:///./lstm_reversal.db"
# Postgres: "postgresql://user:password@localhost:5432/lstm_reversal"
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lstm_reversal.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# SQLite-specific args:
#   check_same_thread=False — allows the same connection across threads,
#     required for FastAPI which handles requests in a thread pool.
#   timeout=20 — wait up to 20s for a locked database before raising.
#
# WAL pragmas below improve concurrent read performance on SQLite, which
# matters when the live prediction engine and the API are both reading.
# ---------------------------------------------------------------------------
if _is_sqlite:
    engine = create_engine(
        DATABASE_URL,
        echo=False,         # Set True temporarily to log all SQL for debugging
        connect_args={
            "check_same_thread": False,
            "timeout": 20,
        },
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        """Apply SQLite performance pragmas on every new connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")    # write-ahead logging
        cursor.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
        cursor.execute("PRAGMA cache_size=10000")    # ~40MB page cache
        cursor.execute("PRAGMA foreign_keys=ON")     # enforce FK constraints
        cursor.close()

else:
    # PostgreSQL — no special connect_args needed
    engine = create_engine(DATABASE_URL, echo=False)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,   # explicit commits required
    autoflush=False,    # flush manually for predictable behaviour
)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
# All models in database/models.py inherit from this Base.
# Base.metadata.create_all(engine) creates every table at once.
# ---------------------------------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    """
    Provide a transactional database session as a context manager.

    Commits on clean exit, rolls back on any exception, and always
    closes the session when done.

    Usage:
        with get_db() as db:
            db.add(record)
            db.commit()

        # Or for read-only queries:
        with get_db() as db:
            results = db.query(OHLCVData).filter_by(ticker="SPY").all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def init_db():
    """
    Create all tables defined in database/models.py if they don't exist.

    Safe to call on every startup — SQLAlchemy uses CREATE TABLE IF NOT EXISTS
    under the hood, so existing tables and data are never touched.

    Call this once at application startup before any database operations:
        from database.db import init_db
        init_db()
    """
    # Import models here so their classes are registered on Base.metadata
    # before create_all is called. Without this import, the tables won't exist
    # in metadata even if models.py has been written correctly.
    import database.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised — {DATABASE_URL}")
    print(f"[db] Tables created/verified at: {DATABASE_URL}")