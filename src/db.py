"""SQLite helpers for the Job Market Intelligence project."""
import sqlite3
from pathlib import Path

# data/jobs.db at the project root (one level up from src/)
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "jobs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT,
    external_id   TEXT,
    title         TEXT,
    company       TEXT,
    location      TEXT,
    remote        INTEGER,
    salary_min    REAL,
    salary_max    REAL,
    salary_raw    TEXT,
    tags          TEXT,
    description   TEXT,
    url           TEXT,
    posted_date   TEXT,
    collected_at  TEXT,
    UNIQUE(source, external_id)
);
"""


def get_connection(db_path=DB_PATH):
    """Open a connection, creating the data/ folder if needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def insert_jobs(conn, jobs):
    """Insert a list of job dicts; duplicates (source+external_id) are ignored."""
    sql = """
    INSERT OR IGNORE INTO jobs
    (source, external_id, title, company, location, remote,
     salary_min, salary_max, salary_raw, tags, description, url, posted_date, collected_at)
    VALUES (:source, :external_id, :title, :company, :location, :remote,
            :salary_min, :salary_max, :salary_raw, :tags, :description, :url,
            :posted_date, :collected_at)
    """
    conn.executemany(sql, jobs)
    conn.commit()


def count_jobs(conn):
    return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
