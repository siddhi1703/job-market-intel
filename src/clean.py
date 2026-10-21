"""
Phase 2a — Cleaning & normalization.
Reads raw postings from jobs.db, categorizes each into a role, normalizes
location/remote, de-duplicates, and writes a `jobs_clean` table + CSV.

Run:  python clean.py
"""
import re
from pathlib import Path

import pandas as pd

from db import get_connection

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Ordered rules: first match wins, so put specific roles before generic ones.
ROLE_RULES = [
    ("Data Scientist",    [r"data scientist", r"\bdata science\b"]),
    ("Data Analyst",      [r"data analyst", r"business analyst", r"\banalytics\b"]),
    ("Data Engineer",     [r"data engineer", r"analytics engineer", r"\betl\b"]),
    ("ML Engineer",       [r"machine learning", r"\bml engineer\b", r"\bml\b"]),
    ("AI Engineer",       [r"\bai engineer\b", r"artificial intelligence", r"ai/ml", r"\bgen ?ai\b", r"\bllm\b"]),
    ("Research Scientist",[r"research scientist", r"research engineer", r"applied scientist"]),
    ("Software Engineer", [r"software engineer", r"\bswe\b", r"developer", r"backend",
                            r"front.?end", r"full.?stack", r"programmer"]),
]


def categorize(title: str) -> str:
    t = (title or "").lower()
    for role, patterns in ROLE_RULES:
        if any(re.search(p, t) for p in patterns):
            return role
    return "Other"


def parse_country(location: str, is_remote: int) -> str:
    loc = (location or "").strip()
    if not loc or "remote" in loc.lower() or "anywhere" in loc.lower():
        return "Remote/Unspecified"
    # take the last comma-separated chunk as the country/region
    part = loc.split(",")[-1].strip()
    return part if part else "Other"


def main():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM jobs", conn)
    print("Raw rows:", len(df))

    df["title"] = df["title"].fillna("").str.strip()
    df["company"] = df["company"].fillna("Unknown").str.strip()
    df["role_category"] = df["title"].apply(categorize)
    df["is_remote"] = (
        (df["remote"] == 1)
        | df["location"].fillna("").str.contains("remote", case=False)
    ).astype(int)
    df["country"] = df.apply(lambda r: parse_country(r["location"], r["is_remote"]), axis=1)

    before = len(df)
    df = df.drop_duplicates(subset=["title", "company"]).reset_index(drop=True)
    print(f"De-duplicated: {before} -> {len(df)}")

    df.to_sql("jobs_clean", conn, if_exists="replace", index=False)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "jobs_clean.csv", index=False)
    print("Saved `jobs_clean` table + data/jobs_clean.csv")

    print("\nRole category counts:")
    print(df["role_category"].value_counts().to_string())
    conn.close()


if __name__ == "__main__":
    main()
