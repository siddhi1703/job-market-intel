"""
Phase 1 — Data collection.
Pulls real job postings from free public sources (no API key required) and
stores them in a local SQLite database (data/jobs.db).

Sources:
  - Remotive        (remote tech jobs, structured JSON)
  - Arbeitnow       (open job-board API)
  - Greenhouse      (public company job boards)

Run:  python collect.py
"""
import re
import html
import time
from datetime import datetime, timezone

import requests

from db import get_connection, init_db, insert_jobs, count_jobs

HEADERS = {"User-Agent": "job-market-intel/1.0 (student portfolio project)"}

# Roles we care about for the AI/ML + Data job market
SEARCH_TERMS = [
    "machine learning",
    "data scientist",
    "data analyst",
    "ai engineer",
    "software engineer",
]

# A few companies whose Greenhouse boards are public
GREENHOUSE_COMPANIES = [
    "stripe", "databricks", "robinhood", "coinbase",
    "dropbox", "gitlab", "doordash", "figma",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_html(text):
    """Strip HTML tags and collapse whitespace from a job description."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_remotive(term):
    url = "https://remotive.com/api/remote-jobs"
    r = requests.get(url, params={"search": term}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = []
    for j in r.json().get("jobs", []):
        rows.append({
            "source": "remotive",
            "external_id": str(j.get("id")),
            "title": j.get("title"),
            "company": j.get("company_name"),
            "location": j.get("candidate_required_location"),
            "remote": 1,
            "salary_min": None,
            "salary_max": None,
            "salary_raw": j.get("salary") or None,
            "tags": ", ".join(j.get("tags", []) or []),
            "description": clean_html(j.get("description")),
            "url": j.get("url"),
            "posted_date": j.get("publication_date"),
            "collected_at": now_iso(),
        })
    return rows


def fetch_arbeitnow():
    url = "https://www.arbeitnow.com/api/job-board-api"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = []
    for j in r.json().get("data", []):
        created = j.get("created_at")
        posted = (
            datetime.fromtimestamp(created, timezone.utc).isoformat()
            if isinstance(created, (int, float)) else None
        )
        rows.append({
            "source": "arbeitnow",
            "external_id": j.get("slug"),
            "title": j.get("title"),
            "company": j.get("company_name"),
            "location": j.get("location"),
            "remote": 1 if j.get("remote") else 0,
            "salary_min": None,
            "salary_max": None,
            "salary_raw": None,
            "tags": ", ".join((j.get("tags") or []) + (j.get("job_types") or [])),
            "description": clean_html(j.get("description")),
            "url": j.get("url"),
            "posted_date": posted,
            "collected_at": now_iso(),
        })
    return rows


def fetch_greenhouse(company):
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs"
    r = requests.get(url, params={"content": "true"}, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return []
    rows = []
    for j in r.json().get("jobs", []):
        loc = (j.get("location") or {}).get("name", "") or ""
        rows.append({
            "source": f"greenhouse:{company}",
            "external_id": str(j.get("id")),
            "title": j.get("title"),
            "company": company,
            "location": loc,
            "remote": 1 if "remote" in loc.lower() else 0,
            "salary_min": None,
            "salary_max": None,
            "salary_raw": None,
            "tags": "",
            "description": clean_html(j.get("content")),
            "url": j.get("absolute_url"),
            "posted_date": j.get("updated_at"),
            "collected_at": now_iso(),
        })
    return rows


def main():
    conn = get_connection()
    init_db(conn)
    before = count_jobs(conn)

    all_jobs = []

    for term in SEARCH_TERMS:
        try:
            jobs = fetch_remotive(term)
            print(f"  Remotive '{term}': {len(jobs)} jobs")
            all_jobs += jobs
            time.sleep(1)  # be polite
        except Exception as e:
            print(f"  Remotive '{term}' error: {e}")

    try:
        aj = fetch_arbeitnow()
        print(f"  Arbeitnow: {len(aj)} jobs")
        all_jobs += aj
    except Exception as e:
        print(f"  Arbeitnow error: {e}")

    for company in GREENHOUSE_COMPANIES:
        try:
            gj = fetch_greenhouse(company)
            print(f"  Greenhouse {company}: {len(gj)} jobs")
            all_jobs += gj
            time.sleep(0.5)
        except Exception as e:
            print(f"  Greenhouse {company} error: {e}")

    insert_jobs(conn, all_jobs)
    after = count_jobs(conn)
    print(f"\nFetched {len(all_jobs)} rows | inserted {after - before} new | total in DB: {after}")
    conn.close()


if __name__ == "__main__":
    main()
