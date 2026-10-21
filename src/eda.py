"""
Phase 2b — Exploratory Data Analysis.
Reads `jobs_clean`, prints SQL-style summaries, and saves a 2x2 overview
chart to reports/eda_overview.png.

Run:  python eda.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from db import get_connection

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM jobs_clean", conn)
    print("Clean rows:", len(df), "\n")

    # ---- Text summaries ----
    print("Top role categories:")
    print(df["role_category"].value_counts().head(8).to_string(), "\n")

    print("Top 10 hiring companies:")
    print(df["company"].value_counts().head(10).to_string(), "\n")

    print("Remote vs On-site:")
    print(df["is_remote"].map({1: "Remote", 0: "On-site"}).value_counts().to_string(), "\n")

    print("Top 10 locations (country/region):")
    print(df["country"].value_counts().head(10).to_string(), "\n")

    # ---- Charts (2x2 overview) ----
    REPORTS.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AI/ML & Data Job Market — Overview", fontsize=16, fontweight="bold")

    # 1) Roles
    df["role_category"].value_counts().head(8).sort_values().plot(
        kind="barh", ax=axes[0, 0], color="#2E75B6")
    axes[0, 0].set_title("Postings by Role Category")
    axes[0, 0].set_xlabel("Postings")

    # 2) Top companies
    df["company"].value_counts().head(10).sort_values().plot(
        kind="barh", ax=axes[0, 1], color="#5B9BD5")
    axes[0, 1].set_title("Top 10 Hiring Companies")
    axes[0, 1].set_xlabel("Postings")

    # 3) Remote vs on-site
    df["is_remote"].map({1: "Remote", 0: "On-site"}).value_counts().plot(
        kind="pie", ax=axes[1, 0], autopct="%1.0f%%",
        colors=["#2E75B6", "#BDD7EE"], ylabel="")
    axes[1, 0].set_title("Remote vs On-site")

    # 4) Top locations
    loc = df[df["country"] != "Remote/Unspecified"]["country"].value_counts().head(10)
    loc.sort_values().plot(kind="barh", ax=axes[1, 1], color="#1F4E78")
    axes[1, 1].set_title("Top Locations")
    axes[1, 1].set_xlabel("Postings")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPORTS / "eda_overview.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved chart -> {out}")
    conn.close()


if __name__ == "__main__":
    main()
