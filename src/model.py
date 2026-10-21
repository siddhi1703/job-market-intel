"""
Phase 4 — Salary prediction model.
Uses the open ai-jobs.net salary dataset (thousands of real AI/data salaries),
engineers features, trains and compares models, and reports metrics +
feature importance.

Run:  python model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
CSV_PATH = DATA_DIR / "salaries.csv"

# Open Data Science / AI salaries dataset (same data as ai-jobs.net, mirrored)
SOURCES = [
    "https://raw.githubusercontent.com/kittenpub/database-repository/main/ds_salaries.csv",
]


def download_salaries():
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 50000:
        print(f"Using cached {CSV_PATH.name}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    for url in SOURCES:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            # validate it's really the CSV, not an HTML redirect page
            if r.status_code == 200 and "salary_in_usd" in r.text[:500]:
                CSV_PATH.write_bytes(r.content)
                print(f"Downloaded salaries from {url} ({len(r.content)//1024} KB)")
                return
        except Exception as e:
            print(f"  source failed ({url}): {e}")
    raise SystemExit("Could not download a salary dataset — place ds_salaries.csv in data/ manually.")


def group_title(title):
    t = str(title).lower()
    if "analyst" in t: return "Data Analyst"
    if "data scien" in t: return "Data Scientist"
    if "data engineer" in t: return "Data Engineer"
    if "machine learning" in t or "ml engineer" in t: return "ML Engineer"
    if "ai " in t or "artificial" in t or "nlp" in t or "computer vision" in t: return "AI/Research"
    if "research" in t: return "AI/Research"
    if "analytics eng" in t: return "Analytics Engineer"
    if "manager" in t or "lead" in t or "head" in t: return "Lead/Manager"
    return "Other"


def main():
    download_salaries()
    df = pd.read_csv(CSV_PATH)
    print("Rows:", len(df), "| columns:", list(df.columns)[:12])

    # Basic hygiene
    df = df[(df["salary_in_usd"] > 10000) & (df["salary_in_usd"] < 500000)].copy()
    df["role"] = df["job_title"].apply(group_title)

    # Keep top company locations, bucket the rest
    top_loc = df["company_location"].value_counts().head(12).index
    df["company_location"] = df["company_location"].where(df["company_location"].isin(top_loc), "Other")

    features_cat = ["experience_level", "employment_type", "company_size", "company_location", "role"]
    features_num = ["work_year", "remote_ratio"]
    X = df[features_cat + features_num]
    y = df["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ("num", "passthrough", features_num),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=None,
                                               min_samples_leaf=3, random_state=42, n_jobs=-1),
    }

    results = {}
    best_name, best_pipe, best_r2 = None, None, -np.inf
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results[name] = (mae, r2)
        print(f"  {name:20} MAE ${mae:,.0f}   R2 {r2:.3f}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    print(f"\nBest model: {best_name} (R2 {best_r2:.3f})")

    # ---- Feature importance (aggregate one-hot back to original feature) ----
    REPORTS.mkdir(exist_ok=True)
    if best_name == "Random Forest":
        ohe = best_pipe.named_steps["pre"].named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(features_cat)
        all_names = list(cat_names) + features_num
        imp = best_pipe.named_steps["model"].feature_importances_
        agg = {}
        for n, v in zip(all_names, imp):
            base = n.split("_")[0] if n in list(cat_names) else n
            # map one-hot col back to its source feature
            base = next((f for f in features_cat if n.startswith(f + "_")), n)
            agg[base] = agg.get(base, 0) + v
        imp_s = pd.Series(agg).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        imp_s.plot(kind="barh", ax=ax, color="#2E75B6")
        ax.set_title(f"Salary Drivers — Feature Importance ({best_name})", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(REPORTS / "salary_feature_importance.png", dpi=130, bbox_inches="tight")
        plt.close()

    # ---- Predicted vs actual ----
    pred = best_pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred, alpha=0.3, color="#2E75B6", s=12)
    lims = [0, max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual salary (USD)")
    ax.set_ylabel("Predicted salary (USD)")
    ax.set_title(f"Predicted vs Actual — {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "salary_pred_vs_actual.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---- Median salary by role & experience (insight table) ----
    piv = (df.groupby(["role", "experience_level"])["salary_in_usd"]
           .median().unstack().round(0))
    piv.to_csv(DATA_DIR / "median_salary_by_role_exp.csv")
    print("\nMedian salary by role x experience (USD):")
    print(piv.to_string())
    print("\nSaved charts -> reports/salary_feature_importance.png, reports/salary_pred_vs_actual.png")


if __name__ == "__main__":
    main()
