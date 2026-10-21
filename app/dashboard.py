"""
Phase 5 — Interactive dashboard (Streamlit).
Ties the project together: market overview, in-demand skills explorer,
and a salary predictor built on the trained model.

Run:  streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from db import get_connection                 # noqa: E402
from analyze import extract_skills, TECH_ROLES  # noqa: E402
from model import group_title, download_salaries  # noqa: E402

st.set_page_config(page_title="AI/ML Job Market Intelligence", page_icon="📊", layout="wide")


# ---------------- Data loaders (cached) ----------------
@st.cache_data
def load_jobs():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM jobs_clean", conn)
    conn.close()
    tech = df[df["role_category"].isin(TECH_ROLES)].copy()
    tech["text"] = (tech["title"].fillna("") + " " + tech["tags"].fillna("")
                    + " " + tech["description"].fillna(""))
    tech["skills"] = tech["text"].apply(extract_skills)
    return df, tech


@st.cache_data
def load_salaries():
    download_salaries()
    df = pd.read_csv(ROOT / "data" / "salaries.csv")
    df = df[(df["salary_in_usd"] > 10000) & (df["salary_in_usd"] < 500000)].copy()
    df["role"] = df["job_title"].apply(group_title)
    top = df["company_location"].value_counts().head(12).index
    df["company_location"] = df["company_location"].where(df["company_location"].isin(top), "Other")
    return df


@st.cache_resource
def train_model(sal):
    cat = ["experience_level", "employment_type", "company_size", "company_location", "role"]
    num = ["work_year", "remote_ratio"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])
    pipe = Pipeline([("pre", pre),
                     ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                 random_state=42, n_jobs=-1))])
    pipe.fit(sal[cat + num], sal["salary_in_usd"])
    return pipe, cat, num


# ---------------- Load ----------------
df_all, tech = load_jobs()
sal = load_salaries()
model, CAT, NUM = train_model(sal)

EXP_LABELS = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}

# ---------------- Header ----------------
st.title("📊 AI/ML & Data — Job Market Intelligence")
st.caption("Live job postings + open salary data. Built by Siddhi Kakani during the co-op search.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Postings analyzed", f"{len(df_all):,}")
c2.metric("Tech/Data roles", f"{len(tech):,}")
c3.metric("Companies", f"{df_all['company'].nunique():,}")
c4.metric("Salary records", f"{len(sal):,}")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
roles_sel = st.sidebar.multiselect("Role category", TECH_ROLES, default=TECH_ROLES)
remote_only = st.sidebar.checkbox("Remote only", value=False)

view = tech[tech["role_category"].isin(roles_sel)]
if remote_only:
    view = view[view["is_remote"] == 1]

tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🧠 In-Demand Skills", "💰 Salary Explorer"])

# ---------------- Tab 1 ----------------
with tab1:
    a, b = st.columns(2)
    with a:
        st.subheader("Postings by role")
        st.bar_chart(view["role_category"].value_counts())
    with b:
        st.subheader("Top hiring companies")
        st.bar_chart(view["company"].value_counts().head(10))
    c, d = st.columns(2)
    with c:
        st.subheader("Remote vs On-site")
        st.bar_chart(view["is_remote"].map({1: "Remote", 0: "On-site"}).value_counts())
    with d:
        st.subheader("Top locations")
        loc = view[view["country"] != "Remote/Unspecified"]["country"].value_counts().head(10)
        st.bar_chart(loc)

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Most in-demand skills (for the selected roles)")
    counts = {}
    for sk in view["skills"]:
        for s in sk:
            counts[s] = counts.get(s, 0) + 1
    if counts:
        sc = pd.Series(counts).sort_values(ascending=False).head(20)
        st.bar_chart(sc)
        st.caption(f"Based on {len(view):,} postings. % = share of postings mentioning the skill.")
        tbl = (sc.rename("postings").reset_index().rename(columns={"index": "skill"}))
        tbl["% of postings"] = (tbl["postings"] / len(view) * 100).round(1)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No postings match the current filters.")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Median salary by role × experience (USD)")
    piv = (sal.groupby(["role", "experience_level"])["salary_in_usd"]
           .median().unstack().round(0).rename(columns=EXP_LABELS))
    st.dataframe(piv, use_container_width=True)

    st.subheader("🔮 Salary predictor")
    st.caption("Estimate a salary from role, experience, location, and work mode (model R² ≈ 0.42).")
    f1, f2, f3 = st.columns(3)
    role_in = f1.selectbox("Role", sorted(sal["role"].unique()))
    exp_in = f2.selectbox("Experience", ["EN", "MI", "SE", "EX"],
                          format_func=lambda x: EXP_LABELS[x])
    loc_in = f3.selectbox("Company location", sorted(sal["company_location"].unique()))
    g1, g2, g3 = st.columns(3)
    size_in = g1.selectbox("Company size", ["S", "M", "L"], index=1)
    remote_in = g2.selectbox("Remote ratio", [0, 50, 100], index=2)
    year_in = g3.selectbox("Year", sorted(sal["work_year"].unique(), reverse=True))

    if st.button("Predict salary", type="primary"):
        row = pd.DataFrame([{
            "experience_level": exp_in, "employment_type": "FT", "company_size": size_in,
            "company_location": loc_in, "role": role_in,
            "work_year": year_in, "remote_ratio": remote_in,
        }])
        pred = model.predict(row[CAT + NUM])[0]
        st.success(f"Estimated salary: **${pred:,.0f} / year**")

st.divider()
st.caption("Sources: Remotive, Arbeitnow, Greenhouse boards (live postings) · ai-jobs.net open salary dataset. "
           "Note: posting sample skews toward a few large employers (Databricks, Stripe); salary model explains ~42% of variance.")
