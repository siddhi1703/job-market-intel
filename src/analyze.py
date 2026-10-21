"""
Phase 3 — Skill extraction & analysis (NLP).
Scans job descriptions for a curated skills taxonomy, ranks the most
in-demand skills overall and per role, and saves charts.

Run:  python analyze.py
"""
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from db import get_connection

REPORTS = Path(__file__).resolve().parents[1] / "reports"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Canonical skill -> regex pattern (matched case-insensitively)
SKILLS = {
    # Languages
    "Python": r"python", "SQL": r"\bsql\b", "Java": r"\bjava\b",
    "JavaScript": r"javascript|\bjs\b", "TypeScript": r"typescript",
    "C++": r"c\+\+", "C#": r"c#", "Scala": r"\bscala\b", "Go": r"\bgolang\b",
    "Rust": r"\brust\b", "Bash/Shell": r"\bbash\b|shell script",
    # ML / AI
    "Machine Learning": r"machine learning", "Deep Learning": r"deep learning",
    "NLP": r"\bnlp\b|natural language processing", "Computer Vision": r"computer vision",
    "LLMs": r"\bllm\b|large language model", "Generative AI": r"generative ai|gen ?ai",
    "PyTorch": r"pytorch", "TensorFlow": r"tensorflow", "Scikit-learn": r"scikit.?learn|sklearn",
    "Hugging Face": r"hugging ?face", "XGBoost": r"xgboost", "OpenCV": r"opencv",
    "LangChain": r"langchain", "MLflow": r"mlflow",
    # Data / BI
    "Pandas": r"pandas", "NumPy": r"numpy", "Spark": r"\bspark\b", "Hadoop": r"hadoop",
    "Kafka": r"kafka", "Airflow": r"airflow", "dbt": r"\bdbt\b", "ETL": r"\betl\b",
    "Snowflake": r"snowflake", "BigQuery": r"bigquery", "Databricks": r"databricks",
    "Tableau": r"tableau", "Power BI": r"power ?bi", "Looker": r"looker", "Excel": r"\bexcel\b",
    # Databases
    "PostgreSQL": r"postgre", "MySQL": r"mysql", "MongoDB": r"mongo",
    "Redis": r"redis", "Elasticsearch": r"elastic ?search",
    # Cloud / DevOps
    "AWS": r"\baws\b|amazon web services", "Azure": r"\bazure\b", "GCP": r"\bgcp\b|google cloud",
    "Docker": r"docker", "Kubernetes": r"kubernetes|\bk8s\b", "Terraform": r"terraform",
    "CI/CD": r"ci/cd|continuous integration", "Git": r"\bgit\b|github|gitlab", "Linux": r"linux",
    # Other
    "REST API": r"rest api|restful", "GraphQL": r"graphql", "Microservices": r"microservice",
}
COMPILED = {name: re.compile(pat, re.IGNORECASE) for name, pat in SKILLS.items()}

TECH_ROLES = ["Data Scientist", "Data Analyst", "Data Engineer", "ML Engineer",
              "AI Engineer", "Research Scientist", "Software Engineer"]


def extract_skills(text):
    text = text or ""
    return {name for name, rx in COMPILED.items() if rx.search(text)}


def main():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM jobs_clean", conn)

    # Focus on AI/ML & Data + SWE roles for a clean "in-demand skills" view
    tech = df[df["role_category"].isin(TECH_ROLES)].copy()
    tech["text"] = (tech["title"].fillna("") + " " + tech["tags"].fillna("")
                    + " " + tech["description"].fillna(""))
    tech["skills"] = tech["text"].apply(extract_skills)
    print(f"Analyzing {len(tech)} tech/data postings\n")

    # ---- Overall skill demand ----
    counts = {}
    for sk in tech["skills"]:
        for s in sk:
            counts[s] = counts.get(s, 0) + 1
    skill_df = (pd.DataFrame({"skill": list(counts), "postings": list(counts.values())})
                .sort_values("postings", ascending=False).reset_index(drop=True))
    skill_df["pct"] = (skill_df["postings"] / len(tech) * 100).round(1)
    skill_df.to_csv(DATA_DIR / "skill_counts.csv", index=False)

    print("TOP 15 IN-DEMAND SKILLS (AI/ML & Data roles):")
    print(skill_df.head(15).to_string(index=False), "\n")

    # ---- Chart 1: top 20 skills ----
    REPORTS.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")
    top = skill_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top["skill"], top["postings"], color="#2E75B6")
    ax.set_title("Top 20 In-Demand Skills — AI/ML & Data Roles", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of postings")
    for i, (v, p) in enumerate(zip(top["postings"], top["pct"])):
        ax.text(v + 1, i, f"{p}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / "top_skills.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---- Chart 2: skills-by-role heatmap (% of role's postings mentioning skill) ----
    roles_present = [r for r in TECH_ROLES if (tech["role_category"] == r).sum() >= 10]
    top_skills = skill_df.head(15)["skill"].tolist()
    matrix = np.zeros((len(top_skills), len(roles_present)))
    for j, role in enumerate(roles_present):
        sub = tech[tech["role_category"] == role]
        n = len(sub)
        for i, skill in enumerate(top_skills):
            cnt = sub["skills"].apply(lambda s: skill in s).sum()
            matrix[i, j] = cnt / n * 100 if n else 0

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(roles_present)))
    ax.set_xticklabels([f"{r}\n(n={(tech['role_category']==r).sum()})" for r in roles_present],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_skills)))
    ax.set_yticklabels(top_skills)
    ax.set_title("Skill Demand by Role (% of postings)", fontsize=14, fontweight="bold")
    for i in range(len(top_skills)):
        for j in range(len(roles_present)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 50 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="% of postings")
    plt.tight_layout()
    plt.savefig(REPORTS / "skills_by_role.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"Saved charts -> reports/top_skills.png, reports/skills_by_role.png")
    conn.close()


if __name__ == "__main__":
    main()
