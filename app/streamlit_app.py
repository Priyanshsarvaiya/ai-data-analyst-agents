from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_data_analyst_agents.core.security import sanitize_user_error_message
from ai_data_analyst_agents.core.sql_source import SQLDataSource, choose_primary_table
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline as run_csv_pipeline
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline as run_sql_pipeline


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

st.set_page_config(page_title="AI Data Analyst Agents", layout="wide")
st.title("AI Data Analyst Agents")
st.caption("Run artifact-grounded analytics on CSV or SQL sources.")


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def safe_read_json(path: Path) -> dict:
    return json.loads(safe_read_text(path))


st.sidebar.header("Inputs")
source_type = st.sidebar.radio("Data source", ["CSV", "SQL"], index=0)
uploaded = None
db_url = ""
base_table = ""

if source_type == "CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    db_url = st.sidebar.text_input(
        "Database URL",
        value="sqlite:///data/sample.db",
        help="Examples: sqlite:///data/my.db or postgresql+psycopg://user:pass@host:5432/dbname",
    )
    base_table = st.sidebar.text_input(
        "Primary table (optional)",
        value="",
        help="Optional table for dataframe profiling/wrangling. SQL tasks can still query across tables.",
    )

question = st.sidebar.text_area(
    "Business question / query",
    placeholder='e.g., "Why is India revenue lower than Germany?"',
    height=120,
)
run_btn = st.sidebar.button("Run Analysis", type="primary", width="stretch")

st.sidebar.markdown("---")
st.sidebar.write("Artifacts are saved under configured `artifacts/`.")


col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.subheader("1) Source Preview")
    if source_type == "CSV":
        if uploaded is not None:
            st.success(f"Uploaded: {uploaded.name} ({uploaded.size:,} bytes)")
            df_preview = pd.read_csv(uploaded)
            st.dataframe(df_preview.head(20), width="stretch")
        else:
            st.info("Upload a CSV to preview.")
    else:
        if db_url.strip():
            try:
                sql_source = SQLDataSource(db_url.strip())
                schema = sql_source.inspect_schema(include_row_counts=False, max_tables=100)
                tables = [t.get("name") for t in schema.get("tables", []) if t.get("name")]
                st.write(f"Detected tables: {len(tables)}")
                if tables:
                    selected = base_table.strip() if base_table.strip() in tables else choose_primary_table(schema)
                    st.write(f"Preview table: `{selected}`")
                    if selected:
                        prev_df = sql_source.load_table(selected, limit=20)
                        st.dataframe(prev_df, width="stretch")
                else:
                    st.warning("No tables found.")
            except Exception as e:
                st.warning(f"Could not inspect SQL source: {e}")
        else:
            st.info("Enter a database URL to preview tables.")

    st.subheader("2) Question")
    if question.strip():
        st.write(question.strip())
    else:
        st.warning("Enter a business question in the sidebar.")


def _render_results(run_dir: Path, fallback_question: str) -> None:
    plan_p = run_dir / "analysis_plan.json"
    prof_p = run_dir / "data_profile.json"
    qa_p = run_dir / "quality_report.json"
    eda_p = run_dir / "eda_summary.json"
    report_p = run_dir / "final_report.md"
    warn_p = run_dir / "quality_warnings.md"
    review_p = run_dir / "review_log.json"
    logs_p = run_dir / "logs.txt"
    charts_dir = run_dir / "charts"

    tabs = st.tabs(["Answer", "Report", "Charts", "Artifacts", "Logs"])

    with tabs[0]:
        st.markdown("## Structured Answer")
        plan = safe_read_json(plan_p) if plan_p.exists() else {}
        prof = safe_read_json(prof_p) if prof_p.exists() else {}
        qa = safe_read_json(qa_p) if qa_p.exists() else {}
        eda = safe_read_json(eda_p) if eda_p.exists() else {}
        review = safe_read_json(review_p) if review_p.exists() else {}

        st.write(f"**Question:** {plan.get('business_question', fallback_question)}")
        st.write(f"**Rows x Cols:** {prof.get('n_rows', '—')} x {prof.get('n_cols', '—')}")
        if "sql_schema" in prof and isinstance(prof["sql_schema"], dict):
            st.write(
                f"**SQL tables:** {prof['sql_schema'].get('table_count', '—')} "
                f"(relationships: {prof['sql_schema'].get('relationship_count', '—')})"
            )
        st.write(
            f"**Duplicate rate:** {qa.get('duplicate_rate', 0.0):.2%}"
            if "duplicate_rate" in qa
            else "**Duplicate rate:** —"
        )

        warnings = qa.get("warnings", [])
        st.write(f"**Data quality warnings:** {len(warnings)}")
        if warnings:
            for w in warnings[:15]:
                st.write(f"- {w}")
            if warn_p.exists():
                st.caption("Full list in quality_warnings.md")

        charts = eda.get("charts", [])
        qa_charts = eda.get("question_aware_charts", [])
        st.write(f"**Question-aware charts:** {len(qa_charts)}")
        st.write(f"**Total charts:** {len(charts)}")

        st.write(f"**Reviewer status:** {review.get('status', '—')}")
        missing = review.get("missing_refs", []) or review.get("missing_artifacts", [])
        if missing:
            st.error(f"Missing evidence refs: {missing}")

    with tabs[1]:
        st.markdown("## Final Report")
        if report_p.exists():
            st.markdown(safe_read_text(report_p))
        else:
            st.warning("final_report.md not found.")

    with tabs[2]:
        st.markdown("## Charts")
        if charts_dir.exists():
            imgs = sorted([p for p in charts_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            if imgs:
                for img in imgs:
                    st.image(str(img), caption=img.name, width="stretch")
            else:
                st.info("No charts found.")
        else:
            st.info("Charts directory not found.")

    with tabs[3]:
        st.markdown("## Artifacts")
        st.code(str(run_dir))
        files = sorted([p for p in run_dir.iterdir() if p.is_file()])
        for p in files:
            st.write(f"- {p.name}")
            if p.suffix.lower() in [".md", ".json", ".csv", ".txt", ".sql"]:
                st.download_button(
                    label=f"Download {p.name}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/octet-stream",
                    key=f"dl_{p.name}_{p.stat().st_size}",
                )

    with tabs[4]:
        st.markdown("## Logs")
        if logs_p.exists():
            st.code(safe_read_text(logs_p))
        else:
            st.info("logs.txt not found.")


with col_right:
    st.subheader("Run & Results")
    st.write("Click **Run Analysis** to generate artifacts and a structured report.")

    if run_btn:
        if not question.strip():
            st.error("Please enter a business question.")
            st.stop()

        try:
            if source_type == "CSV":
                if uploaded is None:
                    st.error("Please upload a CSV file.")
                    st.stop()
                with st.spinner("Saving uploaded CSV..."):
                    tmp_dir = Path(tempfile.mkdtemp(prefix="ai_analyst_"))
                    csv_path = tmp_dir / uploaded.name
                    csv_path.write_bytes(uploaded.getvalue())
                st.info(f"Temporary file: {csv_path}")
                with st.spinner("Running CSV pipeline..."):
                    run_dir = run_csv_pipeline(str(csv_path), question.strip())
            else:
                if not db_url.strip():
                    st.error("Please enter a database URL.")
                    st.stop()
                with st.spinner("Running SQL pipeline..."):
                    run_dir = run_sql_pipeline(
                        db_url=db_url.strip(),
                        business_question=question.strip(),
                        base_table=base_table.strip() or None,
                    )
        except Exception as e:
            st.error(f"Pipeline failed: {sanitize_user_error_message(e)}")
            st.stop()

        st.success(f"Done. Run folder: {run_dir}")
        _render_results(Path(run_dir), question.strip())
    else:
        st.info("Ready. Provide source + question, then run.")
