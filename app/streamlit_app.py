from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import streamlit as st

# Import your existing pipeline runner
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # ensures .env and configs/ are found


st.set_page_config(page_title="AI Data Analyst Agents", layout="wide")

st.title("AI Data Analyst Agents 📊")
st.caption("Upload a CSV, ask a question, and generate artifact-grounded analytics.")

# ---------- Sidebar ----------
st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
question = st.sidebar.text_area(
    "Business question / query",
    placeholder='e.g., "Why did revenue vary by country?"',
    height=120,
)

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Output")
st.sidebar.write("Artifacts are saved under your configured `artifacts/` directory.")

# ---------- Helpers ----------
def safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(errors="ignore")

def safe_read_json(p: Path) -> dict:
    return json.loads(safe_read_text(p))

def find_latest_run_dir(artifacts_root: Path) -> Path | None:
    if not artifacts_root.exists():
        return None
    run_dirs = [d for d in artifacts_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda d: d.name)[-1]

# ---------- Main ----------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("1) Preview")
    if uploaded is not None:
        st.success(f"Uploaded: {uploaded.name} ({uploaded.size:,} bytes)")
        # Show a quick preview
        import pandas as pd
        df_preview = pd.read_csv(uploaded)
        st.dataframe(df_preview.head(20), use_container_width=True)
    else:
        st.info("Upload a CSV to begin.")

    st.subheader("2) Question")
    if question.strip():
        st.write(question.strip())
    else:
        st.warning("Enter a question in the sidebar.")

with col_right:
    st.subheader("Run & Results")
    st.write("Click **Run Analysis** to generate artifacts + report.")

    if run_btn:
        if uploaded is None:
            st.error("Please upload a CSV file.")
            st.stop()
        if not question.strip():
            st.error("Please enter a business question.")
            st.stop()

        # Save uploaded file to a temp path (so your pipeline can read from disk)
        with st.spinner("Saving uploaded CSV..."):
            tmp_dir = Path(tempfile.mkdtemp(prefix="ai_analyst_"))
            csv_path = tmp_dir / uploaded.name
            csv_path.write_bytes(uploaded.getvalue())

        st.info(f"Saved to temporary file: {csv_path}")

        # Run pipeline
        with st.spinner("Running analysis pipeline (Phase 1)..."):
            run_dir = run_pipeline(str(csv_path), question.strip())

        st.success(f"Done! Run folder: {run_dir}")

        # Load artifacts and render structured results
        run_dir = Path(run_dir)

        # Files we expect
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

        # ---------- Answer ----------
        with tabs[0]:
            st.markdown("## ✅ Structured Answer")

            plan = safe_read_json(plan_p) if plan_p.exists() else {}
            prof = safe_read_json(prof_p) if prof_p.exists() else {}
            qa = safe_read_json(qa_p) if qa_p.exists() else {}
            eda = safe_read_json(eda_p) if eda_p.exists() else {}
            review = safe_read_json(review_p) if review_p.exists() else {}

            # Executive Summary
            st.markdown("### Executive Summary")
            st.write(f"**Question:** {plan.get('business_question', question.strip())}")
            st.write(f"**Rows × Cols:** {prof.get('n_rows', '—')} × {prof.get('n_cols', '—')}")
            st.write(f"**Duplicate rate:** {qa.get('duplicate_rate', 0.0):.2%}" if "duplicate_rate" in qa else "**Duplicate rate:** —")

            warnings = qa.get("warnings", [])
            st.write(f"**Data quality warnings:** {len(warnings)}")

            if warnings:
                st.markdown("### ⚠️ Data Quality Warnings")
                for w in warnings[:20]:
                    st.write(f"- {w}")
                if warn_p.exists():
                    st.caption("Full list: quality_warnings.md")

            # EDA summary
            st.markdown("### 📊 EDA Summary")
            num_cols = eda.get("numeric_columns", [])
            st.write(f"**Numeric columns detected:** {len(num_cols)}")
            if num_cols:
                st.write(", ".join(num_cols[:15]) + (" ..." if len(num_cols) > 15 else ""))

            charts = eda.get("charts", [])
            st.write(f"**Charts generated:** {len(charts)}")

            # Reviewer
            st.markdown("### 🛡️ Reviewer Check")
            st.write(f"**Status:** {review.get('status', '—')}")
            missing = review.get("missing_artifacts", [])
            if missing:
                st.error(f"Missing artifacts: {missing}")

        # ---------- Report ----------
        with tabs[1]:
            st.markdown("## 📝 Final Report")
            if report_p.exists():
                st.markdown(safe_read_text(report_p))
            else:
                st.warning("final_report.md not found.")

        # ---------- Charts ----------
        with tabs[2]:
            st.markdown("## 📈 Charts")
            if charts_dir.exists():
                imgs = sorted([p for p in charts_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
                if not imgs:
                    st.info("No charts found.")
                else:
                    for p in imgs:
                        st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.info("Charts directory not found.")

        # ---------- Artifacts ----------
        with tabs[3]:
            st.markdown("## 📦 Artifacts Folder")
            st.code(str(run_dir))

            files = sorted([p for p in run_dir.iterdir() if p.is_file()])
            if files:
                for p in files:
                    st.write(f"- {p.name}")
                    # Download buttons for key artifacts
                    if p.suffix.lower() in [".md", ".json", ".csv", ".txt"]:
                        st.download_button(
                            label=f"Download {p.name}",
                            data=p.read_bytes(),
                            file_name=p.name,
                            mime="application/octet-stream",
                            key=f"dl_{p.name}",
                        )
            else:
                st.info("No artifact files found.")

        # ---------- Logs ----------
        with tabs[4]:
            st.markdown("## 🪵 Logs")
            if logs_p.exists():
                st.code(safe_read_text(logs_p))
            else:
                st.info("logs.txt not found.")

    else:
        st.info("Ready when you are. Upload a CSV and ask a question in the sidebar.")