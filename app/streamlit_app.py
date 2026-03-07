from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from ai_data_analyst_agents.core.local_auth import LocalAuthStore, validate_password_strength
from ai_data_analyst_agents.core.security import sanitize_user_error_message
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.core.sql_source import SQLDataSource, choose_primary_table
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline as run_csv_pipeline
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline as run_sql_pipeline


ROOT = Path(__file__).resolve().parents[1]
AUTH_DB_PATH = ROOT / "data" / "local_auth.db"
AUTH_SESSION_TTL_MIN = int(os.getenv("AUTH_SESSION_TTL_MIN", "240"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

os.chdir(ROOT)

st.set_page_config(page_title="AI Data Analyst Agents", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
          :root {
            --bg: #f5f7f2;
            --ink: #132a13;
            --muted: #436850;
            --brand: #2d6a4f;
            --brand-2: #40916c;
            --accent: #f4a261;
            --card: #ffffff;
            --line: #d8e3da;
          }
          html, body, [class*="st-"] {
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
          }
          .stApp {
            background:
              radial-gradient(1200px 380px at 85% -10%, rgba(244, 162, 97, 0.18), transparent 60%),
              radial-gradient(1000px 420px at -15% -10%, rgba(64, 145, 108, 0.14), transparent 62%),
              var(--bg);
            color: var(--ink);
          }
          .main .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }
          .hero {
            padding: 1.05rem 1.25rem 1.15rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #132a13 0%, #2d6a4f 58%, #40916c 100%);
            color: #f1fff6;
            box-shadow: 0 14px 30px rgba(19, 42, 19, 0.24);
            margin-bottom: 1rem;
          }
          .hero h1 {
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0.2px;
          }
          .hero p {
            margin: 0.35rem 0 0;
            opacity: 0.95;
            font-size: 0.95rem;
          }
          .auth-shell {
            max-width: 760px;
            margin: 1.2rem auto 0;
            padding: 1.1rem 1.1rem 0.5rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            box-shadow: 0 14px 28px rgba(42, 62, 48, 0.12);
          }
          .auth-title {
            margin: 0 0 0.25rem;
            color: var(--ink);
            font-weight: 700;
          }
          .auth-note {
            margin: 0 0 0.8rem;
            color: var(--muted);
            font-size: 0.92rem;
          }
          .stat-chip {
            display: inline-block;
            background: rgba(19, 42, 19, 0.08);
            border: 1px solid rgba(19, 42, 19, 0.14);
            color: #173622;
            padding: 0.16rem 0.52rem;
            border-radius: 999px;
            font-size: 0.76rem;
            margin-right: 0.3rem;
            margin-top: 0.25rem;
          }
          div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #edf5ef 0%, #f9faf8 100%);
            border-right: 1px solid #dde8df;
          }
          div[data-testid="stSidebar"] * {
            color: #173622;
          }
          .mono {
            font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.82rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def _safe_read_json(path: Path) -> dict:
    return json.loads(_safe_read_text(path))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _init_session_state() -> None:
    st.session_state.setdefault("auth_user_id", None)
    st.session_state.setdefault("auth_last_seen", None)


def _clear_auth_session() -> None:
    st.session_state["auth_user_id"] = None
    st.session_state["auth_last_seen"] = None


@st.cache_resource
def _get_auth_store() -> LocalAuthStore:
    return LocalAuthStore(db_path=AUTH_DB_PATH)


@st.cache_resource
def _get_app_cfg():
    return load_app_cfg()


def _current_user(auth: LocalAuthStore):
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return None

    now = _utc_now()
    last_seen_raw = st.session_state.get("auth_last_seen")
    if last_seen_raw:
        try:
            last_seen = datetime.fromisoformat(last_seen_raw)
        except Exception:
            last_seen = now
        if now - last_seen > timedelta(minutes=AUTH_SESSION_TTL_MIN):
            _clear_auth_session()
            return None

    user = auth.get_user_by_id(int(user_id))
    if user is None:
        _clear_auth_session()
        return None

    st.session_state["auth_last_seen"] = now.isoformat()
    return user


def _render_auth_gate(auth: LocalAuthStore):
    st.markdown(
        """
        <div class="auth-shell">
          <h2 class="auth-title">Secure Workspace Access</h2>
          <p class="auth-note">Create a local account or sign in. Passwords are hashed with PBKDF2 and account lockout is enabled after repeated failures.</p>
          <span class="stat-chip">PBKDF2 hashing</span>
          <span class="stat-chip">SQLite local store</span>
          <span class="stat-chip">Lockout policy</span>
          <span class="stat-chip">Session timeout</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@company.com").strip()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", width="stretch")
            if submitted:
                user, msg = auth.authenticate(email=email, password=password)
                if user is None:
                    st.error(msg)
                else:
                    st.session_state["auth_user_id"] = user.id
                    st.session_state["auth_last_seen"] = _utc_now().isoformat()
                    st.success("Signed in.")
                    st.rerun()

    with tab_signup:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name", placeholder="Priyansh Sarvaiya").strip()
            email = st.text_input("Work Email", placeholder="you@company.com").strip()
            password = st.text_input("Create Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            accepted = st.checkbox("I understand this is a local auth store for this machine.")
            submitted = st.form_submit_button("Create Account", width="stretch")
            if submitted:
                if not accepted:
                    st.error("Please confirm local auth usage.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    issues = validate_password_strength(password)
                    if issues:
                        st.error(" ".join(issues))
                    else:
                        ok, msg = auth.create_user(email=email, full_name=full_name, password=password)
                        if not ok:
                            st.error(msg)
                        else:
                            st.success("Account created. You can log in now.")


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
        plan = _safe_read_json(plan_p) if plan_p.exists() else {}
        prof = _safe_read_json(prof_p) if prof_p.exists() else {}
        qa = _safe_read_json(qa_p) if qa_p.exists() else {}
        eda = _safe_read_json(eda_p) if eda_p.exists() else {}
        review = _safe_read_json(review_p) if review_p.exists() else {}

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
            st.markdown(_safe_read_text(report_p))
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
            st.code(_safe_read_text(logs_p))
        else:
            st.info("logs.txt not found.")


def _render_workspace(user) -> None:
    cfg = _get_app_cfg()

    st.markdown(
        """
        <div class="hero">
          <h1>AI Data Analyst Agents</h1>
          <p>Question-driven analytics with artifact-backed reporting for CSV and SQL sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Workspace")
    st.sidebar.write(f"**Signed in:** {user.full_name}")
    st.sidebar.caption(user.email)
    st.sidebar.markdown('<span class="mono">Session timeout: 240 minutes</span>', unsafe_allow_html=True)
    if st.sidebar.button("Logout", width="stretch"):
        _clear_auth_session()
        st.rerun()

    st.sidebar.markdown("---")
    source_type = st.sidebar.radio("Data source", ["CSV", "SQL"], index=0, horizontal=True)

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
            help="Optional table for preview and local dataframe steps.",
        )

    question = st.sidebar.text_area(
        "Business question",
        placeholder='e.g., "Why is India revenue lower than Germany?"',
        height=140,
    )
    run_btn = st.sidebar.button("Run Analysis", type="primary", width="stretch")

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.subheader("Source Preview")
        if source_type == "CSV":
            if uploaded is not None:
                size_mb = uploaded.size / (1024 * 1024)
                if size_mb > MAX_UPLOAD_MB:
                    st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_UPLOAD_MB} MB.")
                else:
                    st.caption(f"{uploaded.name} • {size_mb:.2f} MB")
                    try:
                        df_preview = pd.read_csv(uploaded)
                        st.dataframe(df_preview.head(20), width="stretch")
                    except Exception as e:
                        st.error(f"Preview failed: {sanitize_user_error_message(e)}")
            else:
                st.info("Upload a CSV to preview.")
        else:
            if db_url.strip():
                try:
                    sql_source = SQLDataSource(
                        db_url=db_url.strip(),
                        timeout_s=cfg.llm.timeout_s,
                        max_query_rows=cfg.sql.default_query_row_limit,
                        enforce_read_only_sql=cfg.security.enforce_read_only_sql,
                    )
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
                    st.warning(f"Could not inspect SQL source: {sanitize_user_error_message(e)}")
            else:
                st.info("Enter a database URL to preview tables.")

    with right:
        st.subheader("Question & Run")
        st.write(question.strip() if question.strip() else "Add your business question in the sidebar.")
        st.caption("Pipelines run with artifact-backed outputs and reviewer checks.")

    st.markdown("---")
    st.subheader("Results")

    if run_btn:
        if not question.strip():
            st.error("Please enter a business question.")
            st.stop()

        try:
            if source_type == "CSV":
                if uploaded is None:
                    st.error("Please upload a CSV file.")
                    st.stop()
                size_mb = uploaded.size / (1024 * 1024)
                if size_mb > MAX_UPLOAD_MB:
                    st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_UPLOAD_MB} MB.")
                    st.stop()
                with st.spinner("Saving uploaded CSV..."):
                    tmp_dir = Path(tempfile.mkdtemp(prefix="ai_analyst_"))
                    safe_name = Path(uploaded.name).name
                    csv_path = tmp_dir / safe_name
                    csv_path.write_bytes(uploaded.getvalue())
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
        st.info("Ready. Configure source + question, then run.")


def main() -> None:
    _inject_styles()
    _init_session_state()

    auth = _get_auth_store()
    user = _current_user(auth)
    if user is None:
        _render_auth_gate(auth)
        return
    _render_workspace(user)


if __name__ == "__main__":
    main()
