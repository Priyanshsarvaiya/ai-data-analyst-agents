from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_data_analyst_agents.core.security import sanitize_user_error_message
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.core.sql_source import SQLDataSource, choose_primary_table
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline as run_csv_pipeline
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline as run_sql_pipeline
try:
    from app.postgres_auth import PostgresAuthStore, load_auth_settings, validate_password_strength
    from app.run_tracking import RunTrackingStore, execute_tracked_run
except ModuleNotFoundError:
    from postgres_auth import PostgresAuthStore, load_auth_settings, validate_password_strength
    from run_tracking import RunTrackingStore, execute_tracked_run

os.chdir(ROOT)

st.set_page_config(page_title="AI Data Analyst Agents", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
          :root {
            --bg: #090e1b;
            --ink: #e7efff;
            --muted: #a5b6d8;
            --brand: #4aa8ff;
            --brand-2: #6fe6d1;
            --accent: #ffb86b;
            --card: #111a2f;
            --line: #2a3b64;
          }
          html, body, [class*="st-"] {
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
          }
          /* Restore Streamlit material icon ligatures (prevents keyboard_double_arrow_* text). */
          .material-symbols-rounded,
          .material-icons,
          [data-testid="stIconMaterial"],
          [data-testid="stSidebarCollapseButton"] span,
          [data-testid="collapsedControl"] span {
            font-family: "Material Symbols Rounded", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
          }
          .stApp {
            background:
              radial-gradient(1200px 500px at 88% -10%, rgba(111, 230, 209, 0.16), transparent 62%),
              radial-gradient(900px 420px at -10% -15%, rgba(74, 168, 255, 0.16), transparent 65%),
              var(--bg);
            color: var(--ink);
          }
          .main .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }
          header[data-testid="stHeader"] {
            background: rgba(9, 14, 27, 0.76);
            backdrop-filter: blur(6px);
          }
          [data-testid="stToolbar"] {
            color: #d9e8ff !important;
          }
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li,
          [data-testid="stMarkdownContainer"] span,
          [data-testid="stText"] {
            color: var(--ink);
          }
          .hero {
            padding: 1.05rem 1.25rem 1.15rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #17294d 0%, #244273 55%, #1c7b88 100%);
            color: #eff8ff;
            border: 1px solid #2f4f7f;
            box-shadow: 0 16px 32px rgba(5, 10, 22, 0.45);
            margin-bottom: 1rem;
          }
          .hero h1 {
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0.2px;
          }
          .hero p {
            margin: 0.35rem 0 0;
            color: #d2e5ff;
            font-size: 0.95rem;
          }
          .auth-shell {
            max-width: 940px;
            margin: 1.2rem auto 0;
            padding: 1.2rem 1.25rem;
            border-radius: 24px;
            background: linear-gradient(170deg, rgba(17, 27, 47, 0.96), rgba(14, 23, 39, 0.92));
            border: 1px solid #2d426e;
            box-shadow: 0 18px 36px rgba(3, 8, 20, 0.5);
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
          .auth-form-title {
            margin: 0.2rem 0 0.45rem;
            color: #dceaff;
            font-size: 1.02rem;
            font-weight: 700;
          }
          div[data-testid="stRadio"] > div {
            background: rgba(18, 31, 55, 0.95);
            border: 1px solid #324a7a;
            border-radius: 12px;
            padding: 0.3rem 0.45rem 0.15rem;
          }
          div[data-testid="stForm"] {
            border: 1px solid #304a78;
            border-radius: 16px;
            padding: 0.75rem 0.9rem 0.45rem;
            background: linear-gradient(180deg, rgba(18, 29, 52, 0.96), rgba(14, 24, 42, 0.96));
          }
          div[data-testid="stForm"] label p,
          div[data-testid="stForm"] label span {
            color: #dce9ff !important;
            font-weight: 600 !important;
          }
          div[data-testid="stRadio"] label p,
          div[data-testid="stRadio"] label span {
            color: #dce9ff !important;
            font-weight: 600 !important;
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div {
            background: #10192f !important;
            border: 1px solid #39538a !important;
            box-shadow: none !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] {
            background: transparent !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] button {
            background: transparent !important;
            border: none !important;
            color: #dceaff !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] svg {
            fill: #dceaff !important;
          }
          div[data-baseweb="input"] input,
          div[data-baseweb="select"] input,
          div[data-baseweb="textarea"] textarea {
            color: #ebf3ff !important;
          }
          div[data-baseweb="input"] input::placeholder,
          div[data-baseweb="textarea"] textarea::placeholder {
            color: #8ca6d0 !important;
          }
          div[data-baseweb="textarea"] > div {
            background: #10192f !important;
            border: 1px solid #39538a !important;
          }
          button {
            border-radius: 10px !important;
            min-height: 2.65rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px;
          }
          button[kind="secondary"] {
            background: #182543 !important;
            border: 1px solid #355188 !important;
            color: #dceaff !important;
          }
          div[data-testid="stForm"] button[kind="primary"] {
            background: linear-gradient(120deg, #245ca5, #1e8e9a) !important;
            border: 1px solid #2b6cbe !important;
            color: #eef6ff !important;
          }
          div[data-testid="stForm"] button[kind="primary"]:hover {
            border: 1px solid #5b9fe8 !important;
            color: #ffffff !important;
          }
          div[data-testid="stForm"] [data-testid="stMarkdownContainer"] p {
            color: #a8c0e8;
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
          }
          .stTabs [data-baseweb="tab"] {
            background: #141f39;
            border: 1px solid #2d4573;
            border-radius: 10px;
            color: #cfe1ff;
            padding: 0.3rem 0.7rem;
          }
          .stTabs [aria-selected="true"] {
            background: #1f3a66 !important;
            border-color: #4e7fc7 !important;
            color: #f1f7ff !important;
          }
          div[data-testid="stDataFrame"] div[role="table"] {
            border: 1px solid #2b426f;
            border-radius: 10px;
          }
          [data-testid="stCodeBlockContainer"] pre {
            background: #0f172a !important;
            border: 1px solid #2b426f !important;
            color: #e3edff !important;
          }
          div[data-testid="stSidebar"] {
            background:
              radial-gradient(680px 260px at 120% -10%, rgba(81, 137, 219, 0.18), transparent 72%),
              linear-gradient(180deg, #0c1529 0%, #091124 100%);
            border-right: 1px solid #21345b;
          }
          section[data-testid="stSidebar"] {
            background:
              radial-gradient(680px 260px at 120% -10%, rgba(81, 137, 219, 0.18), transparent 72%),
              linear-gradient(180deg, #0c1529 0%, #091124 100%);
          }
          div[data-testid="stSidebar"] * {
            color: #d6e6ff !important;
          }
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #d6e6ff !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="radio"] label {
            background: #15233f;
            border: 1px solid #2f4c7f;
            border-radius: 8px;
            padding: 0.2rem 0.45rem;
            margin-right: 0.25rem;
          }
          div[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
            color: #ffffff !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] > div,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] > div {
            background: #111c34 !important;
            border: 1px solid #35548c !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] [data-baseweb="input-suffix"] {
            background: transparent !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] [data-baseweb="input-suffix"] button {
            background: transparent !important;
            border: none !important;
            color: #e5efff !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] [data-baseweb="input-suffix"] svg {
            fill: #e5efff !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] input,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
            color: #ecf3ff !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder {
            color: #91abd4 !important;
          }
          div[data-testid="stSidebar"] button {
            background: linear-gradient(120deg, #245ca5, #1e8e9a) !important;
            border: 1px solid #2f6ec0 !important;
            color: #f1f8ff !important;
          }
          div[data-testid="stSidebar"] button:hover {
            border: 1px solid #5ea1e8 !important;
          }
          section[data-testid="stSidebar"] hr {
            border-color: #253d68 !important;
          }
          .mono {
            font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.82rem;
            color: #b9d0f4;
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


_EV_INLINE_PATTERN = re.compile(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]|\[\[EV-([a-f0-9]{10})\]\]")
_EV_INDEX_ROW_PATTERN = re.compile(
    r"^\|?\s*(EV-[a-f0-9]{10})\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|\s*(.*)$"
)


def _format_report_for_display(report: str) -> str:
    text = (report or "").strip()
    if not text:
        return text

    # New format already present.
    if "## 8) Evidence References" in text or "## 9) Evidence References" in text:
        return text

    tags = _EV_INLINE_PATTERN.findall(text)
    ordered_ids: list[str] = []
    ev_to_num: dict[str, int] = {}
    for a, b in tags:
        ev_id = a or (f"EV-{b}" if b else "")
        if not ev_id:
            continue
        if ev_id not in ev_to_num:
            ev_to_num[ev_id] = len(ordered_ids) + 1
            ordered_ids.append(ev_id)

    if not ordered_ids:
        return text

    def _replace(match: re.Match[str]) -> str:
        ev_id = match.group(1) or (f"EV-{match.group(2)}" if match.group(2) else "")
        if not ev_id:
            return ""
        return f"[{ev_to_num[ev_id]}]"

    out = _EV_INLINE_PATTERN.sub(_replace, text).rstrip()

    row_map: dict[str, dict[str, str]] = {}
    for ln in text.splitlines():
        m = _EV_INDEX_ROW_PATTERN.match(ln.strip())
        if not m:
            continue
        ev_id = m.group(1)
        row_map[ev_id] = {
            "kind": m.group(2).strip(),
            "artifact": m.group(3).strip(),
            "pointer": m.group(4).strip(),
            "summary": m.group(5).strip(),
        }

    lines = [
        "## 8) Evidence References",
        "| Ref | Evidence ID | Artifact | Pointer | Summary |",
        "|---|---|---|---|---|",
    ]
    for ev_id in ordered_ids:
        n = ev_to_num[ev_id]
        meta = row_map.get(ev_id, {})
        artifact = meta.get("artifact", "-").replace("|", "/")
        pointer = meta.get("pointer", "-").replace("|", "/")
        summary = meta.get("summary", "").replace("|", "/")
        if not summary:
            summary = meta.get("kind", "legacy evidence ref")
        lines.append(f"| [{n}] | {ev_id} | {artifact} | {pointer} | {summary} |")

    return out + "\n\n" + "\n".join(lines) + "\n"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _init_session_state() -> None:
    st.session_state.setdefault("auth_user_id", None)
    st.session_state.setdefault("auth_last_seen", None)
    st.session_state.setdefault("selected_run_dir", None)
    st.session_state.setdefault("selected_run_question", "")


def _clear_auth_session() -> None:
    st.session_state["auth_user_id"] = None
    st.session_state["auth_last_seen"] = None
    st.session_state["selected_run_dir"] = None
    st.session_state["selected_run_question"] = ""


@st.cache_resource
def _get_auth_store() -> PostgresAuthStore:
    auth_cfg = _get_auth_cfg()
    if not auth_cfg.resolved_database_url:
        raise RuntimeError(
            "AUTH_DATABASE_URL/DATABASE_URL is not configured. Set it in your environment or .env file."
        )
    return PostgresAuthStore(
        database_url=auth_cfg.resolved_database_url,
        pepper=auth_cfg.AUTH_PASSWORD_PEPPER,
        iterations=int(auth_cfg.AUTH_PASSWORD_HASH_ITERATIONS),
        lock_after_failures=int(auth_cfg.AUTH_LOCKOUT_ATTEMPTS),
        lock_minutes=int(auth_cfg.AUTH_LOCKOUT_MINUTES),
    )


@st.cache_resource
def _get_app_cfg():
    return load_app_cfg()


@st.cache_resource
def _get_auth_cfg():
    return load_auth_settings()


@st.cache_resource
def _get_run_store() -> RunTrackingStore:
    auth_cfg = _get_auth_cfg()
    if not auth_cfg.resolved_database_url:
        raise RuntimeError(
            "AUTH_DATABASE_URL/DATABASE_URL is not configured. Set it in your environment or .env file."
        )
    return RunTrackingStore(database_url=auth_cfg.resolved_database_url)


def _current_user(auth: PostgresAuthStore):
    auth_cfg = _get_auth_cfg()
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
        if now - last_seen > timedelta(minutes=int(auth_cfg.AUTH_SESSION_TTL_MIN)):
            _clear_auth_session()
            return None

    user = auth.get_user_by_id(int(user_id))
    if user is None:
        _clear_auth_session()
        return None

    st.session_state["auth_last_seen"] = now.isoformat()
    return user


def _render_auth_gate(auth: PostgresAuthStore):
    st.markdown(
        """
        <div class="auth-shell">
          <h2 class="auth-title">Secure Workspace Access</h2>
          <p class="auth-note">Sign in or create an account to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center_col, _ = st.columns([0.2, 1.0, 0.2], gap="small")
    with center_col:
        mode = st.radio(
            "Authentication Mode",
            ["Login", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown(f'<p class="auth-form-title">{mode}</p>', unsafe_allow_html=True)

        if mode == "Login":
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
        else:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Firstname Lastname").strip()
                email = st.text_input("Work Email", placeholder="you@company.com").strip()
                password = st.text_input("Create Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                st.caption("Use 12+ chars with upper, lower, number, and special character.")
                submitted = st.form_submit_button("Create Account", width="stretch")
                if submitted:
                    if password != confirm:
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
            raw_report = _safe_read_text(report_p)
            st.markdown(_format_report_for_display(raw_report))
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


def _render_workspace(user, run_store: RunTrackingStore) -> None:
    cfg = _get_app_cfg()
    auth_cfg = _get_auth_cfg()
    max_upload_mb = int(auth_cfg.MAX_UPLOAD_MB)
    session_ttl_min = int(auth_cfg.AUTH_SESSION_TTL_MIN)

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
    st.sidebar.markdown(
        f'<span class="mono">Session timeout: {session_ttl_min} minutes</span>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Logout", width="stretch"):
        _clear_auth_session()
        st.rerun()

    recent_runs = run_store.list_runs_for_user(user_id=int(user.id), limit=6)
    if recent_runs:
        def _short_question(run_obj, limit: int = 72) -> str:
            q = str(run_obj.run_metadata.get("business_question", "")).strip()
            if not q:
                q = "Untitled question"
            if len(q) <= limit:
                return q
            return q[: limit - 1].rstrip() + "…"

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Recent Runs")
        for rr in recent_runs:
            st.sidebar.caption(
                f"{_short_question(rr)} | {rr.status} | {rr.created_at[:19]}"
            )
        run_lookup = {
            f"{_short_question(rr, 52)} | {rr.status} | {rr.created_at[:19]}": rr.run_uuid
            for rr in recent_runs
        }
        selected = st.sidebar.selectbox(
            "Open Previous Run",
            options=[""] + list(run_lookup.keys()),
            index=0,
        )
        if selected and st.sidebar.button("Load Selected Run", width="stretch"):
            detail = run_store.get_run_details_for_user(
                user_id=int(user.id),
                run_uuid=run_lookup[selected],
            )
            if detail is None:
                st.sidebar.error("Run not found.")
            else:
                run_dir = Path(str(detail.run.run_metadata.get("run_dir", "")))
                if not run_dir.exists():
                    st.sidebar.error("Run directory no longer exists on disk.")
                else:
                    st.session_state["selected_run_dir"] = str(run_dir)
                    st.session_state["selected_run_question"] = str(
                        detail.run.run_metadata.get("business_question", "")
                    )
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
                if size_mb > max_upload_mb:
                    st.error(f"File too large ({size_mb:.1f} MB). Limit is {max_upload_mb} MB.")
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
    selected_run_dir = st.session_state.get("selected_run_dir")
    selected_run_question = st.session_state.get("selected_run_question", "")

    if run_btn:
        if not question.strip():
            st.error("Please enter a business question.")
            st.stop()

        try:
            run_metadata = {
                "business_question": question.strip(),
                "source_type": source_type.lower(),
            }
            if source_type == "CSV":
                if uploaded is None:
                    st.error("Please upload a CSV file.")
                    st.stop()
                size_mb = uploaded.size / (1024 * 1024)
                if size_mb > max_upload_mb:
                    st.error(f"File too large ({size_mb:.1f} MB). Limit is {max_upload_mb} MB.")
                    st.stop()
                with st.spinner("Saving uploaded CSV..."):
                    tmp_dir = Path(tempfile.mkdtemp(prefix="ai_analyst_"))
                    safe_name = Path(uploaded.name).name
                    csv_path = tmp_dir / safe_name
                    csv_path.write_bytes(uploaded.getvalue())
                run_metadata["source_name"] = safe_name
                run_metadata["file_path"] = str(csv_path)

                with st.spinner("Running CSV pipeline..."):
                    run_row, run_dir = execute_tracked_run(
                        run_store=run_store,
                        user_id=int(user.id),
                        source_type="csv",
                        source_name=safe_name,
                        run_metadata=run_metadata,
                        runner=lambda artifact_cb: run_csv_pipeline(
                            str(csv_path),
                            question.strip(),
                            artifact_callback=artifact_cb,
                        ),
                    )
            else:
                if not db_url.strip():
                    st.error("Please enter a database URL.")
                    st.stop()
                run_metadata["source_name"] = base_table.strip() or "sql_source"
                run_metadata["analysis_table"] = base_table.strip() or None

                with st.spinner("Running SQL pipeline..."):
                    run_row, run_dir = execute_tracked_run(
                        run_store=run_store,
                        user_id=int(user.id),
                        source_type="sql",
                        source_name=base_table.strip() or "sql_source",
                        run_metadata=run_metadata,
                        runner=lambda artifact_cb: run_sql_pipeline(
                            db_url=db_url.strip(),
                            business_question=question.strip(),
                            base_table=base_table.strip() or None,
                            artifact_callback=artifact_cb,
                        ),
                    )
        except Exception as e:
            st.error(f"Pipeline failed: {sanitize_user_error_message(e)}")
            st.stop()

        st.success(f"Done. Run folder: {run_dir} | Run ID: {run_row.run_uuid}")
        st.session_state["selected_run_dir"] = str(run_dir)
        st.session_state["selected_run_question"] = question.strip()
        _render_results(Path(run_dir), question.strip())
    elif selected_run_dir:
        run_path = Path(str(selected_run_dir))
        if run_path.exists():
            st.info(f"Showing previous run: {run_path.name}")
            _render_results(run_path, str(selected_run_question))
        else:
            st.warning("Selected run folder no longer exists.")
    else:
        st.info("Ready. Configure source + question, then run.")


def main() -> None:
    _inject_styles()
    _init_session_state()

    try:
        auth = _get_auth_store()
        run_store = _get_run_store()
    except Exception as e:
        st.error(f"Authentication store initialization failed: {sanitize_user_error_message(e)}")
        st.info(
            "Set `AUTH_DATABASE_URL` (or `DATABASE_URL`) and restart Streamlit. "
            "Example: postgresql+psycopg://USER:PASSWORD@HOST:5432/DB_NAME"
        )
        return
    user = _current_user(auth)
    if user is None:
        _render_auth_gate(auth)
        return
    _render_workspace(user, run_store)


if __name__ == "__main__":
    main()
