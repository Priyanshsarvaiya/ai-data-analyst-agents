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


def _inject_styles(theme: str = "dark") -> None:
    theme_mode = (theme or "dark").strip().lower()
    light_overrides = """
          :root { color-scheme: light; }
          .stApp {
            background:
              radial-gradient(1100px 460px at 88% -12%, rgba(126, 195, 255, 0.18), transparent 66%),
              radial-gradient(760px 360px at -8% -10%, rgba(79, 149, 255, 0.14), transparent 62%),
              #f3f7ff !important;
            color: #172741 !important;
          }
          header[data-testid="stHeader"] {
            background: rgba(244, 248, 255, 0.86) !important;
            border-bottom: 1px solid #d7e3fb !important;
          }
          [data-testid="stToolbar"] {
            color: #2a4575 !important;
          }
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li,
          [data-testid="stMarkdownContainer"] span,
          [data-testid="stText"] {
            color: #172741 !important;
          }
          .hero {
            background: linear-gradient(120deg, #eef4ff 0%, #ddeafe 55%, #d8fbf3 100%) !important;
            color: #173458 !important;
            border: 1px solid #bcd0ef !important;
            box-shadow: 0 14px 28px rgba(80, 114, 170, 0.18) !important;
          }
          .hero p { color: #3a5a8a !important; }
          .auth-shell {
            background: linear-gradient(170deg, rgba(244, 249, 255, 0.98), rgba(236, 244, 255, 0.98)) !important;
            border: 1px solid #c6d7f5 !important;
            box-shadow: 0 14px 28px rgba(83, 116, 172, 0.2) !important;
          }
          .auth-kicker { color: #3f6294 !important; }
          .auth-title { color: #19355e !important; }
          .auth-note { color: #4e6e9f !important; }
          .auth-chip {
            border: 1px solid #bfd1f0 !important;
            background: #edf4ff !important;
            color: #2f507f !important;
          }
          .auth-form-title { color: #234677 !important; }
          div[data-testid="stRadio"] > div {
            background: #eef4ff !important;
            border: 1px solid #c4d7f8 !important;
          }
          div[data-testid="stForm"] {
            background: linear-gradient(180deg, rgba(248, 252, 255, 0.98), rgba(239, 246, 255, 0.98)) !important;
            border: 1px solid #c4d7f8 !important;
            box-shadow: 0 10px 20px rgba(88, 122, 178, 0.16) !important;
          }
          div[data-testid="stForm"] label p,
          div[data-testid="stForm"] label span,
          div[data-testid="stRadio"] label p,
          div[data-testid="stRadio"] label span {
            color: #203f6d !important;
          }
          div[data-testid="stRadio"] input[type="radio"] {
            accent-color: #2f74db !important;
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div {
            background: #ffffff !important;
            border: 1px solid #b9ceef !important;
          }
          div[data-baseweb="input"] > div {
            overflow: hidden !important;
          }
          div[data-baseweb="input"] input,
          div[data-baseweb="select"] input,
          div[data-baseweb="textarea"] textarea {
            color: #173458 !important;
          }
          div[data-baseweb="input"] input::placeholder,
          div[data-baseweb="textarea"] textarea::placeholder {
            color: #6d86ad !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] {
            background: transparent !important;
            border-left: 1px solid #d1ddf3 !important;
            margin: 0 !important;
            padding: 0 !important;
            border-radius: 0 !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            min-height: 0 !important;
            height: 100% !important;
            min-width: 3rem !important;
            border-radius: 0 !important;
            padding: 0 !important;
            color: #305387 !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] button:hover {
            background: rgba(47, 116, 219, 0.08) !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *,
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *::before,
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *::after {
            background: transparent !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] svg {
            fill: #305387 !important;
          }
          button[kind="secondary"] {
            background: #e7f0ff !important;
            border: 1px solid #b8cdf0 !important;
            color: #234677 !important;
          }
          div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(120deg, #2f74db, #4b9dff) !important;
            border: 1px solid #3e81e8 !important;
            color: #f5faff !important;
            box-shadow: none !important;
          }
          div[data-testid="stFormSubmitButton"] button:hover {
            border: 1px solid #62a2ff !important;
            color: #ffffff !important;
          }
          div[data-testid="stForm"] button[kind="primary"] {
            background: linear-gradient(120deg, #2f74db, #4b9dff) !important;
            border: 1px solid #3e81e8 !important;
            color: #f5faff !important;
          }
          div[data-testid="stForm"] button[kind="primary"]:hover {
            border: 1px solid #62a2ff !important;
          }
          .stTabs [data-baseweb="tab"] {
            background: #eaf2ff !important;
            border: 1px solid #c0d4f5 !important;
            color: #2b4f82 !important;
          }
          .stTabs [aria-selected="true"] {
            background: #dce9ff !important;
            border-color: #8cb2ea !important;
            color: #173f76 !important;
          }
          [data-testid="stCodeBlockContainer"] pre {
            background: #f5f9ff !important;
            border: 1px solid #c6d8f6 !important;
            color: #1d385f !important;
          }
          div[data-testid="stDataFrame"] div[role="table"] {
            border: 1px solid #c6d8f6 !important;
          }
          div[data-testid="stSidebar"],
          section[data-testid="stSidebar"] {
            background:
              radial-gradient(620px 240px at 120% -12%, rgba(138, 194, 255, 0.3), transparent 72%),
              linear-gradient(180deg, #f2f7ff 0%, #e9f1ff 100%) !important;
            border-right: 1px solid #c1d4f4 !important;
          }
          div[data-testid="stSidebar"] * {
            color: #1f3f6f !important;
          }
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
          div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
          div[data-testid="stSidebar"] label p,
          div[data-testid="stSidebar"] label span {
            color: #1f3f6f !important;
            background: transparent !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #bdd1f2 !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #1a365f !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #355a8f !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="radio"] label {
            background: #eef4ff !important;
            border: 1px solid #bdd1f2 !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] > div,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] > div {
            background: #ffffff !important;
            border: 1px solid #bdd1f2 !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] input,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
            color: #1a365f !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
          div[data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder {
            color: #6e87ad !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] [data-baseweb="input-suffix"] button {
            color: #355a8f !important;
          }
          div[data-testid="stSidebar"] [data-baseweb="input"] [data-baseweb="input-suffix"] svg {
            fill: #355a8f !important;
          }
          div[data-testid="stSidebar"] button {
            background: linear-gradient(120deg, #2f74db, #4b9dff) !important;
            border: 1px solid #4b8feb !important;
            color: #f7fbff !important;
          }
          div[data-testid="stSidebar"] button:hover {
            border: 1px solid #70aaff !important;
          }
          div[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #f2f7ff !important;
            border: 1px dashed #b7cced !important;
          }
          div[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #1f3f6f !important;
          }
          div[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
          div[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #5a759f !important;
          }
          div[data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"] {
            background: #e8f1ff !important;
            border: 1px solid #a8c2eb !important;
            color: #234677 !important;
          }
          div[data-testid="stSidebar"] [data-testid="stFileUploader"] button[kind="secondary"]:hover {
            border: 1px solid #84aee4 !important;
          }
          section[data-testid="stSidebar"] hr {
            border-color: #bfd2f1 !important;
          }
          .mono {
            color: #4d6f9f !important;
          }
    """ if theme_mode == "light" else ""

    css = """
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
            max-width: 760px;
            margin: 1.1rem auto 0.55rem;
            padding: 1.15rem 1.2rem 1rem;
            border-radius: 8px;
            background: linear-gradient(170deg, rgba(18, 31, 56, 0.96), rgba(12, 23, 44, 0.94));
            border: 1px solid #355288;
            box-shadow: 0 14px 30px rgba(4, 10, 24, 0.42);
          }
          .auth-kicker {
            margin: 0 0 0.35rem;
            color: #90add9;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
          }
          .auth-title {
            margin: 0 0 0.35rem;
            color: var(--ink);
            font-weight: 700;
            font-size: 2rem;
            line-height: 1.06;
          }
          .auth-note {
            margin: 0;
            color: #c7d9fa;
            font-size: 1.02rem;
          }
          .auth-chip-row {
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
          }
          .auth-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid #3a588f;
            background: rgba(18, 31, 56, 0.82);
            color: #cfe2ff;
            border-radius: 8px;
            padding: 0.2rem 0.5rem;
            font-size: 0.77rem;
            font-weight: 600;
            line-height: 1.2;
          }
          .auth-form-title {
            margin: 0.3rem 0 0.55rem;
            color: #e5f0ff;
            font-size: 1.14rem;
            font-weight: 700;
          }
          div[data-testid="stRadio"] > div {
            background: rgba(16, 28, 52, 0.98);
            border: 1px solid #375690;
            border-radius: 8px;
            padding: 0.28rem 0.4rem 0.15rem;
          }
          div[data-testid="stForm"] {
            border: 1px solid #3a5b98;
            border-radius: 8px;
            padding: 0.85rem 0.95rem 0.55rem;
            background: linear-gradient(180deg, rgba(16, 27, 50, 0.97), rgba(12, 22, 42, 0.97));
            box-shadow: 0 10px 24px rgba(2, 8, 20, 0.36);
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
            background: #0f1a33 !important;
            border: 1px solid #4566a4 !important;
            box-shadow: none !important;
          }
          div[data-baseweb="input"] > div {
            overflow: hidden !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] {
            background: transparent !important;
            border-left: 1px solid #2b3f66 !important;
            margin: 0 !important;
            padding: 0 !important;
            border-radius: 0 !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] button {
            background: transparent !important;
            border: none !important;
            min-height: 0 !important;
            height: 100% !important;
            min-width: 3rem !important;
            border-radius: 0 !important;
            padding: 0 !important;
            color: #dceaff !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] button:hover {
            background: rgba(111, 161, 232, 0.16) !important;
          }
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *,
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *::before,
          div[data-baseweb="input"] [data-baseweb="input-suffix"] *::after {
            background: transparent !important;
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
            background: #0f1a33 !important;
            border: 1px solid #4566a4 !important;
          }
          button {
            border-radius: 8px !important;
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
            background: linear-gradient(120deg, #ff5a64, #ff6e4b) !important;
            border: 1px solid #ff7c61 !important;
            color: #fff7f3 !important;
          }
          div[data-testid="stForm"] button[kind="primary"]:hover {
            border: 1px solid #ff9a7c !important;
            color: #ffffff !important;
          }
          div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(120deg, #ff5a64, #ff6e4b) !important;
            border: 1px solid #ff7c61 !important;
            color: #fff7f3 !important;
          }
          div[data-testid="stFormSubmitButton"] button:hover {
            border: 1px solid #ff9a7c !important;
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
          __LIGHT_OVERRIDES__
        </style>
        """
    st.markdown(css.replace("__LIGHT_OVERRIDES__", light_overrides), unsafe_allow_html=True)


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
    st.session_state.setdefault("auth_expires_at", None)
    st.session_state.setdefault("auth_expired_notice", "")
    st.session_state.setdefault("selected_run_dir", None)
    st.session_state.setdefault("selected_run_question", "")
    st.session_state.setdefault("loaded_run_dataset_path", None)
    st.session_state.setdefault("ui_theme", "Dark")


def _render_theme_picker(*, in_sidebar: bool) -> None:
    current_theme = str(st.session_state.get("ui_theme", "Dark"))
    if in_sidebar:
        theme_choice = st.sidebar.radio(
            "Theme",
            ["Dark", "Light"],
            index=0 if current_theme == "Dark" else 1,
            horizontal=True,
            key="theme_picker_sidebar",
        )
    else:
        _, right_col = st.columns([0.76, 0.24], gap="small")
        with right_col:
            theme_choice = st.radio(
                "Theme",
                ["Dark", "Light"],
                index=0 if current_theme == "Dark" else 1,
                horizontal=True,
                key="theme_picker_auth",
            )
    if theme_choice != current_theme:
        st.session_state["ui_theme"] = theme_choice
        st.rerun()


def _clear_auth_session() -> None:
    st.session_state["auth_user_id"] = None
    st.session_state["auth_last_seen"] = None
    st.session_state["auth_expires_at"] = None
    st.session_state["selected_run_dir"] = None
    st.session_state["selected_run_question"] = ""
    st.session_state["loaded_run_dataset_path"] = None


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
    ttl = timedelta(minutes=int(auth_cfg.AUTH_SESSION_TTL_MIN))
    expires_raw = st.session_state.get("auth_expires_at")
    expires_at: datetime | None = None
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(str(expires_raw))
        except Exception:
            expires_at = None
    if expires_at is None:
        last_seen_raw = st.session_state.get("auth_last_seen")
        try:
            fallback_anchor = datetime.fromisoformat(str(last_seen_raw)) if last_seen_raw else now
        except Exception:
            fallback_anchor = now
        expires_at = fallback_anchor + ttl
        st.session_state["auth_expires_at"] = expires_at.isoformat()
    if now >= expires_at:
        _clear_auth_session()
        st.session_state["auth_expired_notice"] = "Session expired. Please sign in again."
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
          <p class="auth-kicker">Workspace Access</p>
          <h2 class="auth-title">Secure Workspace Access</h2>
          <p class="auth-note">Sign in or create an account to continue.</p>
          <div class="auth-chip-row">
            <span class="auth-chip">Protected sessions</span>
            <span class="auth-chip">Tracked analysis runs</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center_col, _ = st.columns([0.26, 0.48, 0.26], gap="small")
    with center_col:
        st.caption("Use your workspace credentials to continue.")
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
                        now = _utc_now()
                        ttl = timedelta(minutes=int(_get_auth_cfg().AUTH_SESSION_TTL_MIN))
                        st.session_state["auth_last_seen"] = now.isoformat()
                        st.session_state["auth_expires_at"] = (now + ttl).isoformat()
                        st.session_state["auth_expired_notice"] = ""
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
    expires_raw = st.session_state.get("auth_expires_at")
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(str(expires_raw))
            remaining = expires_at - _utc_now()
            if timedelta(0) < remaining <= timedelta(minutes=5):
                remaining_minutes = max(1, int(remaining.total_seconds() // 60) + 1)
                st.sidebar.warning(
                    f"You will be logged out soon ({remaining_minutes} minute(s) remaining)."
                )
        except Exception:
            pass
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
        run_lookup = {rr.run_uuid: rr for rr in recent_runs}
        selected_uuid = st.sidebar.selectbox(
            "Open Previous Run",
            options=[""] + list(run_lookup.keys()),
            format_func=lambda run_uuid: (
                ""
                if not run_uuid
                else _short_question(run_lookup[run_uuid], limit=72)
            ),
            index=0,
        )
        if selected_uuid and st.sidebar.button("Load Selected Run", width="stretch"):
            detail = run_store.get_run_details_for_user(
                user_id=int(user.id),
                run_uuid=selected_uuid,
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
                    loaded_csv = run_dir / "cleaned.csv"
                    st.session_state["loaded_run_dataset_path"] = (
                        str(loaded_csv) if loaded_csv.exists() else None
                    )
                    st.rerun()

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.subheader("Source Preview")
        loaded_run_dataset_path = st.session_state.get("loaded_run_dataset_path")
        if loaded_run_dataset_path:
            loaded_path = Path(str(loaded_run_dataset_path))
            if loaded_path.exists():
                st.caption(f"Loaded from previous run dataset: {loaded_path.name}")
                try:
                    loaded_df = pd.read_csv(loaded_path)
                    st.dataframe(loaded_df.head(20), width="stretch")
                except Exception as e:
                    st.warning(f"Could not preview loaded run dataset: {sanitize_user_error_message(e)}")
            else:
                st.session_state["loaded_run_dataset_path"] = None

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

        st.session_state["loaded_run_dataset_path"] = None
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
            label_question = str(selected_run_question).strip() or run_path.name
            st.info(f'Showing results for "{label_question}"')
            _render_results(run_path, str(selected_run_question))
        else:
            st.warning("Selected run folder no longer exists.")
    else:
        st.info("Ready. Configure source + question, then run.")


def main() -> None:
    _init_session_state()
    _inject_styles(theme=str(st.session_state.get("ui_theme", "Dark")).lower())

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
        _render_theme_picker(in_sidebar=False)
        expired_notice = str(st.session_state.get("auth_expired_notice", "")).strip()
        if expired_notice:
            st.warning(expired_notice)
            st.session_state["auth_expired_notice"] = ""
        _render_auth_gate(auth)
        return
    _render_theme_picker(in_sidebar=True)
    _render_workspace(user, run_store)


if __name__ == "__main__":
    main()
