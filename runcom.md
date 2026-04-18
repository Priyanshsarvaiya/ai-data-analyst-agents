# Run Commands

## 1) Setup

Create virtual environment:

`python3 -m venv venv`

Activate virtual environment (macOS/Linux):

`source venv/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

## 2) Run Streamlit App

`streamlit run app/streamlit_app.py`

Optional auth hardening env vars:

`export AUTH_PASSWORD_PEPPER="replace-with-long-random-secret"`

`export AUTH_SESSION_TTL_MIN=240`

`export MAX_UPLOAD_MB=50`

Postgres auth DB connection (used by Streamlit login/signup):

`export AUTH_DATABASE_URL="postgresql+psycopg://ai_analyst_app:change-this-strong-password@localhost:5432/ai_analyst"`

## 3) Run CSV Pipeline

`python -m ai_data_analyst_agents.pipelines.run_csv_pipeline --file data/sample_ecommerce_data.csv --question "Why did India have less revenue than other countries?"`

## 4) Run SQL Pipeline (SQLite)

`python -m ai_data_analyst_agents.pipelines.run_sql_pipeline --db-url "sqlite:///data/sample_ecommerce.db" --table orders --question "Why did India have less revenue than other countries?"`

## 5) Run SQL Pipeline (PostgreSQL)

`python -m ai_data_analyst_agents.pipelines.run_sql_pipeline --db-url "postgresql+psycopg://USER:PASSWORD@HOST:5432/DB_NAME" --table orders --question "Which country has the highest revenue and why?"`

## 6) Phase 4 Statistical Examples

A/B conversion example:

`python -m ai_data_analyst_agents.pipelines.run_csv_pipeline --file data/ab_conversion_demo.csv --question "Did treatment improve conversion versus control?"`

Mean comparison example:

`python -m ai_data_analyst_agents.pipelines.run_csv_pipeline --file data/mean_comparison_demo.csv --question "Is average order value different between segment A and segment B?"`

OLS regression example:

`python -m ai_data_analyst_agents.pipelines.run_csv_pipeline --file data/regression_demo.csv --question "Which variables are most associated with revenue? Use regression."`

## 7) Code Quality

Lint check:

`ruff check ai_data_analyst_agents app --output-format concise`

Compile check:

`python -m compileall ai_data_analyst_agents app`

## 8) Run Tests

Run all tests:

`pytest -q tests`

Run a single test file:

`pytest -q tests/test_pipeline_csv_e2e.py`


Need to do this if folder change:
`deactivate 2>/dev/null`
`rm -rf venv`
`python3 -m venv venv`
`source venv/bin/activate`
`which python`
`which pip`
`python -m pip install --upgrade pip`
`python -m pip install -r requirements.txt`