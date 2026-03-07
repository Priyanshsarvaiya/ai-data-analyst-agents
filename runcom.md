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

## 3) Run CSV Pipeline

`python -m ai_data_analyst_agents.pipelines.run_csv_pipeline --file data/sample_ecommerce_data.csv --question "Why did India have less revenue than other countries?"`

## 4) Run SQL Pipeline (SQLite)

`python -m ai_data_analyst_agents.pipelines.run_sql_pipeline --db-url "sqlite:///data/sample_ecommerce.db" --table orders --question "Why did India have less revenue than other countries?"`

## 5) Run SQL Pipeline (PostgreSQL)

`python -m ai_data_analyst_agents.pipelines.run_sql_pipeline --db-url "postgresql+psycopg://USER:PASSWORD@HOST:5432/DB_NAME" --table orders --question "Which country has the highest revenue and why?"`

## 6) Code Quality

Lint check:

`ruff check ai_data_analyst_agents app --output-format concise`

Compile check:

`python -m compileall ai_data_analyst_agents app`

## 7) Run Tests

Run all tests:

`pytest -q tests`

Run a single test file:

`pytest -q tests/test_pipeline_csv_e2e.py`
