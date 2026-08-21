<p align="center">
  <img src="docs/assets/ai-data-analyst-agents-hero.png" alt="AI Data Analyst Agents" width="900">
</p>

<p align="center">
  <strong>From raw data to defensible decisions.</strong><br>
  A multi-agent analytics system that profiles, validates, analyzes, reviews, and reports—with evidence attached.
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Web-Flask-111827?logo=flask&logoColor=white">
  <img alt="Type checked with Pyright" src="https://img.shields.io/badge/types-Pyright-3178C6">
</p>

---

## What this project does

AI Data Analyst Agents turns a business question and a CSV or read-only SQL source into a reproducible analysis workspace. Instead of asking one model to improvise an answer, specialized agents exchange structured artifacts through shared memory, compute results in Python, and require the final narrative to cite real evidence.

It is built for questions such as:

- Why did conversion decline, and which segments drove the change?
- Did treatment improve conversion compared with control?
- Which variables are associated with revenue?
- What data-quality issues could invalidate this analysis?

The result is not just a chat response. Every run produces its plan, cleaned data, metrics, statistical outputs, charts, report, review findings, message history, and audit trail.

## Highlights

| Capability | What it provides |
|---|---|
| Evidence-first reporting | Report claims resolve to computed metrics, tables, charts, or statistical artifacts. |
| Coordinated specialist agents | Scoping, profiling, quality, wrangling, planning, metrics, EDA, insights, reporting, review, and scoring have separate responsibilities. |
| CSV and SQL analysis | Analyze uploaded files or read-only SQLite/PostgreSQL sources. |
| Statistical guardrails | Assumption checks, confidence intervals, effect sizes, hypothesis tests, A/B analysis, and robust OLS diagnostics. |
| Business-aware KPIs | KPI templates cover general, ecommerce, SaaS, marketing, operations, finance, product, support, and people analytics. |
| Shared-memory auditability | Agent reads, writes, messages, revisions, and final facts are recorded for inspection. |
| Flask workspace | Sign in, upload and preview CSVs, add more data, launch analyses, and browse generated artifacts. |
| Review and score gates | Unsupported language, missing computations, weak citations, and incomplete coverage remain visible instead of being rewritten away. |

## How it works

```mermaid
flowchart LR
    A[Business question<br/>CSV or SQL] --> B[Intake]
    B --> C[Profile & quality]
    C --> D[Wrangle]
    D --> E[Plan]
    E --> F[Metrics & statistics]
    F --> G[Gap-closing next steps]
    G --> H[EDA & insights]
    H --> I[Report]
    I --> J{Reviewer}
    J -->|Evidence-safe revision| I
    J --> K[Scorecard & run artifacts]
    M[(Shared memory<br/>& messages)] --- B
    M --- E
    M --- F
    M --- H
    M --- I
    M --- J
```

> Intake → Profiling → Quality → Wrangling → Planner → Metrics → Next Steps → EDA → Insights → Reporting → Reviewer → Scorecard

| Stage | Primary responsibility | Key output |
|---|---|---|
| Intake | Frame the question, grain, KPIs, segments, and time window | `analysis_plan.json` |
| Profiling | Infer schema, candidate keys, distributions, and dataset shape | `data_profile.json` |
| Quality | Find missingness, duplicates, invalid ranges, and outliers | `quality_report.json` |
| Wrangling | Apply traceable cleaning and feature engineering | `cleaned.csv`, `feature_log.json` |
| Planner | Convert the question into executable analytical tasks | `analysis_tasks.json` |
| Metrics | Compute descriptive, diagnostic, and statistical results | `metrics_outputs.json`, `statistics/` |
| Next Steps | Identify and execute bounded gap-closing work | `next_steps_plan.json`, `next_steps_metrics_outputs.json` |
| EDA | Produce summaries and decision-relevant visualizations | `eda_summary.json`, `charts/` |
| Insights | Gate question-level analytical coverage | `analysis_readiness.json` |
| Reporting | Assemble an executive-ready, citation-bearing report | `final_report.md` |
| Reviewer | Validate evidence, wording, coverage, and statistical claims | `review_log.json` |
| Scorecard | Summarize run quality and remaining deficiencies | `run_scorecard.json` |

## Quick start

### 1. Install

```bash
git clone https://github.com/Priyanshsarvaiya/ai-data-analyst-agents.git
cd ai-data-analyst-agents

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Add your OpenRouter key and select a compatible model:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=z-ai/glm-5.2
LLM_PLANNER_MAX_TOKENS=8192
LLM_REPORT_MAX_TOKENS=16384
LLM_MAX_ATTEMPTS=4
```

For the web app, also configure authentication storage and secrets:

```env
AUTH_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DB_NAME
AUTH_PASSWORD_PEPPER=replace_with_a_long_random_secret
FLASK_SECRET_KEY=replace_with_a_different_long_random_secret
```

Environment variables override values in `configs/settings.yaml`. Keep secrets out of source control.

### 3. Launch the web workspace

```bash
python -m flask --app app.flask_app:create_app run --debug
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), sign in, and start an analysis. The Flask interface supports CSV drag-and-drop, an in-browser data preview, additional CSV uploads, read-only SQL connections, run progress, and artifact browsing.

## Run from the CLI

### CSV

```bash
python -m ai_data_analyst_agents.pipelines.run_csv_pipeline \
  --file data/sample_ecommerce_data.csv \
  --question "Which segments are driving revenue and conversion?"
```

### SQL

```bash
python -m ai_data_analyst_agents.pipelines.run_sql_pipeline \
  --db-url "sqlite:///data/sample_ecommerce.db" \
  --question "Which products and customer segments drive performance?"
```

SQL access is constrained to read-only analysis. The source layer validates statements, applies row limits, and supports SQLite and PostgreSQL.

### Statistical examples

```bash
# Two-proportion / A/B analysis
python -m ai_data_analyst_agents.pipelines.run_csv_pipeline \
  --file data/ab_conversion_demo.csv \
  --question "Did treatment improve conversion versus control?"

# Mean comparison
python -m ai_data_analyst_agents.pipelines.run_csv_pipeline \
  --file data/mean_comparison_demo.csv \
  --question "Is average order value different between segment A and segment B?"

# Robust OLS regression
python -m ai_data_analyst_agents.pipelines.run_csv_pipeline \
  --file data/regression_demo.csv \
  --question "Which variables are most associated with revenue? Use regression."
```

## Statistical intelligence

The task selector chooses methods from the question and available columns. Supported analysis includes:

- Welch and paired mean comparisons
- Mann–Whitney tests
- Chi-square and Fisher exact tests
- Two-proportion tests for conversion experiments
- Confidence intervals and effect sizes
- OLS regression with HC3 robust standard errors
- Assumption diagnostics and explicit limitations

Statistical tasks write self-contained evidence bundles:

```text
statistics/<task_id>_<method>/
├── summary.json
├── assumptions.json
├── results.md
├── coefficients.csv      # regression only
└── diagnostics.json      # when applicable
```

Association is not presented as causation. Experimental and causal language is allowed only when the design and evidence support it.

## Run artifacts

Each execution creates an isolated, timestamped directory:

```text
artifacts/run_YYYYMMDD_HHMMSS/
├── analysis_plan.json
├── data_profile.json
├── quality_report.json
├── quality_warnings.md
├── cleaned.csv
├── feature_log.json
├── analysis_tasks.json
├── metrics_outputs.json
├── next_steps_plan.json
├── next_steps_metrics_outputs.json
├── eda_summary.json
├── analysis_readiness.json
├── final_report.md
├── report_metadata.json
├── review_log.json
├── run_scorecard.json
├── agent_messages.json
├── shared_memory_audit.json
├── run_manifest.json
├── charts/
└── statistics/
```

`shared_memory_audit.json` shows which facts each agent read and wrote. `agent_messages.json` captures collaboration events. The manifest and review artifacts make it possible to trace a final claim back through the pipeline to its computed source.

## Token budgets

The planner and reporter use separate output budgets:

| Setting | Default | Purpose |
|---|---:|---|
| `LLM_PLANNER_MAX_TOKENS` | 8,192 | Structured plans and analytical task definitions |
| `LLM_REPORT_MAX_TOKENS` | 16,384 | Full evidence-grounded reports |
| `llm.max_tokens` | 16,384 | Compatibility fallback |

These are maximum **output** tokens, not prompt limits. The selected provider model must fit both the prompt and requested output within its context window. Increase a budget only when the corresponding metadata reports a length-based truncation; larger values increase latency and cost but do not automatically improve reasoning quality.

Provider-reported usage and finish reasons are saved in `planner_llm_metadata.json` and `report_llm_metadata.json`.

## Security model

- SQL inputs are restricted to read-only statements and bounded result sizes.
- Raw-row exposure to the LLM is disabled by default.
- Upload size and user-facing error detail are configurable.
- Authentication uses PostgreSQL-compatible storage, password hashing, session expiry, and lockout controls.
- Reports must cite artifact-backed evidence; the reviewer rejects unsupported claims.

Review the defaults in `.env.example`, `configs/settings.yaml`, and `configs/rules.yaml` before deploying beyond local development.

## Project structure

```text
.
├── app/                         # Flask routes, auth, templates, CSS, and JavaScript
├── ai_data_analyst_agents/
│   ├── agents/                  # Specialized analytics agents
│   ├── core/                    # Orchestration, memory, evidence, security, settings
│   ├── evaluation/              # Benchmark and quality evaluation harness
│   ├── pipelines/               # CSV and SQL command-line entry points
│   ├── statistics/              # Tests, models, diagnostics, and statistical artifacts
│   └── tools/                   # Pandas, plotting, and validation helpers
├── benchmarks/                  # Evaluation suites
├── configs/                     # Runtime, LLM, QA, SQL, and security configuration
├── data/                        # Sample datasets and databases
├── docs/assets/                 # README and documentation media
├── artifacts/                   # Generated analysis runs
└── tests/                       # Unit, integration, security, and end-to-end tests
```

## Development

Run the automated checks before opening a pull request:

```bash
python -m pytest -q
pyright
ruff check .
```

Contributions are welcome. Please keep changes focused, add tests for new behavior, preserve artifact compatibility where practical, and document user-facing configuration.

## Current scope

Available today:

- End-to-end CSV and SQL analytics
- Shared-memory multi-agent orchestration
- Evidence-linked reports and reviewer feedback loops
- Business KPI templates and statistical testing
- Flask authentication, uploads, previews, run tracking, and artifact viewing
- Evaluation harness and run scorecards

Planned areas include follow-up Q&A over computed evidence, editable report sections, chart customization, scheduled analyses, drift monitoring, forecasting, and deployment tooling.

## Author

Built by [Priyansh Sarvaiya](https://github.com/Priyanshsarvaiya).

---

<p align="center">
  <strong>Serious analytics needs more than a plausible answer—it needs a traceable one.</strong>
</p>
