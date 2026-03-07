# AI Data Analyst Agents 🤖📊

A production-oriented multi-agent system that simulates a real-world
data analytics team.\
This framework mirrors how professional data analysts work --- from
problem scoping to data validation, analysis, and executive-ready
reporting --- with strict artifact-based guardrails to prevent
hallucinated insights.

------------------------------------------------------------------------

## 🌟 Overview

**AI Data Analyst Agents** is a modular multi-agent analytics system designed to:

- Turn raw datasets (CSV / SQL) into validated insights
- Enforce reproducibility and analytical integrity
- Generate executive-ready reports with traceable evidence
- Simulate a real data analyst workflow

Unlike simple “ask CSV questions” tools, this system follows a structured pipeline:

> Question → Profiling → Data Quality → Cleaning → EDA → Stats → Insights → Report → Review

Every claim in the final report must reference a computed artifact (table, metric, chart, or query result).

------------------------------------------------------------------------

# 🚀 Running AI Data Analyst Agents (Local Setup)

## 1️⃣ Clone the Repository

``` bash
git clone https://github.com/Priyanshsarvaiya/ai-data-analyst-agents.git
cd ai-data-analyst-agents
```

------------------------------------------------------------------------

## 2️⃣ Create a Virtual Environment

### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

------------------------------------------------------------------------

## 3️⃣ Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

``` bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key and model:

``` env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=z-ai/glm-5
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=ai-data-analyst-agents

ENV=local
ARTIFACTS_DIR=artifacts
LOG_LEVEL=INFO
```

------------------------------------------------------------------------

## 5️⃣ Add Your Dataset

Place your CSV file inside the `data/` folder:

    ai-data-analyst-agents/
      data/
        your_dataset.csv

------------------------------------------------------------------------

## 6️⃣ Run the Analysis Pipeline

``` bash
python -m ai_data_analyst_agents.pipelines.run_csv_pipeline   --file data/your_dataset.csv   --question "What insights can we derive from this dataset?"
```

For SQL sources (SQLite/PostgreSQL):

``` bash
python -m ai_data_analyst_agents.pipelines.run_sql_pipeline   --db-url "sqlite:///data/your.db"   --question "What insights can we derive from this database?"
```

------------------------------------------------------------------------

## 7️⃣ View Results

After execution, a new folder will be created inside:

    artifacts/
      run_YYYYMMDD_HHMMSS/

Inside you will find:

-   analysis_plan.json
-   data_profile.json
-   quality_report.json
-   quality_warnings.md
-   cleaned.csv
-   feature_log.json
-   eda_summary.json
-   charts/
-   final_report.md
-   review_log.json
-   logs.txt

Open `final_report.md` to see the generated analysis report.

------------------------------------------------------------------------

## 🎯 Design Philosophy

This project is built around 5 principles:

1. **Artifact-First Reporting** – No insight without evidence.
2. **Multi-Agent Specialization** – Each agent has a clearly defined responsibility.
3. **Reproducibility** – One command generates the full analysis folder.
4. **Data Integrity First** – Data quality validation before analysis.
5. **Business Framing** – Insights must answer a stakeholder question.

------------------------------------------------------------------------

# 🏗️ Architecture

## Multi-Agent System

The system consists of the following specialized agents:

---

### 1️⃣ Intake Agent (Scoping Agent)

**Purpose:** Clarifies the business question before analysis begins.

**Responsibilities:**
- Define KPI(s)
- Define time window
- Define segmentation
- Identify dataset grain (1 row = ?)
- Produce structured analysis plan

**Output:**
- `analysis_plan.json`

---

### 2️⃣ Data Profiling Agent

**Purpose:** Understand dataset structure and schema.

**Responsibilities:**
- Infer column types
- Identify potential keys
- Detect dataset grain
- Generate summary statistics
- Create data dictionary

**Output:**
- `data_profile.json`

---

### 3️⃣ Data Quality Agent

**Purpose:** Validate data reliability before analysis.

**Checks Include:**
- Missing values %
- Duplicate rows
- Impossible values (negative revenue, etc.)
- Outliers (IQR / Z-score)
- Schema drift detection
- Range validation
- Null-heavy columns

**Output:**
- `quality_report.json`
- `quality_warnings.md`

---

### 4️⃣ Data Wrangling Agent

**Purpose:** Prepare analysis-ready dataset.

**Responsibilities:**
- Handle missing values
- Remove duplicates
- Standardize formats
- Feature engineering
- Cohort creation
- Derived KPIs

**Output:**
- `cleaned.csv`
- `feature_log.json`

---

### 5️⃣ EDA Agent (Exploratory Data Analysis)

**Purpose:** Discover patterns and trends.

**Produces:**
- Summary statistics
- Distribution plots
- Correlation matrix
- Segment comparisons
- Time series trends
- Funnel or cohort analysis

**Output:**
- `/charts/`
- `eda_summary.json`

---

### 6️⃣ Statistics / Experiment Agent (Optional)

**Purpose:** Perform statistical validation when required.

**Includes:**
- Hypothesis testing
- Confidence intervals
- Effect size
- Basic regression (OLS)
- A/B test evaluation

**Output:**
- `stat_results.json`

---

### 7️⃣ Insights Agent

**Purpose:** Convert numbers into business insights.

**Responsibilities:**
- Identify key findings
- Explain metric changes
- Suggest business actions
- Estimate potential impact
- Highlight limitations

**Output:**
- `insights.md`

---

### 8️⃣ Reviewer Agent (Guardrails)

**Purpose:** Prevent hallucinations and unsupported claims.

**Validates:**
- Every statement maps to artifact
- No metric invented
- Charts referenced correctly
- Statistical claims justified

**Output:**
- `review_log.json`

------------------------------------------------------------------------

# 📁 Project Structure

```
ai-data-analyst-agents/
  README.md
  app/                    # Streamlit / web UI
  agents/
    intake.py
    profiling.py
    quality.py
    wrangling.py
    eda.py
    stats.py
    reporting.py
    reviewer.py
  tools/
    pandas_tools.py
    plotting_tools.py
    sql_tools.py
    validation_tools.py
  pipelines/
    run_csv_pipeline.py
    run_sql_pipeline.py
  artifacts/
    (generated outputs)
  tests/
    test_quality_checks.py
  configs/
    rules.yaml            # guardrails, KPI templates
```

------------------------------------------------------------------------

# 🗺️ Development Roadmap

---

## 🧭 Phase-by-Phase Implementation Plan

---

## 🟢 Phase 1 – MVP (Single CSV, Single Pipeline)

### 🎯 Goal
Build a fully automated **CSV → Report** analytics system.

### 📦 Deliverables
- CSV ingestion pipeline
- Data Profiling Agent
- Data Quality validation checks
- Basic EDA visualizations
- Auto-generated Markdown report
- Artifact validation system

### ✅ Success Criteria
- One command produces a complete `/artifacts/` folder
- Report references only real computed values
- No unsupported claims in insights

---

## 🟡 Phase 2 – Structured Multi-Agent Orchestration

### 🎯 Goal
Enable true multi-agent collaboration with modular architecture.

### ➕ Add
- Agent communication layer
- Shared memory system
- Task Planner Agent
- Evidence-linking system
- Reviewer / Guardrail Agent

### ✅ Success Criteria
- Agents operate independently but collaboratively
- Claims are validated against artifacts
- Pipeline remains modular and extensible

---

## 🟠 Phase 3 – SQL + Business Context Support

### 🎯 Goal
Support real-world analytics workflows with structured databases.

### ➕ Add
- SQL query generation
- Schema-aware query planner
- Join reasoning logic
- KPI template library (SaaS, Ecommerce, Marketing, etc.)
- Business metric definition engine

### ✅ Success Criteria
- Multi-table datasets can be analyzed
- Stakeholder-style executive summaries generated
- Business metrics computed consistently

---

## 🔵 Phase 4 – Statistical Intelligence Layer

### 🎯 Goal
Introduce statistical rigor and experiment evaluation.

### ➕ Add
- Hypothesis testing module
- Confidence interval computation
- Effect size estimation
- Basic regression models (OLS)
- A/B testing workflow

### ✅ Success Criteria
- All statistical claims include assumptions
- No false or exaggerated significance
- Statistical limitations explicitly documented in reports

---

## 🟣 Phase 5 – Dashboard + Interactive Mode

### 🎯 Goal
Enable human-agent collaboration and interactive refinement.

### ➕ Add
- Streamlit web interface
- Interactive artifact viewer
- Follow-up Q&A over computed artifacts
- Editable report sections
- Chart customization
- Agent monitoring panel

### ✅ Success Criteria
- Analysts can refine and adjust output
- Stakeholders can ask follow-up questions
- Interactive insights remain artifact-grounded

---

## 🔴 Phase 6 – Advanced & Production Features

### ➕ Add
- Automated anomaly detection
- Data drift monitoring
- Scheduled report generation
- Model-based forecasting
- Cloud deployment support
- Multi-user authentication & access control
- Full audit logs for reproducibility

---

# 🧪 Testing Strategy

- Unit tests for data quality checks
- Snapshot tests for report consistency
- Schema validation tests
- Artifact-reference validation tests
- Regression tests on benchmark datasets
- End-to-end pipeline tests

---

🚀 *From raw data to defensible decisions — systematically and reproducibly.*

------------------------------------------------------------------------

# 🛠️ Technology Stack

### Core
- Python 3.10+
- Pandas
- NumPy
- SciPy
- Matplotlib / Plotly

### Validation
- Great Expectations / Pandera

### Agent Framework
- LangGraph / CrewAI / Custom Orchestrator

### Execution
- Docker sandbox for code execution

### Storage
- JSON artifacts
- Markdown reports
- CSV outputs

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Open a Pull Request

------------------------------------------------------------------------

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

------------------------------------------------------------------------

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

------------------------------------------------------------------------

## 👨‍💻 Author

**Priyansh Sarvaiya**
- GitHub: [@Priyanshsarvaiya](https://github.com/Priyanshsarvaiya)

------------------------------------------------------------------------

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Inspired by the latest developments in AI agents and autonomous systems

------------------------------------------------------------------------

## 📮 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Reach out via GitHub discussions

------------------------------------------------------------------------

## 🗺️ Roadmap

### 🟢 Phase 1 – MVP: End-to-End CSV Analytics Pipeline
- [✅] Implement CSV ingestion pipeline
- [✅] Add Intake (Scoping) Agent
- [✅] Implement Data Profiling Agent
- [✅] Implement Data Quality validation checks
- [✅] Build Data Cleaning & Feature Engineering Agent
- [✅] Generate automated EDA charts
- [✅] Auto-generate structured Markdown report
- [✅] Enforce artifact-based reporting (no unsupported claims)
- [✅] One-command pipeline execution → `/artifacts/` folder output

---

### 🟡 Phase 2 – True Multi-Agent Orchestration
- [✅] Introduce structured agent communication layer
- [✅] Add shared memory between agents
- [✅] Implement task planning & delegation logic
- [✅] Add Reviewer / Guardrail Agent for claim validation
- [✅] Evidence-linking system (every claim references artifact)
- [✅] Improve modularity for agent swapping/extending

---

### 🟠 Phase 3 – SQL & Business Context Expansion
- [✅] Add SQL data source support (PostgreSQL, SQLite)
- [✅] Schema-aware query generation
- [✅] Multi-table join reasoning
- [✅] KPI template library (SaaS, Ecommerce, Marketing, Ops)
- [✅] Business metric definition engine
- [✅] Segment & cohort analysis templates

---

### 🔵 Phase 4 – Statistical Intelligence Layer
- [ ] Hypothesis testing module
- [ ] Confidence interval reporting
- [ ] Effect size calculation
- [ ] A/B testing workflow
- [ ] Basic regression (OLS) integration
- [ ] Assumption validation & statistical guardrails
- [ ] Explicit statistical limitations section in reports

---

### 🟣 Phase 5 – Interactive Dashboard & Human-Agent Collaboration
- [ ] Streamlit web interface
- [ ] Interactive artifact viewer
- [ ] Follow-up Q&A over computed results
- [ ] Editable report sections
- [ ] Chart customization support
- [ ] Agent activity monitoring panel
- [ ] User feedback loop for refinement

---

### 🔴 Phase 6 – Advanced & Production-Ready Features
- [ ] Automated anomaly detection
- [ ] Data drift monitoring
- [ ] Scheduled report generation
- [ ] Forecasting module
- [ ] Cloud deployment support
- [ ] Multi-user support & access control
- [ ] Audit logs for reproducibility
- [ ] Performance benchmarking against human baseline

------------------------------------------------------------------------

**Note**: This project is under active development. Features and documentation may change.
**Note**: This project is designed as a serious analytics engineering system — not just a chatbot over CSV. It aims to demonstrate production-grade data reasoning with AI agents.
