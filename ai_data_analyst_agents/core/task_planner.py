from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal

TaskName = Literal["intake", "profiling", "quality", "wrangling", "eda", "reporting", "reviewer", "planner", "metrics"]

@dataclass(frozen=True)
class Task:
    name: TaskName
    reason: str

def default_tasks_phase2() -> List[Task]:
    return [
        Task(
            "intake",
            "Interpret and formalize the business question into an explicit analysis objective."
        ),
        Task(
            "profiling",
            "Inspect dataset schema, column types, distributions, and structural characteristics."
        ),
        Task(
            "quality",
            "Assess data reliability by checking missingness, duplicates, outliers, and integrity risks."
        ),
        Task(
            "wrangling",
            "Prepare cleaned and analysis-ready dataset with consistent types and derived fields."
        ),
        Task(
            "eda",
            "Generate descriptive statistics and exploratory visualizations to understand data patterns."
        ),

        # Phase 2 – Question-aware intelligence
        Task(
            "planner",
            "Determine the specific computations, aggregations, and comparisons required to answer the business question."
        ),
        Task(
            "metrics",
            "Execute planned analytical computations, produce metric artifacts, and register evidence references."
        ),

        Task(
            "reporting",
            "Synthesize findings into a structured, evidence-backed analytical report answering the business question."
        ),
        Task(
            "reviewer",
            "Validate report claims against registered evidence and enforce artifact-grounded reasoning."
        ),
    ]