from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal

TaskName = Literal["intake", "profiling", "quality", "wrangling", "eda", "reporting", "reviewer"]

@dataclass(frozen=True)
class Task:
    name: TaskName
    reason: str

def default_tasks_phase2() -> List[Task]:
    return [
        Task("intake", "Clarify question and analysis plan"),
        Task("profiling", "Understand schema and dataset structure"),
        Task("quality", "Run data quality checks"),
        Task("wrangling", "Prepare analysis-ready dataset"),
        Task("eda", "Generate EDA summary and charts"),
        Task("reporting", "Write artifact-grounded report (LLM)"),
        Task("reviewer", "Validate report claims against evidence"),
    ]