from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80]


def save_histogram(df: pd.DataFrame, col: str, out_dir: Path) -> Optional[str]:
    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    if s.empty:
        return None

    plt.figure()
    s.hist()
    out_path = out_dir / f"hist_{sanitize_filename(col)}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name


def save_bar_top_categories(df: pd.DataFrame, col: str, out_dir: Path, top_k: int = 15) -> Optional[str]:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None

    vc = s.astype(str).value_counts().head(top_k)
    if vc.empty:
        return None

    plt.figure()
    vc.sort_values().plot(kind="barh")
    plt.xlabel("count")
    plt.ylabel(col)

    out_path = out_dir / f"bar_{sanitize_filename(col)}_top{top_k}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name