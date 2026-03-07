from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def save_bar_from_mapping(
    data: dict[str, float],
    out_dir: Path,
    chart_name: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    top_k: int = 20,
) -> Optional[str]:
    if not data:
        return None
    items = []
    for k, v in data.items():
        try:
            items.append((str(k), float(v)))
        except Exception:
            continue
    if not items:
        return None
    items = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    plt.figure(figsize=(8, max(3, min(10, len(items) * 0.35))))
    plt.barh(labels, vals)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    out_path = out_dir / f"{sanitize_filename(chart_name)}.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name


def save_line_from_mapping(
    data: dict[str, float],
    out_dir: Path,
    chart_name: str,
    title: str,
    xlabel: str = "x",
    ylabel: str = "value",
) -> Optional[str]:
    if not data:
        return None
    items = []
    for k, v in data.items():
        try:
            items.append((str(k), float(v)))
        except Exception:
            continue
    if not items:
        return None
    try:
        items = sorted(items, key=lambda kv: pd.to_datetime(kv[0], errors="raise"))
    except Exception:
        items = sorted(items, key=lambda kv: kv[0])

    x = [k for k, _ in items]
    y = [v for _, v in items]
    plt.figure(figsize=(9, 4))
    plt.plot(range(len(x)), y, marker="o")
    tick_idx = np.linspace(0, max(0, len(x) - 1), num=min(10, len(x)), dtype=int).tolist()
    plt.xticks(tick_idx, [x[i] for i in tick_idx], rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_path = out_dir / f"{sanitize_filename(chart_name)}.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name


def save_heatmap_from_rows(
    rows: list[dict],
    out_dir: Path,
    chart_name: str,
    title: str,
    x_col: str,
    y_col: str,
    value_col: str = "value",
) -> Optional[str]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if x_col not in df.columns or y_col not in df.columns or value_col not in df.columns:
        return None
    df = df[[x_col, y_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return None

    pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="sum", fill_value=0.0)
    if pivot.empty:
        return None

    # Keep plot legible.
    pivot = pivot.iloc[:20, :20]
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto", cmap="Blues")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    out_path = out_dir / f"{sanitize_filename(chart_name)}.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name
