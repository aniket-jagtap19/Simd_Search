"""Shared helpers for the plotting scripts. Everything is computed from recorded CSV files."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402

COLORS = plt.get_cmap("tab10").colors


def load_meta(path):
    p = pathlib.Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def cpus_from_meta(meta, default=None):
    try:
        return int(meta["cpus_available_to_process"])
    except (KeyError, ValueError):
        return default


def style_workers_axis(ax, workers, cpus=None):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(workers)))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.grid(True, alpha=0.3)
    if cpus is not None and max(workers) > cpus:
        ax.axvspan(cpus * 1.03, max(workers) * 1.2, color="red", alpha=0.07, lw=0)
        ax.text(0.98, 0.02, f"red zone: workers > {cpus} CPU(s)\n(oversubscribed)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="firebrick")


def plot_series(ax, x, y, over, label, color, marker="o", ls="-"):
    """Line through all points; oversubscribed points are drawn hollow."""
    ax.plot(x, y, ls=ls, color=color, marker=marker, label=label, ms=5, lw=1.5)
    over = np.asarray(over, dtype=bool)
    if over.any():
        xs, ys = np.asarray(x)[over], np.asarray(y)[over]
        ax.plot(xs, ys, ls="none", marker=marker, mfc="white", mec=color, ms=6, zorder=5)


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(fig, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)
