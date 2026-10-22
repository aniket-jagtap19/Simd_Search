#!/usr/bin/env python3
"""Draws the architecture / data-flow diagrams used in docs/REPORT.md."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path("docs")
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="#dbe9f6", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7, color="#a00")


# ---- 1. SimdSearch ----
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 0.2, 4.9, 2.4, 0.9, "Dataset\n(synthetic seed | .bin | .csv)\nvalidate: finite, shape", "#fff2cc")
box(ax, 0.2, 3.5, 2.4, 0.9, "Index build (untimed)\ncontiguous float32 n x dim\n+ double norms[n]", "#fff2cc")
box(ax, 0.2, 2.1, 2.4, 0.9, "Query + K + metric\nvalidate dim, K>=1,\nfinite values", "#fff2cc")
arrow(ax, 1.4, 4.9, 1.4, 4.4); arrow(ax, 1.4, 3.5, 1.4, 3.0)
ax.text(1.4, 1.6, "Timed region = everything to the right", ha="center", fontsize=8, style="italic")
box(ax, 3.4, 4.3, 3.0, 1.3, "SERIAL\nfor i in 0..n-1:\n  score(i) -> one bounded heap", "#e2f0d9")
arrow(ax, 2.6, 2.6, 3.4, 4.6)
box(ax, 3.4, 0.4, 3.0, 3.5, "OPENMP  (T threads)\n#pragma omp parallel\n#pragma omp for schedule(static)\n\nthread 0: rows [0, n/T)  -> heap0\nthread 1: rows [n/T, 2n/T) -> heap1\n   ...           (private heaps,\nthread T-1: ...           no locks)\n\nimplicit barrier", "#e2f0d9")
arrow(ax, 2.6, 2.4, 3.4, 2.2)
box(ax, 7.2, 0.9, 2.2, 1.6, "MERGE (1 thread)\nconcat <= T*K entries\npartial_sort -> K", "#fce4d6")
arrow(ax, 6.4, 2.2, 7.2, 1.8, "T heaps")
box(ax, 7.2, 4.3, 2.2, 1.3, "Finalize\npartial_sort -> K", "#fce4d6")
arrow(ax, 6.4, 4.9, 7.2, 4.9)
box(ax, 10.0, 2.6, 1.8, 1.3, "Top-K\n(id, score)\nties: smaller id", "#d9d2e9")
arrow(ax, 9.4, 4.8, 10.4, 3.9); arrow(ax, 9.4, 1.8, 10.4, 2.6)
box(ax, 10.0, 0.6, 1.8, 1.3, "Verify\nIDs exact,\nscore tol 1e-9", "#d9d2e9")
arrow(ax, 10.9, 2.6, 10.9, 1.9)
ax.set_title("SimdSearch architecture: serial vs OpenMP exact Top-K", fontsize=11)
fig.savefig(OUT / "simdsearch_architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 2. MPI matvec data flow ----
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
box(ax, 0.2, 2.0, 2.2, 1.6, "Rank 0\nA (M x N), x\n(generated, serial\nreference y_ref)", "#fff2cc")
for i, y in enumerate([4.0, 2.6, 1.2, -0.1]):
    pass
ranks = [("Rank 0", 4.2), ("Rank 1", 2.9), ("Rank 2", 1.6), ("Rank P-1", 0.3)]
for name, y in ranks:
    box(ax, 5.0, y, 3.0, 0.95, f"{name}\nA_local (rows_r x N), x\ny_local = A_local * x", "#e2f0d9")
    arrow(ax, 2.4, 2.8, 5.0, y + 0.5)
    arrow(ax, 8.0, y + 0.5, 9.6, 2.8)
ax.text(3.7, 4.9, "MPI_Scatterv(A rows)\nMPI_Bcast(x)", ha="center", fontsize=8, color="#a00")
ax.text(8.8, 4.9, "MPI_Gatherv(y)", ha="center", fontsize=8, color="#a00")
box(ax, 9.6, 2.0, 2.2, 1.6, "Rank 0\ny (M)\nverify vs y_ref\n(tol 1e-12)", "#d9d2e9")
ax.text(6.5, 0.0, "rows split in contiguous blocks; first (M mod P) ranks get 1 extra row; P > M legal (empty ranks)",
        ha="center", fontsize=8, style="italic")
ax.set_title("MPI matrix-vector multiplication: row-block decomposition", fontsize=11)
fig.savefig(OUT / "matvec_mpi_dataflow.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote docs/simdsearch_architecture.png, docs/matvec_mpi_dataflow.png")
