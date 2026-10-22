#!/usr/bin/env python3
"""SimdSearch graphs + summary tables from results/simdsearch_benchmark.csv (real measurements only)."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *  # noqa: F401,F403,E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/simdsearch_benchmark.csv")
ap.add_argument("--meta", default="results/simdsearch_meta.json")
ap.add_argument("--outdir", default="results/figures")
ap.add_argument("--k", type=int, default=10, help="K shown in the plots")
ap.add_argument("--dim", type=int, default=None, help="dimension shown in the plots (default: largest)")
args = ap.parse_args()

df = pd.read_csv(args.csv)
if df.empty:
    sys.exit("benchmark CSV is empty")
assert (df["verified"] == 1).all(), "unverified rows present - refusing to plot"
meta = load_meta(args.meta)
cpus = cpus_from_meta(meta)
out = pathlib.Path(args.outdir)
dim = args.dim or int(df["dim"].max())
k = args.k if args.k in set(df["k"]) else int(df["k"].min())
sel = df[(df["dim"] == dim) & (df["k"] == k)]
metrics = sorted(sel["metric"].unique())
sizes = sorted(sel["n"].unique())
omp = sel[sel["impl"] == "openmp"]
ser = sel[sel["impl"] == "serial"]
workers = sorted(omp["workers"].unique()) or [1]
reps, stat = int(df["repeats"].iloc[0]), df["stat"].iloc[0]
subtitle = f"dim={dim}, K={k}, {stat} of {reps} runs, {cpus if cpus else '?'} CPU(s) available; hollow marker = oversubscribed"


def facet(fn_name, ylabel, ycol, ylog=False, ideal=False, hline=None):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for i, n in enumerate(sizes):
            d = omp[(omp["metric"] == m) & (omp["n"] == n)].sort_values("workers")
            if d.empty:
                continue
            plot_series(ax, d["workers"], d[ycol], d["oversubscribed"], f"n={n:,}", COLORS[i % 10])
            if ycol == "time_ms":
                s = ser[(ser["metric"] == m) & (ser["n"] == n)]["time_ms"]
                if len(s):
                    ax.axhline(float(s.iloc[0]), color=COLORS[i % 10], ls="--", lw=1, alpha=0.7)
        if ideal:
            ax.plot(workers, workers, "k:", lw=1, label="ideal (linear)")
        if hline is not None:
            ax.axhline(hline, color="gray", lw=1)
        style_workers_axis(ax, workers, cpus)
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel("OpenMP threads (workers)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{m}")
        ax.legend(fontsize=7)
    fig.suptitle(f"SimdSearch: {ylabel} vs workers" + ("  (dashed = serial baseline)" if ycol == "time_ms" else "") + f"\n{subtitle}", fontsize=9)
    save(fig, out / fn_name)


facet("simdsearch_time_vs_workers.png", "execution time (ms)", "time_ms", ylog=True)
facet("simdsearch_speedup_vs_workers.png", "speedup = T_serial / T_openmp", "speedup", ideal=True, hline=1.0)
facet("simdsearch_efficiency_vs_workers.png", "efficiency = speedup / threads", "efficiency", hline=1.0)


def vs_size(fn_name, ylabel, ycol, speed=False):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        if not speed:
            s = ser[ser["metric"] == m].sort_values("n")
            ax.plot(s["n"], s["time_ms"], "k--", marker="s", label="serial")
        for i, w in enumerate(workers):
            d = omp[(omp["metric"] == m) & (omp["workers"] == w)].sort_values("n")
            if d.empty:
                continue
            plot_series(ax, d["n"], d[ycol], d["oversubscribed"], f"OpenMP T={w}", COLORS[i % 10])
        if speed:
            ax.axhline(1.0, color="gray", lw=1)
        ax.set_xscale("log")
        if not speed:
            ax.set_yscale("log")
        ax.set_xlabel("dataset size n (vectors)")
        ax.set_ylabel(ylabel)
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"SimdSearch: {ylabel} vs dataset size\n{subtitle}", fontsize=9)
    save(fig, out / fn_name)


vs_size("simdsearch_time_vs_size.png", "execution time (ms)", "time_ms")
vs_size("simdsearch_speedup_vs_size.png", "speedup", "speedup", speed=True)

# Combined scaling overview for the largest dataset.
nmax = max(sizes)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (col, lab) in enumerate([("time_ms", "time (ms)"), ("speedup", "speedup"), ("efficiency", "efficiency")]):
    ax = axes[j]
    for i, m in enumerate(metrics):
        d = omp[(omp["metric"] == m) & (omp["n"] == nmax)].sort_values("workers")
        plot_series(ax, d["workers"], d[col], d["oversubscribed"], m, COLORS[i], marker="os"[i % 2])
    if col == "speedup":
        ax.plot(workers, workers, "k:", label="ideal")
    style_workers_axis(ax, workers, cpus)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel(lab)
    ax.legend(fontsize=8)
fig.suptitle(f"SimdSearch thread-count scaling, n={nmax:,}\n{subtitle}", fontsize=9)
save(fig, out / "simdsearch_scaling_overview.png")

# Dimension effect (serial time per vector-component shows memory/compute behaviour).
fig, ax = plt.subplots(figsize=(6.2, 4.4))
big = df[(df["k"] == k) & (df["impl"] == "serial") & (df["metric"] == metrics[0])]
for i, n in enumerate(sorted(big["n"].unique())):
    d = big[big["n"] == n].sort_values("dim")
    ax.plot(d["dim"], d["time_ms"], marker="o", color=COLORS[i % 10], label=f"serial n={n:,}")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
ax.set_xlabel("vector dimension"); ax.set_ylabel("execution time (ms)"); ax.legend(fontsize=7)
ax.set_title(f"SimdSearch serial time vs dimension ({metrics[0]}, K={k})", fontsize=9)
save(fig, out / "simdsearch_time_vs_dim.png")

# ---- summary tables ----
full = df.copy()
full.to_csv(pathlib.Path(args.csv).with_name("simdsearch_summary_full.csv"), index=False)
rows = []
for (m, n), g in sel.groupby(["metric", "n"]):
    r = {"metric": m, "n": int(n), "serial_ms": float(g[g["impl"] == "serial"]["time_ms"].iloc[0])}
    for _, o in g[g["impl"] == "openmp"].sort_values("workers").iterrows():
        star = "*" if o["oversubscribed"] else ""
        r[f"T={int(o['workers'])}{star}"] = f"{o['time_ms']:.3f} ms ({o['speedup']:.2f}x, e={o['efficiency']:.2f})"
    rows.append(r)
tab = pd.DataFrame(rows).sort_values(["metric", "n"])
md = [f"### SimdSearch benchmark summary (dim={dim}, K={k})", "",
      f"Statistic: {stat} of {reps} timed runs (+ warm-up). Cells: time (speedup, parallel efficiency). "
      f"CPUs available during the run: {cpus}. `*` = oversubscribed (threads > CPUs): such points measure "
      f"scheduling overhead, **not** parallel scaling.", "", md_table(tab), ""]
(pathlib.Path(args.csv).with_name("simdsearch_summary.md")).write_text("\n".join(md))
print("wrote", pathlib.Path(args.csv).with_name("simdsearch_summary.md"))
fastest = omp.loc[omp["speedup"].idxmax()]
print(f"best observed speedup in plotted slice: {fastest['speedup']:.3f}x at T={int(fastest['workers'])}, "
      f"n={int(fastest['n'])}, {fastest['metric']}, oversubscribed={bool(fastest['oversubscribed'])}")
