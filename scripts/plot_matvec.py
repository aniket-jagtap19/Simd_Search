#!/usr/bin/env python3
"""Matrix-vector graphs + summary: serial vs OpenMP vs MPI, from recorded CSVs only."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *  # noqa: F401,F403,E402

ap = argparse.ArgumentParser()
ap.add_argument("--omp", default="results/matvec_omp.csv")
ap.add_argument("--mpi", default="results/matvec_mpi.csv")
ap.add_argument("--meta", default="results/matvec_meta.json")
ap.add_argument("--outdir", default="results/figures")
args = ap.parse_args()

omp_df = pd.read_csv(args.omp)
frames = [omp_df]
have_mpi = pathlib.Path(args.mpi).exists() and pathlib.Path(args.mpi).stat().st_size > 0
if have_mpi:
    frames.append(pd.read_csv(args.mpi))
else:
    print("NOTE: MPI results missing -> plots contain serial + OpenMP only (MPI experiment pending)")
df = pd.concat(frames, ignore_index=True)
assert (df["verified"] == 1).all(), "unverified rows present - refusing to plot"
cpus = cpus_from_meta(load_meta(args.meta))
out = pathlib.Path(args.outdir)

ser = omp_df[omp_df["impl"] == "serial"].set_index("rows")["time_ms"]
df["serial_ms"] = df["rows"].map(ser)
df["speedup"] = df["serial_ms"] / df["time_ms"]
df["efficiency"] = df["speedup"] / df["workers"]
df.to_csv(pathlib.Path(args.omp).with_name("matvec_combined.csv"), index=False)

sizes = sorted(df["rows"].unique())
par = df[df["impl"] != "serial"]
workers = sorted(par["workers"].unique())
reps = int(df["repeats"].iloc[0])
STY = {"openmp": ("OpenMP (threads)", "o", "-", COLORS[0]),
       "mpi_full": ("MPI total incl. Scatterv of A (processes)", "s", "-", COLORS[3]),
       "mpi_resident": ("MPI, matrix already distributed (processes)", "^", "--", COLORS[2])}
subtitle = f"median of {reps} runs, {cpus if cpus else '?'} CPU(s) available; hollow marker = oversubscribed; dashed black = serial"


def grid(fn, col, ylabel, ylog=False, ideal=False, hline=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, n in zip(axes.flat, sizes):
        for impl, (lab, mk, ls, c) in STY.items():
            d = par[(par["rows"] == n) & (par["impl"] == impl)].sort_values("workers")
            if len(d):
                plot_series(ax, d["workers"], d[col], d["oversubscribed"], lab, c, mk, ls)
        if col == "time_ms":
            ax.axhline(float(ser[n]), color="k", ls="--", lw=1, label="serial")
        if ideal:
            ax.plot(workers, workers, "k:", lw=1, label="ideal")
        if hline is not None:
            ax.axhline(hline, color="gray", lw=1)
        style_workers_axis(ax, workers, cpus)
        if ylog:
            ax.set_yscale("log")
        ax.set_title(f"{n} x {n}")
        ax.set_xlabel("workers (OpenMP threads / MPI processes)")
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"Matrix-vector multiplication: {ylabel} vs workers\n{subtitle}", fontsize=9)
    save(fig, out / fn)


grid("matvec_time_vs_workers.png", "time_ms", "execution time (ms)", ylog=True)
grid("matvec_speedup_vs_workers.png", "speedup", "speedup vs serial", ideal=True, hline=1.0)
grid("matvec_efficiency_vs_workers.png", "efficiency", "parallel efficiency", hline=1.0)


def vs_size(fn, col, ylabel, speed):
    impls = [i for i in ("openmp", "mpi_full") if (par["impl"] == i).any()]
    fig, axes = plt.subplots(1, len(impls), figsize=(6 * len(impls), 4.5), squeeze=False)
    for ax, impl in zip(axes[0], impls):
        if not speed:
            ax.plot(sizes, [ser[n] for n in sizes], "k--", marker="s", label="serial")
        for i, w in enumerate(sorted(par[par["impl"] == impl]["workers"].unique())):
            d = par[(par["impl"] == impl) & (par["workers"] == w)].sort_values("rows")
            plot_series(ax, d["rows"], d[col], d["oversubscribed"], f"{STY[impl][0].split(' (')[0]} W={w}", COLORS[i % 10])
        if speed:
            ax.axhline(1.0, color="gray", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if not speed:
            ax.set_yscale("log")
        ax.set_xlabel("matrix order N (N x N)")
        ax.set_ylabel(ylabel)
        ax.set_title(STY[impl][0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Matrix-vector: {ylabel} vs problem size\n{subtitle}", fontsize=9)
    save(fig, out / fn)


vs_size("matvec_time_vs_size.png", "time_ms", "execution time (ms)", False)
vs_size("matvec_speedup_vs_size.png", "speedup", "speedup vs serial", True)

if (par["impl"] == "mpi_full").any():  # communication vs computation breakdown
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.2), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        d = par[(par["rows"] == n) & (par["impl"] == "mpi_full")].sort_values("workers")
        bottom = np.zeros(len(d))
        for col, lab, c in [("scatter_ms", "Scatterv(A)", COLORS[3]), ("bcast_ms", "Bcast(x)", COLORS[1]),
                            ("compute_ms", "compute", COLORS[2]), ("gather_ms", "Gatherv(y)", COLORS[4])]:
            ax.bar([str(w) for w in d["workers"]], d[col], bottom=bottom, label=lab, color=c)
            bottom += d[col].values
        ax.set_title(f"{n} x {n}")
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("ms (median of each phase, max over ranks)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("MPI phase breakdown (phase medians; they need not sum to the median total)", fontsize=9)
    save(fig, out / "matvec_mpi_phase_breakdown.png")

# ---- summary table ----
rows = []
for n in sizes:
    r0 = {"N": n, "workers": 1, "serial ms": f"{ser[n]:.4f}"}
    rows.append(r0)
tab = []
for n in sizes:
    for w in workers:
        r = {"N": f"{n}", "workers": w}
        for impl, short in [("openmp", "OpenMP"), ("mpi_full", "MPI total"), ("mpi_resident", "MPI resident")]:
            d = par[(par["rows"] == n) & (par["impl"] == impl) & (par["workers"] == w)]
            if len(d):
                x = d.iloc[0]
                star = "*" if x["oversubscribed"] else ""
                r[short] = f"{x['time_ms']:.3f} ms ({x['speedup']:.2f}x, e={x['efficiency']:.2f}){star}"
            else:
                r[short] = "-"
        r["serial ms"] = f"{ser[n]:.3f}"
        tab.append(r)
tab = pd.DataFrame(tab)[["N", "serial ms", "workers", "OpenMP", "MPI total", "MPI resident"]]
md = ["### Matrix-vector benchmark summary", "",
      f"Median of {reps} timed runs. Cells: time (speedup vs serial, efficiency). Serial time is measured in a separate "
      f"process by `bench_matvec_omp`. CPUs available during the runs: {cpus}. `*` = oversubscribed (workers > CPUs) "
      f"so it is **not** valid scaling evidence. `-` = not run/pending.", "", md_table(tab), ""]
p = pathlib.Path(args.omp).with_name("matvec_summary.md")
p.write_text("\n".join(md))
print("wrote", p)
