"""SimdSearch dashboard. All searching and timing is done by the C++ `simdsearch` binary
(invoked with --json); this file only builds the command line and visualises the output.

Run:  streamlit run dashboard/app.py
"""
import json
import os
import pathlib
import subprocess

import altair as alt
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("SIMDSEARCH_BIN", ROOT / "build" / "part1_simdsearch" / "simdsearch"))
BENCH_CSV = ROOT / "results" / "simdsearch_benchmark.csv"

st.set_page_config(page_title="SimdSearch", page_icon=":mag:", layout="wide")
st.title("SimdSearch: parallel exact Top-K similarity search")
st.caption("Serial baseline vs OpenMP. Every number below comes from the C++ implementation; nothing is simulated.")

if not BIN.exists():
    st.error(f"C++ binary not found at `{BIN}`. Build it first:\n\n"
             "`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`")
    st.stop()


@st.cache_data(show_spinner=False)
def hw_info(binary: str):
    out = subprocess.run([binary, "info"], capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


info = hw_info(str(BIN))
cpus = int(info.get("cpus_available_to_process", 1))

with st.sidebar:
    st.header("Configuration")
    source = st.radio("Dataset", ["Synthetic (reproducible)", "Load from file"])
    data_path = ""
    if source.startswith("Synthetic"):
        n = st.number_input("Vectors (n)", 100, 2_000_000, 50_000, step=1000)
        dim = st.number_input("Dimensions", 2, 4096, 128, step=2)
        seed = st.number_input("Dataset seed", 0, 2**31 - 1, 42)
    else:
        data_path = st.text_input("Dataset path (.bin / .csv)", str(ROOT / "data" / "sample_dataset.bin"))
        n = dim = seed = None
    metric = st.selectbox("Similarity metric", ["cosine", "euclidean"])
    k = st.number_input("K (neighbours)", 1, 10_000, 10)
    opts = [1, 2, 4, 8, 16]
    threads = st.multiselect("OpenMP thread counts", opts, default=[t for t in opts if t <= cpus] or [1])
    if any(t > cpus for t in threads):
        st.warning(f"Only {cpus} CPU(s) available: thread counts above that are oversubscribed and "
                   "show overhead, not scaling.")
    qmode = st.radio("Query", ["Random (seeded)", "Existing vector (by index)"])
    qseed = st.number_input("Query seed", 0, 2**31 - 1, 7919) if qmode.startswith("Random") else None
    qidx = st.number_input("Query index", 0, 10**9, 0) if qmode.startswith("Existing") else None
    repeats = st.slider("Timed repetitions", 1, 20, 5)
    run = st.button("Run search", type="primary", width="stretch")


def build_cmd():
    cmd = [str(BIN), "search", "--json", "--metric", metric, "--k", str(int(k)), "--repeats", str(repeats),
           "--warmup", "1", "--threads", ",".join(str(t) for t in (threads or [1]))]
    if data_path:
        cmd += ["--data", data_path]
    else:
        cmd += ["--n", str(int(n)), "--dim", str(int(dim)), "--seed", str(int(seed))]
    if qseed is not None:
        cmd += ["--query-seed", str(int(qseed))]
    if qidx is not None:
        cmd += ["--query-index", str(int(qidx))]
    return cmd


if run:
    with st.spinner("Running C++ search..."):
        proc = subprocess.run(build_cmd(), capture_output=True, text=True, timeout=600)
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        st.session_state.pop("result", None)
        st.error(proc.stderr.strip() or "search failed")
    else:
        st.session_state["result"] = json.loads(proc.stdout)
        st.session_state["cmd"] = " ".join(build_cmd())

tab_res, tab_perf, tab_bench, tab_about = st.tabs(["Ranked results", "Serial vs OpenMP", "Benchmark CSV", "About"])

with tab_res:
    res = st.session_state.get("result")
    if not res:
        st.info("Choose a configuration in the sidebar and press **Run search**.")
    else:
        cfg = res["config"]
        c = st.columns(4)
        c[0].metric("Dataset", f"{cfg['n']:,} x {cfg['dim']}")
        c[1].metric("Metric", cfg["metric"])
        c[2].metric("K", cfg["k"])
        c[3].metric("Verified (serial == OpenMP)", "yes" if res["all_verified"] else "NO")
        order = "highest similarity first" if cfg["metric"] == "cosine" else "smallest distance first"
        st.subheader(f"Top-{cfg['k']} ({order})")
        st.dataframe(pd.DataFrame(res["results"]).rename(columns={"score": cfg["score_label"]}),
                     hide_index=True, width="stretch")
        with st.expander("Exact command executed"):
            st.code(st.session_state["cmd"], language="bash")

with tab_perf:
    res = st.session_state.get("result")
    if not res or not res.get("openmp"):
        st.info("Run a search to see timings.")
    else:
        ser = res["serial"]["median_ms"]
        rows = [{"threads": r["threads"], "time_ms": r["median_ms"], "speedup": r["speedup"],
                 "efficiency": r["efficiency"], "oversubscribed": r["oversubscribed"]} for r in res["openmp"]]
        df = pd.DataFrame(rows)
        m = st.columns(3)
        m[0].metric("Serial (median)", f"{ser:.3f} ms")
        best = df.loc[df["time_ms"].idxmin()]
        m[1].metric("Fastest OpenMP", f"{best['time_ms']:.3f} ms", f"T={int(best['threads'])}")
        m[2].metric("Speedup of fastest", f"{best['speedup']:.2f}x")
        if df["oversubscribed"].any():
            st.warning("Rows marked oversubscribed ran more threads than CPUs.")
        base = alt.Chart(df).encode(x=alt.X("threads:O", title="OpenMP threads"))
        t_chart = (base.mark_line(point=True).encode(y=alt.Y("time_ms:Q", title="time (ms)"),
                   tooltip=list(df.columns)) + alt.Chart(pd.DataFrame({"y": [ser]})).mark_rule(
                   strokeDash=[5, 3], color="gray").encode(y="y:Q")).properties(title="Execution time (dashed = serial)", height=260)
        ideal = pd.DataFrame({"threads": df["threads"].astype(str), "ideal": df["threads"]})
        s_chart = (base.mark_line(point=True).encode(y=alt.Y("speedup:Q", title="speedup"), tooltip=list(df.columns)) +
                   alt.Chart(ideal).mark_line(strokeDash=[4, 4], color="gray").encode(x="threads:O", y="ideal:Q")
                   ).properties(title="Speedup (dashed = ideal)", height=260)
        e_chart = base.mark_line(point=True).encode(y=alt.Y("efficiency:Q", title="efficiency", scale=alt.Scale(domain=[0, 1.2])),
                                                    tooltip=list(df.columns)).properties(title="Parallel efficiency", height=260)
        a, b, c3 = st.columns(3)
        a.altair_chart(t_chart, width="stretch")
        b.altair_chart(s_chart, width="stretch")
        c3.altair_chart(e_chart, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"Median of {res['config']['repeats']} timed runs after {res['config']['warmup']} warm-up run. "
                   "Serial and OpenMP timings include Top-K selection and the final merge; dataset generation is excluded.")

with tab_bench:
    if not BENCH_CSV.exists():
        st.info("No benchmark CSV yet. Run `scripts/run_all.sh` (or `bench_search`) to create results/simdsearch_benchmark.csv.")
    else:
        b = pd.read_csv(BENCH_CSV)
        f = st.columns(4)
        metric_b = f[0].selectbox("Metric ", sorted(b["metric"].unique()))
        dim_b = f[1].selectbox("Dimension", sorted(b["dim"].unique()), index=len(b["dim"].unique()) - 1)
        k_b = f[2].selectbox("K ", sorted(b["k"].unique()))
        y_b = f[3].selectbox("Y axis", ["time_ms", "speedup", "efficiency"])
        sel = b[(b.metric == metric_b) & (b.dim == dim_b) & (b.k == k_b)].copy()
        sel["n"] = sel["n"].astype(str)
        o = sel[sel.impl == "openmp"]
        ch = alt.Chart(o).mark_line(point=True).encode(
            x=alt.X("workers:O", title="OpenMP threads"), y=alt.Y(f"{y_b}:Q", scale=alt.Scale(type="log") if y_b == "time_ms" else alt.Undefined),
            color="n:N", shape="oversubscribed:N", tooltip=["n", "workers", "time_ms", "speedup", "efficiency", "oversubscribed"])
        st.altair_chart(ch.properties(height=360), width="stretch")
        st.caption("Shape distinguishes oversubscribed (threads > CPUs) points. Loaded from the recorded CSV.")
        st.dataframe(sel, hide_index=True, width="stretch")

with tab_about:
    st.markdown(f"""
**Environment:** {info.get('cpu_model','?')} | {cpus} CPU(s) available | {info.get('compiler','?')} | build `{info.get('build_type','?')}`

* Exact Top-K; cosine is ranked descending, Euclidean ascending; ties broken by smaller vector ID.
* OpenMP: static partition of vectors, thread-private bounded heaps, one merge after the parallel region.
* No explicit SIMD is implemented, so no SIMD speedup is claimed.
""")
