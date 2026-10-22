#!/usr/bin/env python3
"""Independent cross-check: recompute Top-K with NumPy and compare with the C++ binary's output.
Usage: python3 scripts/crosscheck_numpy.py [--bin build/part1_simdsearch/simdsearch] [--data data/sample_dataset.bin]
"""
import argparse, json, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="build/part1_simdsearch/simdsearch")
ap.add_argument("--data", default="data/sample_dataset.bin")
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()

raw = open(a.data, "rb").read()
assert raw[:8] == b"SIMDSRC1", "not a SIMDSRC1 file"
n, dim = np.frombuffer(raw[8:24], dtype="<u8")
X = np.frombuffer(raw[24:], dtype="<f4").reshape(int(n), int(dim)).astype(np.float64)
ok = True
for qi in (0, 17, int(n) - 1):
    q = X[qi]
    for metric in ("cosine", "euclidean"):
        out = subprocess.run([a.bin, "search", "--data", a.data, "--query-index", str(qi), "--k", str(a.k),
                              "--metric", metric, "--threads", "1,3", "--repeats", "1", "--warmup", "0", "--json"],
                             capture_output=True, text=True, check=True)
        res = json.loads(out.stdout)["results"]
        if metric == "cosine":
            nrm = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
            s = np.divide(X @ q, nrm, out=np.zeros(int(n)), where=nrm > 0)
            order = np.lexsort((np.arange(int(n)), -s))[: a.k]
        else:
            s = np.linalg.norm(X - q, axis=1)
            order = np.lexsort((np.arange(int(n)), s))[: a.k]
        ids_ok = [r["id"] for r in res] == order.tolist()
        err = max(abs(r["score"] - s[o]) for r, o in zip(res, order))
        good = ids_ok and err < 1e-9
        ok &= good
        print(f"query={qi:<4} {metric:<9} ids match: {ids_ok}  max |score diff|: {err:.2e}  {'PASS' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
