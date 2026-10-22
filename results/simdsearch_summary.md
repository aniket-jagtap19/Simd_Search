### SimdSearch benchmark summary (dim=128, K=10)

Statistic: median of 7 timed runs (+ warm-up). Cells: time (speedup, parallel efficiency). CPUs available during the run: 20. `*` = oversubscribed (threads > CPUs): such points measure scheduling overhead, **not** parallel scaling.

| metric | n | serial_ms | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| cosine | 1000 | 0.04543 | 0.046 ms (0.98x, e=0.98) | 0.027 ms (1.68x, e=0.84) | 0.018 ms (2.46x, e=0.62) | 0.015 ms (2.94x, e=0.37) |
| cosine | 10000 | 0.3566 | 0.353 ms (1.01x, e=1.01) | 0.202 ms (1.76x, e=0.88) | 0.163 ms (2.19x, e=0.55) | 0.097 ms (3.68x, e=0.46) |
| cosine | 50000 | 2.064 | 2.447 ms (0.84x, e=0.84) | 1.345 ms (1.53x, e=0.77) | 1.044 ms (1.98x, e=0.49) | 1.156 ms (1.79x, e=0.22) |
| cosine | 100000 | 4.639 | 4.612 ms (1.01x, e=1.01) | 2.840 ms (1.63x, e=0.82) | 1.703 ms (2.72x, e=0.68) | 1.329 ms (3.49x, e=0.44) |
| euclidean | 1000 | 0.04724 | 0.051 ms (0.93x, e=0.93) | 0.028 ms (1.67x, e=0.83) | 0.019 ms (2.49x, e=0.62) | 0.016 ms (3.04x, e=0.38) |
| euclidean | 10000 | 0.3676 | 0.377 ms (0.98x, e=0.98) | 0.196 ms (1.87x, e=0.94) | 0.146 ms (2.51x, e=0.63) | 0.501 ms (0.73x, e=0.09) |
| euclidean | 50000 | 2.257 | 2.153 ms (1.05x, e=1.05) | 1.157 ms (1.95x, e=0.97) | 0.962 ms (2.35x, e=0.59) | 0.821 ms (2.75x, e=0.34) |
| euclidean | 100000 | 4.837 | 4.807 ms (1.01x, e=1.01) | 3.079 ms (1.57x, e=0.79) | 1.764 ms (2.74x, e=0.69) | 1.033 ms (4.68x, e=0.59) |
