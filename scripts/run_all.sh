#!/usr/bin/env bash
# Build, test, benchmark and plot everything.   Usage:  scripts/run_all.sh
#
# Environment overrides:
#   BUILD_DIR=build            CMake build directory
#   THREADS="1 2 4 8"          OpenMP thread counts
#   PROCS="1 2 4 8"            MPI process counts
#   ALLOW_OVERSUBSCRIBE=0|1    1 = also run worker counts above the CPU count (rows are flagged
#                              oversubscribed=1 in the CSV and drawn hollow in the plots)
#   SEARCH_CONFIG=configs/simdsearch_full.cfg
#   MPIEXEC=mpiexec            launcher;  MPIEXEC_FLAGS="--oversubscribe" for Open MPI if needed
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build}
THREADS=${THREADS:-"1 2 4 8"}
PROCS=${PROCS:-"1 2 4 8"}
ALLOW_OVERSUBSCRIBE=${ALLOW_OVERSUBSCRIBE:-0}
SEARCH_CONFIG=${SEARCH_CONFIG:-configs/simdsearch_full.cfg}
MPIEXEC=${MPIEXEC:-mpiexec}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
CPUS=$(nproc)
OVER_FLAG=""; [[ "$ALLOW_OVERSUBSCRIBE" == "1" ]] && OVER_FLAG="--allow-oversubscribe"
csv_list() { echo "$*" | tr ' ' ','; }

echo "== CPUs available: $CPUS | threads: $THREADS | procs: $PROCS | oversubscribe: $ALLOW_OVERSUBSCRIBE"
mkdir -p results/figures

echo "== [1/6] configure + build"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j"$CPUS" >/dev/null
HAVE_MPI=0; [[ -x "$BUILD_DIR/part2_matvec/matvec_mpi" ]] && HAVE_MPI=1

echo "== [2/6] correctness tests"
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "== [3/6] SimdSearch benchmark (serial vs OpenMP)"
"$BUILD_DIR/part1_simdsearch/bench_search" --config "$SEARCH_CONFIG" --threads "$(csv_list $THREADS)" $OVER_FLAG \
    --out results/simdsearch_benchmark.csv --meta results/simdsearch_meta.json

echo "== [4/6] matrix-vector: serial + OpenMP"
"$BUILD_DIR/part2_matvec/bench_matvec_omp" --config configs/matvec_assignment.cfg --threads "$(csv_list $THREADS)" $OVER_FLAG \
    --out results/matvec_omp.csv --meta results/matvec_meta.json

echo "== [5/6] matrix-vector: MPI"
if [[ "$HAVE_MPI" == "1" ]]; then
  rm -f results/matvec_mpi.csv
  for P in $PROCS; do
    if (( P > CPUS )) && [[ "$ALLOW_OVERSUBSCRIBE" != "1" ]]; then
      echo "SKIP  P=$P > $CPUS CPU(s) (set ALLOW_OVERSUBSCRIBE=1 to run it, flagged)"; continue
    fi
    $MPIEXEC -n "$P" $MPIEXEC_FLAGS "$BUILD_DIR/part2_matvec/matvec_mpi" --config configs/matvec_assignment.cfg --expect-procs "$P" \
        $OVER_FLAG --out results/matvec_mpi.csv --meta results/matvec_mpi_meta.json
  done
else
  echo "MPI executable not built (no MPI found): MPI experiment PENDING. See README 'Running without MPI'."
fi

echo "== [6/6] graphs + summary tables"
python3 scripts/plot_simdsearch.py
python3 scripts/plot_matvec.py
echo "== done. CSV/JSON/MD in results/, figures in results/figures/"
