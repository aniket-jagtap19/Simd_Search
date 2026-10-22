// matvec_mpi — MPI matrix-vector benchmark. Run with mpirun/mpiexec -np P.
// Appends rows (impl = mpi_full, mpi_resident) to a CSV file.
#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hpc/args.hpp"
#include "hpc/sysinfo.hpp"
#include "hpc/timer.hpp"
#include "matvec/matvec.hpp"
#include "matvec/matvec_mpi.hpp"
#include "matvec/results.hpp"

namespace {
double median_of(std::vector<double> v) { return hpc::summarize(std::move(v)).median; }
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rc = 0;
  {
    // ---- 1. parse + validate arguments (identical on all ranks; agree on the verdict) ----
    std::vector<long long> sizes;
    int repeats = 30, warmup = 5;
    std::uint64_t seed = 42;
    bool allow_over = false;
    std::string out = "results/matvec_mpi.csv", meta;
    int status = 0;
    std::string err;
    try {
      hpc::Args a(argc, argv);
      a.require_known({"sizes", "repeats", "warmup", "seed", "out", "meta", "allow-oversubscribe", "expect-procs", "help"});
      if (a.flag("help")) {
        if (rank == 0)
          std::puts("mpiexec -np P matvec_mpi --sizes 256,512,1024,2048 --repeats 30 --warmup 5 --seed 42\n"
                    "                         --out results/matvec_mpi.csv [--meta file.json] [--allow-oversubscribe] [--expect-procs P]");
        MPI_Finalize();
        return 0;
      }
      if (a.has("expect-procs") && a.get_int("expect-procs", 0) != size)
        throw std::invalid_argument("expected " + std::to_string(a.get_int("expect-procs", 0)) + " MPI processes but MPI_COMM_WORLD has " +
                                    std::to_string(size) + " (launcher/library mismatch? see README troubleshooting)");
      sizes = a.get_int_list("sizes", {256, 512, 1024, 2048});
      repeats = static_cast<int>(a.get_int("repeats", 30));
      warmup = static_cast<int>(a.get_int("warmup", 5));
      seed = static_cast<std::uint64_t>(a.get_int("seed", 42));
      allow_over = a.flag("allow-oversubscribe");
      out = a.get("out", out);
      meta = a.get("meta", "");
      if (repeats < 1 || warmup < 0) throw std::invalid_argument("--repeats >= 1 and --warmup >= 0 required");
      for (auto v : sizes) if (v < 1) throw std::invalid_argument("--sizes must be positive");
    } catch (const std::exception& e) {
      status = 2;
      err = e.what();
    }
    int worst = 0;
    MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (worst) {
      if (rank == 0) std::fprintf(stderr, "error: %s\n", err.empty() ? "invalid arguments" : err.c_str());
      MPI_Finalize();
      return 2;
    }

    // ---- 2. oversubscription check (per node: ranks sharing a node vs CPUs of that node) ----
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int node_ranks = 1, cpus = hpc::available_cpus(), node_cpus = cpus;
    MPI_Comm_size(node, &node_ranks);
    MPI_Allreduce(&cpus, &node_cpus, 1, MPI_INT, MPI_MIN, node);
    int over = node_ranks > node_cpus ? 1 : 0, any_over = 0;
    MPI_Allreduce(&over, &any_over, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Comm_free(&node);
    if (rank == 0) std::printf("MPI processes: %d   (CPUs per node visible: %d)\n", size, node_cpus);
    if (any_over && !allow_over) {
      if (rank == 0)
        std::fprintf(stderr, "refusing to run: %d MPI processes on a node with only %d CPU(s) would be oversubscribed.\n"
                             "Use fewer processes, or pass --allow-oversubscribe (rows are then flagged in the CSV).\n",
                     node_ranks, node_cpus);
      MPI_Finalize();
      return 3;
    }

    // ---- 3. benchmark ----
    try {
      for (auto N : sizes) {
        const std::size_t n = static_cast<std::size_t>(N);
        matvec::Problem p;
        std::vector<double> yref, y;
        if (rank == 0) {  // generation and the serial reference are untimed and root-only
          p = matvec::make_problem(n, n, seed);
          yref.resize(n);
          y.resize(n);
          matvec::multiply_serial(p.A.data(), p.x.data(), yref.data(), n, n);
        }
        matvec::MpiMatVec mm(MPI_COMM_WORLD, n, n);
        const double* A = rank == 0 ? p.A.data() : nullptr;
        const double* x = rank == 0 ? p.x.data() : nullptr;
        double* yp = rank == 0 ? y.data() : nullptr;

        for (int mode = 0; mode < 2; ++mode) {  // 0 = full (scatter every run), 1 = resident matrix
          const bool full = (mode == 0);
          if (!full) mm.scatter_matrix(A);       // distribute once; excluded from timing
          // correctness first
          mm.run(A, x, yp, full);
          int ok = 1;
          double err_rel = 0;
          if (rank == 0) {
            err_rel = matvec::max_relative_error(y.data(), yref.data(), n);
            ok = err_rel <= matvec::kTolerance;
          }
          MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
          if (!ok) {
            if (rank == 0) std::fprintf(stderr, "VERIFICATION FAILED N=%zu P=%d mode=%s (err %.3e)\n", n, size, full ? "full" : "resident", err_rel);
            MPI_Finalize();
            return 1;
          }
          for (int w = 1; w < warmup; ++w) mm.run(A, x, yp, full);
          std::vector<double> tot, sc, bc, cp, ga;
          for (int r = 0; r < repeats; ++r) {
            matvec::PhaseTimes t = mm.run(A, x, yp, full);
            tot.push_back(t.total_ms); sc.push_back(t.scatter_ms); bc.push_back(t.bcast_ms);
            cp.push_back(t.compute_ms); ga.push_back(t.gather_ms);
          }
          if (rank == 0) {
            matvec::Row row;
            row.rows = row.cols = n;
            row.impl = full ? "mpi_full" : "mpi_resident";
            row.workers = size; row.worker_type = "processes";
            row.repeats = repeats; row.warmup = warmup; row.stats = hpc::summarize(tot);
            row.scatter_ms = median_of(sc); row.bcast_ms = median_of(bc); row.compute_ms = median_of(cp); row.gather_ms = median_of(ga);
            row.oversubscribed = any_over != 0; row.verified = true; row.max_rel_err = err_rel; row.seed = seed;
            matvec::append_row(out, row);
            std::printf("N=%-5zu P=%-2d %-12s total %9.4f ms (scatter %8.4f bcast %7.4f compute %8.4f gather %7.4f)%s\n", n,
                        size, row.impl.c_str(), row.stats.median, row.scatter_ms, row.bcast_ms, row.compute_ms,
                        row.gather_ms, any_over ? "  [oversubscribed]" : "");
          }
        }
      }
    } catch (const std::exception& e) {  // thrown identically on all ranks (argument-only validation)
      if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
      rc = 2;
    }

    if (rc == 0 && rank == 0 && !meta.empty()) {
      char lib[MPI_MAX_LIBRARY_VERSION_STRING];
      int len = 0;
      MPI_Get_library_version(lib, &len);
      hpc::KV kv = hpc::environment_info();
      kv.push_back({"benchmark", "matvec MPI (row-block Scatterv/Bcast/Gatherv)"});
      kv.push_back({"mpi_library", std::string(lib, static_cast<std::size_t>(len))});
      kv.push_back({"mpi_clock", "MPI_Wtime"});
      kv.push_back({"statistic", "median of repeats"});
      kv.push_back({"repeats", std::to_string(repeats)});
      kv.push_back({"warmup_runs", std::to_string(warmup)});
      kv.push_back({"seed", std::to_string(seed)});
      std::ofstream(meta, std::ios::trunc) << hpc::to_json(kv);
    }
  }
  MPI_Finalize();
  return rc;
}
