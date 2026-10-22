// bench_matvec_omp — serial and OpenMP matrix-vector multiplication benchmark.
#include <omp.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hpc/args.hpp"
#include "hpc/sysinfo.hpp"
#include "hpc/timer.hpp"
#include "matvec/matvec.hpp"
#include "matvec/results.hpp"

namespace {
volatile double g_sink = 0;

template <class Fn>
std::vector<double> measure(Fn&& fn, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> ms;
  for (int i = 0; i < repeats; ++i) {
    const double a = hpc::now_ms();
    fn();
    const double b = hpc::now_ms();
    ms.push_back(b - a);
  }
  return ms;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    hpc::Args a(argc, argv);
    if (a.flag("help")) {
      std::puts("bench_matvec_omp --sizes 256,512,1024,2048 --threads 1,2,4,8 --repeats 30 --warmup 5 --seed 42\n"
                "                 --out results/matvec_omp.csv --meta results/matvec_meta.json [--allow-oversubscribe]");
      return 0;
    }
    a.require_known({"sizes", "threads", "repeats", "warmup", "seed", "out", "meta", "allow-oversubscribe", "help"});
    const auto sizes = a.get_int_list("sizes", {256, 512, 1024, 2048});
    const auto threads = a.get_int_list("threads", {1, 2, 4, 8});
    const int repeats = static_cast<int>(a.get_int("repeats", 30)), warmup = static_cast<int>(a.get_int("warmup", 5));
    const auto seed = static_cast<std::uint64_t>(a.get_int("seed", 42));
    const bool allow_over = a.flag("allow-oversubscribe");
    const std::string out = a.get("out", "results/matvec_omp.csv");
    const std::string meta = a.get("meta", "results/matvec_meta.json");
    if (repeats < 1 || warmup < 0) throw std::invalid_argument("--repeats >= 1 and --warmup >= 0 required");
    for (auto v : sizes) if (v < 1) throw std::invalid_argument("--sizes must be positive");
    for (auto v : threads) if (v < 1) throw std::invalid_argument("--threads must be positive");

    const int cpus = hpc::available_cpus();
    std::printf("CPUs available to this process: %d\n", cpus);
    std::vector<int> usable;
    for (auto t : threads) {
      if (t > cpus && !allow_over)
        std::printf("SKIP  threads=%lld > %d available CPU(s) (use --allow-oversubscribe; rows will be flagged)\n", t, cpus);
      else usable.push_back(static_cast<int>(t));
    }
    omp_set_dynamic(0);
    std::remove(out.c_str());

    int failures = 0;
    for (auto N : sizes) {
      const std::size_t n = static_cast<std::size_t>(N);
      matvec::Problem p = matvec::make_problem(n, n, seed);  // untimed
      // Independent oracle: long-double accumulation in a different code path.
      std::vector<double> oracle(n);
      for (std::size_t i = 0; i < n; ++i) {
        long double s = 0;
        for (std::size_t j = 0; j < n; ++j) s += static_cast<long double>(p.A[i * n + j]) * p.x[j];
        oracle[i] = static_cast<double>(s);
      }
      std::vector<double> ys(n), yp(n);

      auto ms = measure([&] { matvec::multiply_serial(p.A.data(), p.x.data(), ys.data(), n, n); g_sink += ys[0]; }, warmup, repeats);
      matvec::Row row;
      row.rows = row.cols = n; row.impl = "serial"; row.workers = 1; row.worker_type = "none";
      row.repeats = repeats; row.warmup = warmup; row.stats = hpc::summarize(ms); row.compute_ms = row.stats.median;
      row.max_rel_err = matvec::max_relative_error(ys.data(), oracle.data(), n);
      row.verified = row.max_rel_err <= matvec::kTolerance; row.seed = seed;
      if (!row.verified) { std::fprintf(stderr, "serial disagrees with oracle (err %.3e)\n", row.max_rel_err); ++failures; continue; }
      matvec::append_row(out, row);
      const double ser_med = row.stats.median;
      std::printf("N=%-5zu serial        %9.4f ms\n", n, ser_med);

      for (int T : usable) {
        int used = 0;
        auto pm = measure([&] { matvec::multiply_openmp(p.A.data(), p.x.data(), yp.data(), n, n, T, &used); g_sink += yp[0]; }, warmup, repeats);
        const double err = matvec::max_relative_error(yp.data(), ys.data(), n);
        if (err > matvec::kTolerance || used != T) {
          std::fprintf(stderr, "VERIFICATION FAILED N=%zu T=%d (err %.3e, team %d) -- timings rejected\n", n, T, err, used);
          ++failures;
          continue;
        }
        matvec::Row r = row;
        r.impl = "openmp"; r.workers = T; r.worker_type = "threads"; r.stats = hpc::summarize(pm);
        r.compute_ms = r.stats.median; r.oversubscribed = T > cpus; r.verified = true; r.max_rel_err = err;
        matvec::append_row(out, r);
        const double sp = ser_med / r.stats.median;
        std::printf("N=%-5zu openmp T=%-3d %9.4f ms  speedup %6.3f  eff %6.3f%s\n", n, T, r.stats.median, sp, sp / T,
                    T > cpus ? "  [oversubscribed]" : "");
      }
    }
    hpc::KV kv = hpc::environment_info();
    kv.push_back({"benchmark", "matvec serial vs openmp"});
    kv.push_back({"statistic", "median of repeats"});
    kv.push_back({"repeats", std::to_string(repeats)});
    kv.push_back({"warmup_runs", std::to_string(warmup)});
    kv.push_back({"seed", std::to_string(seed)});
    std::ofstream(meta, std::ios::trunc) << hpc::to_json(kv);
    std::printf("wrote %s and %s\n", out.c_str(), meta.c_str());
    return failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
