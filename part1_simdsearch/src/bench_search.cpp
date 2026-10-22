// bench_search — SimdSearch benchmark grid (serial vs OpenMP) -> CSV + metadata JSON.
#include <omp.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "hpc/args.hpp"
#include "hpc/sysinfo.hpp"
#include "hpc/timer.hpp"
#include "simdsearch/dataset.hpp"
#include "simdsearch/search.hpp"

using namespace simdsearch;

namespace {

volatile std::uint64_t g_sink = 0;  // consumes results so the searches cannot be optimised away

struct Series {
  std::vector<double> ms;
  std::vector<Hit> last;
};

template <class Fn>
Series measure(Fn&& fn, int warmup, int repeats) {
  Series s;
  for (int i = 0; i < warmup; ++i) s.last = fn();
  for (int i = 0; i < repeats; ++i) {
    const double a = hpc::now_ms();
    auto r = fn();
    const double b = hpc::now_ms();
    s.ms.push_back(b - a);
    s.last = std::move(r);
  }
  g_sink += s.last.empty() ? 0 : s.last[0].id;
  return s;
}

void write_row(std::ofstream& csv, std::size_t n, std::size_t dim, std::size_t k, Metric m, const char* impl,
               int workers, const char* worker_type, int repeats, int warmup, const hpc::Stats& st,
               double speedup, double eff, bool oversub, bool verified, double maxdiff, std::uint64_t seed) {
  char buf[640];
  std::snprintf(buf, sizeof buf, "%zu,%zu,%zu,%s,%s,%d,%s,%d,%d,median,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.3e,%llu\n",
                n, dim, k, metric_name(m), impl, workers, worker_type, repeats, warmup, st.median, st.min, st.mean,
                st.stddev, speedup, eff, oversub ? 1 : 0, verified ? 1 : 0, maxdiff, (unsigned long long)seed);
  csv << buf;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    hpc::Args a(argc, argv);
    if (a.flag("help")) {
      std::puts("bench_search --sizes 1000,10000,50000 --dims 32,64,128 --ks 10 --metrics cosine,euclidean\n"
                "             --threads 1,2,4,8 --repeats 7 --warmup 2 --seed 42\n"
                "             --out results/simdsearch_benchmark.csv --meta results/simdsearch_meta.json\n"
                "             [--allow-oversubscribe] [--config configs/simdsearch_quick.cfg]");
      return 0;
    }
    a.require_known({"sizes", "dims", "ks", "metrics", "threads", "repeats", "warmup", "seed", "query-seed", "out",
                     "meta", "allow-oversubscribe", "help"});
    const auto sizes = a.get_int_list("sizes", {1000, 10000, 50000});
    const auto dims = a.get_int_list("dims", {32, 64, 128});
    const auto ks = a.get_int_list("ks", {10});
    const auto threads = a.get_int_list("threads", {1, 2, 4, 8});
    std::vector<Metric> metrics;
    for (const auto& m : a.get_list("metrics", {"cosine", "euclidean"})) metrics.push_back(parse_metric(m));
    const int repeats = static_cast<int>(a.get_int("repeats", 7)), warmup = static_cast<int>(a.get_int("warmup", 2));
    const auto seed = static_cast<std::uint64_t>(a.get_int("seed", 42));
    const auto qseed = static_cast<std::uint64_t>(a.get_int("query-seed", 7919));
    const bool allow_over = a.flag("allow-oversubscribe");
    const std::string out = a.get("out", "results/simdsearch_benchmark.csv");
    const std::string meta = a.get("meta", "results/simdsearch_meta.json");
    if (repeats < 1 || warmup < 0) throw std::invalid_argument("--repeats >= 1 and --warmup >= 0 required");
    for (auto v : sizes) if (v < 1) throw std::invalid_argument("--sizes must be positive");
    for (auto v : dims) if (v < 1) throw std::invalid_argument("--dims must be positive");
    for (auto v : ks) if (v < 1) throw std::invalid_argument("--ks must be positive");
    for (auto v : threads) if (v < 1) throw std::invalid_argument("--threads must be positive");

    const int cpus = hpc::available_cpus();
    std::printf("CPUs available to this process: %d\n", cpus);
    std::vector<int> usable;
    for (auto t : threads) {
      if (t > cpus && !allow_over)
        std::printf("SKIP  threads=%lld > %d available CPU(s) (use --allow-oversubscribe to run it, flagged in the CSV)\n", t, cpus);
      else
        usable.push_back(static_cast<int>(t));
    }
    omp_set_dynamic(0);

    std::ofstream csv(out, std::ios::trunc);
    if (!csv) throw std::runtime_error("cannot write " + out);
    csv << "n,dim,k,metric,impl,workers,worker_type,repeats,warmup,stat,time_ms,min_ms,mean_ms,stddev_ms,"
           "speedup,efficiency,oversubscribed,verified,max_abs_score_diff,seed\n";

    int failures = 0;
    for (auto n : sizes) for (auto dim : dims) {
      Dataset ds = generate_dataset(static_cast<std::size_t>(n), static_cast<std::size_t>(dim), seed);  // untimed
      const std::vector<float> query = generate_query(ds.dim, qseed);
      for (Metric m : metrics) for (auto k : ks) {
        const std::size_t K = static_cast<std::size_t>(k);
        Series ser = measure([&] { return search_serial(ds, query, K, m); }, warmup, repeats);
        const hpc::Stats sst = hpc::summarize(ser.ms);
        write_row(csv, ds.n, ds.dim, std::min(K, ds.n), m, "serial", 1, "none", repeats, warmup, sst, 1.0, 1.0, false, true, 0.0, seed);
        std::printf("n=%-7zu dim=%-4zu k=%-4zu %-9s serial        %9.4f ms\n", ds.n, ds.dim, std::min(K, ds.n), metric_name(m), sst.median);
        for (int T : usable) {
          int used = 0;
          Series par = measure([&] { return search_openmp(ds, query, K, m, T, &used); }, warmup, repeats);
          MatchReport rep = compare_hits(par.last, ser.last, 1e-9);  // correctness BEFORE accepting timings
          if (used != T) { rep.ok = false; rep.why = "runtime gave " + std::to_string(used) + " threads, requested " + std::to_string(T); }
          if (!rep.ok) {
            std::fprintf(stderr, "VERIFICATION FAILED (T=%d): %s -- timings rejected\n", T, rep.why.c_str());
            ++failures;
            continue;
          }
          const hpc::Stats pst = hpc::summarize(par.ms);
          const double sp = sst.median / pst.median;
          write_row(csv, ds.n, ds.dim, std::min(K, ds.n), m, "openmp", T, "threads", repeats, warmup, pst, sp, sp / T, T > cpus, true, rep.max_abs_score_diff, seed);
          std::printf("n=%-7zu dim=%-4zu k=%-4zu %-9s openmp T=%-3d  %9.4f ms  speedup %6.3f  eff %6.3f%s\n", ds.n, ds.dim,
                      std::min(K, ds.n), metric_name(m), T, pst.median, sp, sp / T, T > cpus ? "  [oversubscribed]" : "");
        }
      }
    }
    csv.close();

    hpc::KV kv = hpc::environment_info();
    kv.push_back({"benchmark", "simdsearch serial vs openmp"});
    kv.push_back({"statistic", "median of repeats"});
    kv.push_back({"repeats", std::to_string(repeats)});
    kv.push_back({"warmup_runs", std::to_string(warmup)});
    kv.push_back({"dataset_seed", std::to_string(seed)});
    kv.push_back({"oversubscription_allowed", allow_over ? "yes (rows flagged oversubscribed=1)" : "no"});
    kv.push_back({"simd", "no explicit SIMD; scalar double-precision loops (see docs/REPORT.md)"});
    std::ofstream mf(meta, std::ios::trunc);
    mf << hpc::to_json(kv);
    std::printf("\nwrote %s and %s\n", out.c_str(), meta.c_str());
    if (failures) { std::fprintf(stderr, "%d verification failure(s)\n", failures); return 1; }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
