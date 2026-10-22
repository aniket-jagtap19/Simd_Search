// simdsearch — command-line front end (search | generate | info).
#include <omp.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hpc/args.hpp"
#include "hpc/sysinfo.hpp"
#include "hpc/timer.hpp"
#include "simdsearch/dataset.hpp"
#include "simdsearch/io.hpp"
#include "simdsearch/search.hpp"

using namespace simdsearch;

namespace {

void usage() {
  std::puts(
      "SimdSearch - exact Top-K similarity search (serial baseline vs OpenMP)\n"
      "\n"
      "  simdsearch search   [options]   run a search and compare serial vs OpenMP\n"
      "  simdsearch generate [options]   write a reproducible synthetic dataset to a file\n"
      "  simdsearch info                 print hardware/software environment\n"
      "\n"
      "search options:\n"
      "  --data FILE            load dataset (.csv or SIMDSRC1 binary); default: synthetic\n"
      "  --n N --dim D          synthetic dataset size / dimension   (10000 / 64)\n"
      "  --seed S               dataset seed                          (42)\n"
      "  --query-seed S         seed of the random query              (7919)\n"
      "  --query-index I        use dataset vector I as the query\n"
      "  --query-file FILE      use first vector of FILE as the query\n"
      "  --k K                  number of neighbours                  (10)\n"
      "  --metric cosine|euclidean                                    (cosine)\n"
      "  --threads T1,T2,...    OpenMP thread counts                  (CPUs available)\n"
      "  --mode serial|omp|both                                       (both)\n"
      "  --repeats R --warmup W timing repetitions                    (5 / 1)\n"
      "  --show M               rows of results to print              (K)\n"
      "  --json                 machine-readable output (used by the dashboard)\n"
      "  --config FILE          read options from a key = value file\n"
      "\n"
      "generate options:  --n N --dim D --seed S --out FILE   (.csv => text, else binary)\n");
}

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

struct Timed {
  std::vector<double> ms;
  std::vector<Hit> last;
};

template <class Fn>
Timed time_runs(Fn&& fn, int warmup, int repeats) {
  Timed t;
  for (int i = 0; i < warmup; ++i) t.last = fn();
  for (int i = 0; i < repeats; ++i) {
    const double a = hpc::now_ms();
    auto r = fn();
    const double b = hpc::now_ms();
    t.ms.push_back(b - a);
    t.last = std::move(r);
  }
  return t;
}

struct OmpRun {
  int threads = 0, used = 0;
  bool oversubscribed = false, verified = true;
  double max_diff = 0, median = 0, speedup = 0, efficiency = 0;
  std::string why;
  std::vector<double> ms;
};

int cmd_generate(const hpc::Args& a) {
  a.require_known({"n", "dim", "seed", "out"});
  if (!a.has("out")) throw std::invalid_argument("generate requires --out FILE");
  const auto n = a.get_int("n", 1000), dim = a.get_int("dim", 32), seed = a.get_int("seed", 42);
  if (n <= 0 || dim <= 0) throw std::invalid_argument("--n and --dim must be positive");
  Dataset ds = generate_dataset(static_cast<std::size_t>(n), static_cast<std::size_t>(dim),
                                static_cast<std::uint64_t>(seed));
  const std::string out = a.get("out", "");
  if (out.size() >= 4 && out.compare(out.size() - 4, 4, ".csv") == 0) save_csv(ds, out);
  else save_binary(ds, out);
  std::printf("wrote %lld vectors x %lld dims (seed %lld) to %s\n", n, dim, seed, out.c_str());
  return 0;
}

int cmd_info() {
  std::printf("%s", hpc::to_json(hpc::environment_info()).c_str());
  return 0;
}

int cmd_search(const hpc::Args& a) {
  a.require_known({"data", "n", "dim", "seed", "query-seed", "query-file", "query-index", "k", "metric",
                   "threads", "mode", "repeats", "warmup", "json", "show"});
  const bool json = a.flag("json");
  const Metric metric = parse_metric(a.get("metric", "cosine"));
  const long long k_ll = a.get_int("k", 10);
  if (k_ll < 1) throw std::invalid_argument("--k must be >= 1");
  const std::size_t k = static_cast<std::size_t>(k_ll);
  const std::string mode = a.get("mode", "both");
  if (mode != "serial" && mode != "omp" && mode != "both")
    throw std::invalid_argument("--mode must be serial, omp or both");
  const int repeats = static_cast<int>(a.get_int("repeats", 5)), warmup = static_cast<int>(a.get_int("warmup", 1));
  if (repeats < 1 || warmup < 0) throw std::invalid_argument("--repeats must be >= 1 and --warmup >= 0");
  const int cpus = hpc::available_cpus();
  std::vector<long long> tl = a.get_int_list("threads", {cpus});
  for (auto t : tl)
    if (t < 1 || t > 4096) throw std::invalid_argument("--threads values must be in [1,4096]");

  // ---- dataset (never timed) ----
  Dataset ds;
  std::string source = "synthetic";
  const auto seed = static_cast<std::uint64_t>(a.get_int("seed", 42));
  if (a.has("data")) {
    source = a.get("data", "");
    ds = load_dataset(source);
  } else {
    const auto n = a.get_int("n", 10000), dim = a.get_int("dim", 64);
    if (n <= 0 || dim <= 0) throw std::invalid_argument("--n and --dim must be positive");
    ds = generate_dataset(static_cast<std::size_t>(n), static_cast<std::size_t>(dim), seed);
  }
  // ---- query ----
  std::vector<float> query;
  std::string query_source = "random";
  if (a.has("query-file")) {
    Dataset qd = load_dataset(a.get("query-file", ""));
    if (qd.dim != ds.dim) throw std::invalid_argument("query file dimension != dataset dimension");
    query.assign(qd.row(0), qd.row(0) + qd.dim);
    query_source = "file:" + a.get("query-file", "");
  } else if (a.has("query-index")) {
    const auto qi = a.get_int("query-index", 0);
    if (qi < 0 || static_cast<std::size_t>(qi) >= ds.n) throw std::invalid_argument("--query-index out of range");
    query.assign(ds.row(static_cast<std::size_t>(qi)), ds.row(static_cast<std::size_t>(qi)) + ds.dim);
    query_source = "dataset_index:" + std::to_string(qi);
  } else {
    query = generate_query(ds.dim, static_cast<std::uint64_t>(a.get_int("query-seed", 7919)));
  }
  const std::size_t k_eff = std::min(k, ds.n);

  // ---- serial ----
  std::vector<Hit> reference;
  Timed ser;
  double ser_med = 0;
  if (mode != "omp") {
    ser = time_runs([&] { return search_serial(ds, query, k, metric); }, warmup, repeats);
    ser_med = hpc::summarize(ser.ms).median;
    reference = ser.last;
  }
  // ---- OpenMP ----
  omp_set_dynamic(0);
  std::vector<OmpRun> runs;
  std::vector<Hit> shown = reference;
  if (mode != "serial") {
    for (auto t : tl) {
      OmpRun r;
      r.threads = static_cast<int>(t);
      r.oversubscribed = r.threads > cpus;
      if (r.oversubscribed)
        std::fprintf(stderr, "warning: %d threads requested but only %d CPU(s) available: oversubscribed run\n",
                     r.threads, cpus);
      Timed o = time_runs(
          [&] { return search_openmp(ds, query, k, metric, r.threads, &r.used); }, warmup, repeats);
      r.ms = o.ms;
      r.median = hpc::summarize(o.ms).median;
      if (mode == "both") {
        MatchReport rep = compare_hits(o.last, reference, 1e-9);
        r.verified = rep.ok;
        r.max_diff = rep.max_abs_score_diff;
        r.why = rep.why;
        r.speedup = ser_med / r.median;
        r.efficiency = r.speedup / r.threads;
      }
      if (shown.empty()) shown = o.last;
      runs.push_back(std::move(r));
    }
  }
  bool all_ok = true;
  for (auto& r : runs) all_ok = all_ok && r.verified;

  const std::size_t show = static_cast<std::size_t>(std::min<long long>(a.get_int("show", static_cast<long long>(k_eff)),
                                                                        static_cast<long long>(shown.size())));
  const char* label = metric_score_label(metric);

  if (json) {
    std::ostringstream o;
    o << "{\n \"config\": {\"source\": \"" << hpc::json_escape(source) << "\", \"n\": " << ds.n
      << ", \"dim\": " << ds.dim << ", \"k\": " << k_eff << ", \"metric\": \"" << metric_name(metric)
      << "\", \"score_label\": \"" << label << "\", \"seed\": " << seed << ", \"query\": \""
      << hpc::json_escape(query_source) << "\", \"mode\": \"" << mode << "\", \"repeats\": " << repeats
      << ", \"warmup\": " << warmup << ", \"statistic\": \"median\", \"cpus_available\": " << cpus << "},\n";
    o << " \"results\": [";
    for (std::size_t i = 0; i < show; ++i)
      o << (i ? ", " : "") << "{\"rank\": " << i + 1 << ", \"id\": " << shown[i].id << ", \"score\": "
        << num(shown[i].score) << "}";
    o << "],\n";
    if (mode != "omp") {
      o << " \"serial\": {\"median_ms\": " << num(ser_med) << ", \"runs_ms\": [";
      for (std::size_t i = 0; i < ser.ms.size(); ++i) o << (i ? ", " : "") << num(ser.ms[i]);
      o << "]},\n";
    }
    o << " \"openmp\": [";
    for (std::size_t i = 0; i < runs.size(); ++i) {
      const auto& r = runs[i];
      o << (i ? ",\n   " : "\n   ") << "{\"threads\": " << r.threads << ", \"threads_used\": " << r.used
        << ", \"median_ms\": " << num(r.median) << ", \"oversubscribed\": " << (r.oversubscribed ? "true" : "false");
      if (mode == "both")
        o << ", \"speedup\": " << num(r.speedup) << ", \"efficiency\": " << num(r.efficiency)
          << ", \"verified\": " << (r.verified ? "true" : "false") << ", \"max_abs_score_diff\": " << num(r.max_diff);
      o << ", \"runs_ms\": [";
      for (std::size_t j = 0; j < r.ms.size(); ++j) o << (j ? ", " : "") << num(r.ms[j]);
      o << "]}";
    }
    o << (runs.empty() ? "]" : "\n ]") << ",\n \"all_verified\": " << (all_ok ? "true" : "false") << "\n}\n";
    std::cout << o.str();
  } else {
    std::printf("SimdSearch  metric=%s  n=%zu  dim=%zu  k=%zu  source=%s  query=%s\n", metric_name(metric), ds.n,
                ds.dim, k_eff, source.c_str(), query_source.c_str());
    std::printf("\n%-6s %-10s %s\n", "rank", "id", label);
    for (std::size_t i = 0; i < show; ++i) std::printf("%-6zu %-10llu %.9f\n", i + 1, (unsigned long long)shown[i].id, shown[i].score);
    std::printf("\ntiming: median of %d run(s) after %d warm-up run(s); %d CPU(s) available\n", repeats, warmup, cpus);
    if (mode != "omp") std::printf("  serial        : %10.4f ms\n", ser_med);
    for (const auto& r : runs) {
      std::printf("  openmp T=%-4d : %10.4f ms", r.threads, r.median);
      if (mode == "both") std::printf("   speedup %6.3fx   efficiency %6.3f   %s", r.speedup, r.efficiency,
                                      r.verified ? "verified" : "MISMATCH");
      if (r.oversubscribed) std::printf("   [oversubscribed]");
      std::printf("\n");
    }
    if (mode == "both")
      std::printf(all_ok ? "\nverification: OpenMP results identical to serial (IDs exact, scores within 1e-9)\n"
                         : "\nverification: FAILED\n");
  }
  if (!all_ok) {
    for (const auto& r : runs)
      if (!r.verified) std::fprintf(stderr, "T=%d mismatch: %s\n", r.threads, r.why.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "help" || std::string(argv[1]) == "--help") {
      usage();
      return argc < 2 ? 2 : 0;
    }
    const std::string cmd = argv[1];
    hpc::Args args(argc, argv, 2);
    if (cmd == "search") return cmd_search(args);
    if (cmd == "generate") return cmd_generate(args);
    if (cmd == "info") return cmd_info();
    std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
    usage();
    return 2;
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  } catch (const std::bad_alloc&) {
    std::fprintf(stderr, "error: out of memory (reduce --n / --dim)\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
