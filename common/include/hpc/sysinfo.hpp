#pragma once
// Hardware / software environment discovery, recorded next to every benchmark.
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "hpc_build_info.h"

namespace hpc {

// CPU limit imposed by a container (cgroup v2 / v1), or 0 when unlimited/unknown.
inline int cgroup_cpu_limit() {
  std::ifstream f2("/sys/fs/cgroup/cpu.max");
  if (f2) {
    std::string q;
    long long period = 0;
    if (f2 >> q >> period && q != "max" && period > 0) {
      long long quota = std::stoll(q);
      if (quota > 0) return static_cast<int>((quota + period - 1) / period);
    }
  }
  std::ifstream fq("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), fp("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
  long long quota = -1, period = 0;
  if (fq && fp && (fq >> quota) && (fp >> period) && quota > 0 && period > 0)
    return static_cast<int>((quota + period - 1) / period);
  return 0;
}

// Number of CPUs this process may actually use (affinity mask and cgroup quota).
inline int available_cpus() {
  int n = 0;
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
#endif
  if (n <= 0) {
    unsigned h = std::thread::hardware_concurrency();
    n = h ? static_cast<int>(h) : 1;
  }
  int cg = cgroup_cpu_limit();
  if (cg > 0 && cg < n) n = cg;
  return n < 1 ? 1 : n;
}

inline std::string cpu_model() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto c = line.find(':');
      if (c != std::string::npos) return line.substr(c + 2);
    }
  }
  return "unknown";
}

inline long long mem_total_mb() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  long long kb = 0;
  while (f >> key >> kb) {
    if (key == "MemTotal:") return kb / 1024;
    std::string rest;
    std::getline(f, rest);
  }
  return 0;
}

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

inline std::string env_or(const char* name) {
  const char* v = std::getenv(name);
  return v ? v : "(unset)";
}

using KV = std::vector<std::pair<std::string, std::string>>;

inline KV environment_info() {
  KV kv;
  char host[256] = "unknown";
#ifdef __linux__
  gethostname(host, sizeof(host) - 1);
#endif
  std::time_t t = std::time(nullptr);
  char ts[64] = "unknown";
  std::tm tmv{};
#ifdef __linux__
  gmtime_r(&t, &tmv);
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
#endif
  kv.push_back({"timestamp_utc", ts});
  kv.push_back({"hostname", host});
  kv.push_back({"cpu_model", cpu_model()});
  kv.push_back({"logical_cpus_hardware_concurrency", std::to_string(std::thread::hardware_concurrency())});
  kv.push_back({"cpus_available_to_process", std::to_string(available_cpus())});
  kv.push_back({"memory_total_mb", std::to_string(mem_total_mb())});
#ifdef __VERSION__
  kv.push_back({"compiler", __VERSION__});
#endif
  kv.push_back({"cxx_standard_macro", std::to_string(__cplusplus)});
  kv.push_back({"build_type", HPC_BUILD_TYPE});
  kv.push_back({"compile_flags", HPC_CXX_FLAGS});
#ifdef _OPENMP
  kv.push_back({"openmp_version_macro", std::to_string(_OPENMP)});
  kv.push_back({"omp_max_threads_default", std::to_string(omp_get_max_threads())});
#endif
  kv.push_back({"OMP_NUM_THREADS", env_or("OMP_NUM_THREADS")});
  kv.push_back({"OMP_PROC_BIND", env_or("OMP_PROC_BIND")});
  kv.push_back({"OMP_PLACES", env_or("OMP_PLACES")});
  kv.push_back({"OMP_SCHEDULE", env_or("OMP_SCHEDULE")});
  kv.push_back({"clock", "std::chrono::steady_clock (monotonic)"});
  return kv;
}

inline std::string to_json(const KV& kv) {
  std::ostringstream o;
  o << "{\n";
  for (std::size_t i = 0; i < kv.size(); ++i)
    o << "  \"" << json_escape(kv[i].first) << "\": \"" << json_escape(kv[i].second) << "\""
      << (i + 1 < kv.size() ? "," : "") << "\n";
  o << "}\n";
  return o.str();
}

}  // namespace hpc
