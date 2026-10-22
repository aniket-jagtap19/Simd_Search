#pragma once
// Monotonic timing and summary statistics used by every benchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace hpc {

using Clock = std::chrono::steady_clock;  // monotonic, never adjusted by NTP

inline double now_ms() {
  return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

struct Stats {
  double median = 0, min = 0, max = 0, mean = 0, stddev = 0;
  std::size_t n = 0;
};

inline Stats summarize(std::vector<double> v) {
  Stats s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  s.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  if (v.size() > 1) {
    double acc = 0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(acc / static_cast<double>(v.size() - 1));
  }
  return s;
}

}  // namespace hpc
