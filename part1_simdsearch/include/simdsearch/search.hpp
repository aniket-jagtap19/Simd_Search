#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "simdsearch/dataset.hpp"

namespace simdsearch {

// One search hit. score = cosine similarity in [-1,1] (higher is better) or the
// Euclidean distance >= 0 (lower is better).
struct Hit {
  std::uint64_t id;
  double score;
};

// Exact Top-K, sequential baseline: one pass, one bounded heap.
// k is clamped to ds.n. Throws std::invalid_argument on invalid input
// (k == 0, empty dataset, dimension mismatch, NaN/Inf in the query, missing norms).
std::vector<Hit> search_serial(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                               Metric metric);

// Exact Top-K, OpenMP: vectors are split statically across `threads` threads; each
// thread owns a private bounded heap (no shared mutable state in the hot loop); the
// T local heaps (<= T*K candidates) are merged once after the parallel region.
// threads <= 0 means "OpenMP default". *threads_used (optional) reports the team size.
std::vector<Hit> search_openmp(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                               Metric metric, int threads, int* threads_used = nullptr);

// Independent brute-force oracle used only by tests: scores everything with a
// different code path and fully sorts.
std::vector<Hit> search_reference(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                                  Metric metric);

struct MatchReport {
  bool ok = true;
  double max_abs_score_diff = 0.0;
  std::string why;
};
// IDs and order must match exactly; scores must agree to
// |a-b| <= tol * (1 + |b|).
MatchReport compare_hits(const std::vector<Hit>& a, const std::vector<Hit>& b, double tol);

}  // namespace simdsearch
