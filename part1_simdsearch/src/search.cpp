#include "simdsearch/search.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace simdsearch {
namespace {

// Internal candidate. `key` is a "larger is better" value shared by both metrics:
//   cosine    -> key = similarity
//   euclidean -> key = -(squared distance)   (sqrt is monotonic; applied once at output)
struct Entry {
  double key;
  std::uint64_t id;
};

// Strict total order: higher key first; equal keys -> smaller id first.
// This makes the result independent of thread count, schedule and merge order.
inline bool better(const Entry& a, const Entry& b) {
  return a.key > b.key || (a.key == b.key && a.id < b.id);
}

// Bounded selection heap of the K best entries seen so far. The heap front is the
// WORST retained entry, so a new candidate is rejected with a single comparison.
class TopK {
 public:
  explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }  // one allocation, up front
  inline void push(double key, std::uint64_t id) {
    const Entry e{key, id};
    if (heap_.size() < k_) {
      heap_.push_back(e);
      std::push_heap(heap_.begin(), heap_.end(), better);
    } else if (better(e, heap_.front())) {
      std::pop_heap(heap_.begin(), heap_.end(), better);
      heap_.back() = e;
      std::push_heap(heap_.begin(), heap_.end(), better);
    }
  }
  const std::vector<Entry>& entries() const { return heap_; }
  std::vector<Entry> take() { return std::move(heap_); }

 private:
  std::size_t k_;
  std::vector<Entry> heap_;
};

// ----- per-vector scoring functors (identical for serial and OpenMP) ----------
struct CosineKey {
  const float* base;
  const double* norms;
  const double* q;
  double qnorm;
  std::size_t dim;
  double operator()(std::size_t i) const {
    const double nv = norms[i];
    if (nv == 0.0 || qnorm == 0.0) return 0.0;  // zero vector: similarity defined as 0
    const float* v = base + i * dim;
    double dot = 0.0;
    for (std::size_t j = 0; j < dim; ++j) dot += static_cast<double>(v[j]) * q[j];
    const double s = dot / (nv * qnorm);
    return s > 1.0 ? 1.0 : (s < -1.0 ? -1.0 : s);  // clamp rounding noise
  }
};

struct EuclidKey {
  const float* base;
  const double* q;
  std::size_t dim;
  double operator()(std::size_t i) const {
    const float* v = base + i * dim;
    double acc = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const double d = static_cast<double>(v[j]) - q[j];
      acc += d * d;
    }
    return -acc;
  }
};

template <class F>
std::vector<Entry> topk_serial(std::size_t n, std::size_t k, const F& f) {
  TopK top(k);
  for (std::size_t i = 0; i < n; ++i) top.push(f(i), i);
  return top.take();
}

// Thread-local heaps padded to a cache line so neighbouring heaps never share one.
struct alignas(64) Local {
  explicit Local(std::size_t k) : top(k) {}
  TopK top;
};

template <class F>
std::vector<Entry> topk_openmp(std::size_t n, std::size_t k, int threads, const F& f, int* used) {
  std::vector<Local> locals;
  locals.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) locals.emplace_back(k);  // allocated BEFORE the parallel region

  int team = 1;
  const std::int64_t N = static_cast<std::int64_t>(n);
#pragma omp parallel num_threads(threads)
  {
#pragma omp single nowait
    team = omp_get_num_threads();
    TopK& mine = locals[static_cast<std::size_t>(omp_get_thread_num())].top;  // private: no locks
#pragma omp for schedule(static) nowait
    for (std::int64_t i = 0; i < N; ++i) mine.push(f(static_cast<std::size_t>(i)), static_cast<std::uint64_t>(i));
  }  // implicit barrier: all threads finished

  if (used) *used = team;
  // Merge: concatenate the <= T*K survivors; final selection happens in to_hits().
  std::vector<Entry> all;
  all.reserve(static_cast<std::size_t>(threads) * k);
  for (const auto& l : locals) all.insert(all.end(), l.top.entries().begin(), l.top.entries().end());
  return all;
}

std::vector<Hit> to_hits(std::vector<Entry>& cand, Metric metric, std::size_t k) {
  const std::size_t kk = std::min(k, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(kk), cand.end(), better);
  std::vector<Hit> hits;
  hits.reserve(kk);
  for (std::size_t i = 0; i < kk; ++i) {
    const double s = metric == Metric::Cosine ? cand[i].key : std::sqrt(-cand[i].key);
    hits.push_back({cand[i].id, s});
  }
  return hits;
}

struct Prepared {
  std::vector<double> q;
  double qnorm = 0.0;
  std::size_t k = 0;
};

Prepared prepare(const Dataset& ds, const std::vector<float>& query, std::size_t k, Metric metric) {
  if (ds.n == 0 || ds.dim == 0) throw std::invalid_argument("dataset is empty");
  if (ds.data.size() != ds.n * ds.dim) throw std::invalid_argument("dataset buffer size != n*dim");
  if (k == 0) throw std::invalid_argument("k must be >= 1");
  if (query.size() != ds.dim)
    throw std::invalid_argument("query dimension " + std::to_string(query.size()) +
                                " != dataset dimension " + std::to_string(ds.dim));
  if (metric == Metric::Cosine && ds.norms.size() != ds.n)
    throw std::invalid_argument("dataset norms missing: call Dataset::compute_norms()");
  Prepared p;
  p.q.resize(query.size());
  double acc = 0.0;
  for (std::size_t j = 0; j < query.size(); ++j) {
    if (!std::isfinite(query[j])) throw std::invalid_argument("query contains NaN/Inf");
    p.q[j] = static_cast<double>(query[j]);
    acc += p.q[j] * p.q[j];
  }
  p.qnorm = std::sqrt(acc);
  p.k = std::min(k, ds.n);
  return p;
}

}  // namespace

std::vector<Hit> search_serial(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                               Metric metric) {
  const Prepared p = prepare(ds, query, k, metric);
  std::vector<Entry> cand;
  if (metric == Metric::Cosine)
    cand = topk_serial(ds.n, p.k, CosineKey{ds.data.data(), ds.norms.data(), p.q.data(), p.qnorm, ds.dim});
  else
    cand = topk_serial(ds.n, p.k, EuclidKey{ds.data.data(), p.q.data(), ds.dim});
  return to_hits(cand, metric, p.k);
}

std::vector<Hit> search_openmp(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                               Metric metric, int threads, int* threads_used) {
  const Prepared p = prepare(ds, query, k, metric);
  if (threads <= 0) threads = omp_get_max_threads();
  std::vector<Entry> cand;
  if (metric == Metric::Cosine)
    cand = topk_openmp(ds.n, p.k, threads,
                       CosineKey{ds.data.data(), ds.norms.data(), p.q.data(), p.qnorm, ds.dim}, threads_used);
  else
    cand = topk_openmp(ds.n, p.k, threads, EuclidKey{ds.data.data(), p.q.data(), ds.dim}, threads_used);
  return to_hits(cand, metric, p.k);
}

std::vector<Hit> search_reference(const Dataset& ds, const std::vector<float>& query, std::size_t k,
                                  Metric metric) {
  if (k == 0) throw std::invalid_argument("k must be >= 1");
  if (query.size() != ds.dim) throw std::invalid_argument("query dimension mismatch");
  std::vector<Hit> all(ds.n);
  double nq2 = 0;
  for (float x : query) nq2 += double(x) * double(x);
  for (std::size_t i = 0; i < ds.n; ++i) {
    double dot = 0, nv2 = 0, sq = 0;
    for (std::size_t j = 0; j < ds.dim; ++j) {
      const double a = ds.data[i * ds.dim + j], b = query[j];
      dot += a * b;
      nv2 += a * a;
      sq += (a - b) * (a - b);
    }
    double score;
    if (metric == Metric::Cosine) {
      score = (nv2 == 0.0 || nq2 == 0.0) ? 0.0 : dot / (std::sqrt(nv2) * std::sqrt(nq2));
      score = std::min(1.0, std::max(-1.0, score));
    } else {
      score = std::sqrt(sq);
    }
    all[i] = {i, score};
  }
  const bool desc = metric == Metric::Cosine;
  std::stable_sort(all.begin(), all.end(), [desc](const Hit& a, const Hit& b) {
    if (a.score != b.score) return desc ? a.score > b.score : a.score < b.score;
    return a.id < b.id;
  });
  all.resize(std::min(k, all.size()));
  return all;
}

MatchReport compare_hits(const std::vector<Hit>& a, const std::vector<Hit>& b, double tol) {
  MatchReport r;
  if (a.size() != b.size()) {
    r.ok = false;
    r.why = "different result sizes " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
    return r;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i].score - b[i].score);
    r.max_abs_score_diff = std::max(r.max_abs_score_diff, d);
    if (a[i].id != b[i].id) {
      r.ok = false;
      r.why = "rank " + std::to_string(i) + ": id " + std::to_string(a[i].id) + " vs " + std::to_string(b[i].id);
      return r;
    }
    if (!(d <= tol * (1.0 + std::fabs(b[i].score)))) {
      r.ok = false;
      r.why = "rank " + std::to_string(i) + ": score differs by " + std::to_string(d);
      return r;
    }
  }
  return r;
}

}  // namespace simdsearch
