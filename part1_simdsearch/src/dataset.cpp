#include "simdsearch/dataset.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "hpc/rng.hpp"

namespace simdsearch {

Metric parse_metric(const std::string& s) {
  if (s == "cosine" || s == "cos") return Metric::Cosine;
  if (s == "euclidean" || s == "euclid" || s == "l2") return Metric::Euclidean;
  throw std::invalid_argument("unknown metric '" + s + "' (use cosine or euclidean)");
}

const char* metric_name(Metric m) { return m == Metric::Cosine ? "cosine" : "euclidean"; }
const char* metric_score_label(Metric m) {
  return m == Metric::Cosine ? "cosine_similarity" : "euclidean_distance";
}

double vector_norm(const float* v, std::size_t dim) {
  double acc = 0.0;
  for (std::size_t j = 0; j < dim; ++j) acc += static_cast<double>(v[j]) * static_cast<double>(v[j]);
  return std::sqrt(acc);
}

void Dataset::compute_norms() {
  norms.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) norms[i] = vector_norm(row(i), dim);
}

void Dataset::validate() const {
  if (n == 0 || dim == 0) throw std::invalid_argument("dataset is empty (n and dim must be > 0)");
  if (data.size() != n * dim) throw std::invalid_argument("dataset buffer size != n*dim");
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (!std::isfinite(data[i]))
      throw std::invalid_argument("dataset contains NaN/Inf at vector " + std::to_string(i / dim) +
                                  ", component " + std::to_string(i % dim));
  }
}

static void check_shape(std::size_t n, std::size_t dim) {
  if (n == 0) throw std::invalid_argument("n must be > 0");
  if (dim == 0) throw std::invalid_argument("dim must be > 0");
  if (n > std::numeric_limits<std::size_t>::max() / dim / sizeof(float))
    throw std::invalid_argument("n*dim*sizeof(float) overflows size_t");
}

Dataset generate_dataset(std::size_t n, std::size_t dim, std::uint64_t seed) {
  check_shape(n, dim);
  Dataset ds;
  ds.n = n;
  ds.dim = dim;
  ds.data.resize(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t state = hpc::mix64(seed ^ hpc::mix64(static_cast<std::uint64_t>(i) + 1));
    float* r = ds.data.data() + i * dim;
    for (std::size_t j = 0; j < dim; ++j) r[j] = hpc::uniform_pm1_f(state);
  }
  ds.compute_norms();
  return ds;
}

std::vector<float> generate_query(std::size_t dim, std::uint64_t seed) {
  if (dim == 0) throw std::invalid_argument("dim must be > 0");
  std::vector<float> q(dim);
  // Separate domain constant: the query stream never coincides with a data row stream.
  std::uint64_t state = hpc::mix64(seed ^ 0x51D5EA7C4E57A11DULL);
  for (std::size_t j = 0; j < dim; ++j) q[j] = hpc::uniform_pm1_f(state);
  return q;
}

}  // namespace simdsearch
