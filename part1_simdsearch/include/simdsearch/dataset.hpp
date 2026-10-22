#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simdsearch {

enum class Metric { Cosine, Euclidean };

// Accepts: cosine | cos | euclidean | euclid | l2. Throws std::invalid_argument otherwise.
Metric parse_metric(const std::string& s);
const char* metric_name(Metric m);
// "cosine_similarity" (higher is better) or "euclidean_distance" (lower is better).
const char* metric_score_label(Metric m);

// A dataset of n vectors of `dim` float32 components stored row-major in ONE
// contiguous allocation: vector i occupies data[i*dim .. i*dim+dim).
// `norms` holds the Euclidean norm of every vector (double precision). It is an
// index-build product (like an ANN index would be): computed once by
// compute_norms(), outside every timed region, and used by cosine similarity.
struct Dataset {
  std::size_t n = 0, dim = 0;
  std::vector<float> data;
  std::vector<double> norms;

  const float* row(std::size_t i) const { return data.data() + i * dim; }
  void compute_norms();
  // Throws std::invalid_argument if sizes are inconsistent or any value is NaN/Inf.
  void validate() const;
};

double vector_norm(const float* v, std::size_t dim);

// Reproducible synthetic data: components i.i.d. uniform in [-1,1) from a
// platform-independent SplitMix64 stream; every row derives its own stream from
// (seed,row) so the result does not depend on generation order or thread count.
Dataset generate_dataset(std::size_t n, std::size_t dim, std::uint64_t seed);
std::vector<float> generate_query(std::size_t dim, std::uint64_t seed);

}  // namespace simdsearch
