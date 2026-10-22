#include "matvec/matvec.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "hpc/rng.hpp"

namespace matvec {
namespace {
inline double dot_row(const double* a, const double* x, std::size_t n) {
  double s = 0.0;
  for (std::size_t j = 0; j < n; ++j) s += a[j] * x[j];  // fixed order -> reproducible
  return s;
}
}  // namespace

void validate_dims(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) throw std::invalid_argument("matrix dimensions must be > 0");
  if (rows > std::numeric_limits<std::size_t>::max() / cols / sizeof(double))
    throw std::invalid_argument("rows*cols*sizeof(double) overflows size_t");
}

Problem make_problem(std::size_t rows, std::size_t cols, std::uint64_t seed) {
  validate_dims(rows, cols);
  Problem p;
  p.rows = rows;
  p.cols = cols;
  p.A.resize(rows * cols);
  p.x.resize(cols);
  std::uint64_t s = hpc::mix64(seed ^ 0xA11CE5ULL);
  for (auto& v : p.A) v = hpc::uniform_pm1_d(s);
  std::uint64_t sx = hpc::mix64(seed ^ 0x0EC70AULL);
  for (auto& v : p.x) v = hpc::uniform_pm1_d(sx);
  return p;
}

void multiply_serial(const double* A, const double* x, double* y, std::size_t rows, std::size_t cols) {
  if (rows == 0) return;
  if (!A || !x || !y || cols == 0) throw std::invalid_argument("multiply_serial: null pointer or zero cols");
  for (std::size_t i = 0; i < rows; ++i) y[i] = dot_row(A + i * cols, x, cols);
}

void multiply_openmp(const double* A, const double* x, double* y, std::size_t rows, std::size_t cols, int threads,
                     int* threads_used) {
  if (rows == 0) return;
  if (!A || !x || !y || cols == 0) throw std::invalid_argument("multiply_openmp: null pointer or zero cols");
  if (threads <= 0) threads = omp_get_max_threads();
  const std::int64_t R = static_cast<std::int64_t>(rows);
  int team = 1;
#pragma omp parallel num_threads(threads)
  {
#pragma omp single nowait
    team = omp_get_num_threads();
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < R; ++i)
      y[i] = dot_row(A + static_cast<std::size_t>(i) * cols, x, cols);  // each thread writes disjoint y[i]
  }
  if (threads_used) *threads_used = team;
}

double max_relative_error(const double* y, const double* ref, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(y[i]) || std::isnan(ref[i])) return std::numeric_limits<double>::infinity();
    m = std::max(m, std::fabs(y[i] - ref[i]) / std::max(1.0, std::fabs(ref[i])));
  }
  return m;
}

}  // namespace matvec
