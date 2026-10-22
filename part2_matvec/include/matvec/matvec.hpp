#pragma once
// Matrix-vector multiplication y = A x  (A: rows x cols, row-major doubles).
// The SAME row kernel is used by the serial, OpenMP and MPI versions, so all three
// perform bit-identical arithmetic per output element.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace matvec {

struct Problem {
  std::size_t rows = 0, cols = 0;
  std::vector<double> A;  // rows*cols, row-major, contiguous
  std::vector<double> x;  // cols
};

// Throws std::invalid_argument for zero dimensions or rows*cols overflow.
void validate_dims(std::size_t rows, std::size_t cols);

// Deterministic data (SplitMix64, uniform in [-1,1)), identical on every platform.
Problem make_problem(std::size_t rows, std::size_t cols, std::uint64_t seed);

// y[i] = sum_j A[i*cols + j] * x[j], one row at a time, strict left-to-right summation.
// Works on any contiguous block of rows (this is what each MPI rank calls).
void multiply_serial(const double* A, const double* x, double* y, std::size_t rows, std::size_t cols);

// Row-parallel OpenMP version: rows are independent -> no reductions, no races.
// threads <= 0 -> OpenMP default. *threads_used (optional) reports the real team size.
void multiply_openmp(const double* A, const double* x, double* y, std::size_t rows, std::size_t cols,
                     int threads, int* threads_used = nullptr);

// max_i |y[i]-ref[i]| / max(1,|ref[i]|). Returns +inf if any value is NaN.
double max_relative_error(const double* y, const double* ref, std::size_t n);

constexpr double kTolerance = 1e-12;  // acceptance threshold (identical arithmetic gives 0)

}  // namespace matvec
