#include "matvec/matvec_mpi.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "matvec/matvec.hpp"

namespace matvec {

std::vector<int> block_row_counts(std::size_t rows, int procs) {
  if (procs <= 0) throw std::invalid_argument("procs must be > 0");
  std::vector<int> c(static_cast<std::size_t>(procs));
  const std::size_t base = rows / static_cast<std::size_t>(procs), rem = rows % static_cast<std::size_t>(procs);
  for (std::size_t r = 0; r < c.size(); ++r) c[r] = static_cast<int>(base + (r < rem ? 1 : 0));
  return c;
}

std::vector<int> block_row_displs(const std::vector<int>& counts) {
  std::vector<int> d(counts.size(), 0);
  for (std::size_t r = 1; r < counts.size(); ++r) d[r] = d[r - 1] + counts[r - 1];
  return d;
}

MpiMatVec::MpiMatVec(MPI_Comm comm, std::size_t rows, std::size_t cols) : comm_(comm), rows_(rows), cols_(cols) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  validate_dims(rows, cols);  // same inputs on all ranks -> same exception everywhere (no deadlock)
  if (rows > static_cast<std::size_t>(INT_MAX)) throw std::invalid_argument("rows exceeds INT_MAX");
  row_counts_ = block_row_counts(rows, size_);
  row_displs_ = block_row_displs(row_counts_);
  const std::size_t max_rows = static_cast<std::size_t>(row_counts_[0]);  // rank 0 always has the most
  if (max_rows * cols > static_cast<std::size_t>(INT_MAX))
    throw std::invalid_argument("per-rank block exceeds MPI int count limit; use more processes");
  a_counts_.resize(row_counts_.size());
  a_displs_.resize(row_counts_.size());
  for (std::size_t r = 0; r < row_counts_.size(); ++r) {
    a_counts_[r] = static_cast<int>(static_cast<std::size_t>(row_counts_[r]) * cols);
    a_displs_[r] = static_cast<int>(static_cast<std::size_t>(row_displs_[r]) * cols);
  }
  local_rows_ = static_cast<std::size_t>(row_counts_[static_cast<std::size_t>(rank_)]);
  local_A_.assign(std::max<std::size_t>(1, local_rows_ * cols), 0.0);  // >=1 element: never a null buffer
  local_y_.assign(std::max<std::size_t>(1, local_rows_), 0.0);
  x_.assign(cols, 0.0);
}

void MpiMatVec::scatter_matrix(const double* A_root) {
  MPI_Scatterv(A_root, a_counts_.data(), a_displs_.data(), MPI_DOUBLE, local_A_.data(),
               a_counts_[static_cast<std::size_t>(rank_)], MPI_DOUBLE, 0, comm_);
}

void MpiMatVec::broadcast_vector(const double* x_root) {
  if (rank_ == 0) std::copy(x_root, x_root + cols_, x_.begin());
  MPI_Bcast(x_.data(), static_cast<int>(cols_ > static_cast<std::size_t>(INT_MAX) ? INT_MAX : cols_), MPI_DOUBLE, 0, comm_);
}

void MpiMatVec::compute_local() { multiply_serial(local_A_.data(), x_.data(), local_y_.data(), local_rows_, cols_); }

void MpiMatVec::gather_result(double* y_root) {
  MPI_Gatherv(local_y_.data(), row_counts_[static_cast<std::size_t>(rank_)], MPI_DOUBLE, y_root, row_counts_.data(),
              row_displs_.data(), MPI_DOUBLE, 0, comm_);
}

PhaseTimes MpiMatVec::run(const double* A_root, const double* x_root, double* y_root, bool include_scatter) {
  MPI_Barrier(comm_);  // common start line (outside the timed region)
  const double t0 = MPI_Wtime();
  if (include_scatter) scatter_matrix(A_root);
  const double t1 = MPI_Wtime();
  broadcast_vector(x_root);
  const double t2 = MPI_Wtime();
  compute_local();
  const double t3 = MPI_Wtime();
  gather_result(y_root);
  const double t4 = MPI_Wtime();
  double mine[5] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0}, mx[5];
  MPI_Allreduce(mine, mx, 5, MPI_DOUBLE, MPI_MAX, comm_);
  return {mx[0] * 1e3, mx[1] * 1e3, mx[2] * 1e3, mx[3] * 1e3, mx[4] * 1e3};
}

}  // namespace matvec
