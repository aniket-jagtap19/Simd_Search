#pragma once
// MPI row-block distributed y = A x.
//
//   rank 0:  A (rows x cols), x           ──MPI_Scatterv(A rows)──▶  each rank: A_local (rows_r x cols)
//   rank 0:  x                            ──MPI_Bcast(x)──────────▶  each rank: full x
//   every rank: y_local = A_local * x                       (no communication)
//   each rank: y_local ──MPI_Gatherv──▶ rank 0: y
//
// Rows are split in contiguous blocks; the first (rows % P) ranks receive one extra
// row, so counts differ by at most 1. Ranks with 0 rows (P > rows) are legal.
#include <mpi.h>

#include <cstddef>
#include <vector>

namespace matvec {

std::vector<int> block_row_counts(std::size_t rows, int procs);
std::vector<int> block_row_displs(const std::vector<int>& counts);

struct PhaseTimes {            // milliseconds, maximum over ranks
  double scatter_ms = 0, bcast_ms = 0, compute_ms = 0, gather_ms = 0, total_ms = 0;
};

class MpiMatVec {
 public:
  // Collective. Throws std::invalid_argument (identically on every rank) for invalid
  // dimensions or when a block would exceed MPI's int element-count limit.
  MpiMatVec(MPI_Comm comm, std::size_t rows, std::size_t cols);

  int rank() const { return rank_; }
  int size() const { return size_; }
  std::size_t local_rows() const { return local_rows_; }

  // --- collective building blocks (A_root, x_root, y_root only used on rank 0) ---
  void scatter_matrix(const double* A_root);
  void broadcast_vector(const double* x_root);
  void compute_local();
  void gather_result(double* y_root);

  // Barrier, then timed scatter(optional)/bcast/compute/gather. Returns max-over-ranks times
  // (identical on all ranks). include_scatter=false reuses the already-distributed matrix.
  PhaseTimes run(const double* A_root, const double* x_root, double* y_root, bool include_scatter);

 private:
  MPI_Comm comm_;
  int rank_ = 0, size_ = 1;
  std::size_t rows_, cols_, local_rows_ = 0;
  std::vector<int> row_counts_, row_displs_, a_counts_, a_displs_;
  std::vector<double> local_A_, x_, local_y_;
};

}  // namespace matvec
