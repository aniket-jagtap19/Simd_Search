// MPI correctness test; run under mpiexec with any number of processes (ctest runs 1,2,3,4,8).
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <numeric>
#include <vector>

#include "matvec/matvec.hpp"
#include "matvec/matvec_mpi.hpp"

using namespace matvec;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int failures = 0;
  // Guard against a broken launcher: if mpiexec/library mismatch makes every process a singleton
  // ("rank 0 of 1"), a test would pass vacuously. --expect-np N makes that an error.
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--expect-np" && std::atoi(argv[i + 1]) != size) {
      if (rank == 0)
        std::printf("FAIL: expected %d processes but MPI_COMM_WORLD has %d. "
                    "mpiexec and the MPI library probably do not match (see README troubleshooting).\n",
                    std::atoi(argv[i + 1]), size);
      MPI_Finalize();
      return 1;
    }
  auto check = [&](bool c, const char* what) {
    if (!c) { ++failures; if (rank == 0) std::printf("FAIL: %s\n", what); }
  };

  // Partition properties, including more processes than rows.
  for (std::size_t rows : {1u, 2u, 7u, 8u, 9u, 100u, 2048u}) {
    for (int p : {1, 2, 3, 4, 8, 16}) {
      auto c = block_row_counts(rows, p);
      auto d = block_row_displs(c);
      int total = std::accumulate(c.begin(), c.end(), 0);
      int mn = *std::min_element(c.begin(), c.end()), mx = *std::max_element(c.begin(), c.end());
      check(total == (int)rows && mx - mn <= 1 && d[0] == 0, "row partition covers all rows, counts differ by <= 1");
      for (int r = 1; r < p; ++r) check(d[r] == d[r - 1] + c[r - 1], "contiguous displacements");
    }
  }

  // Distributed multiply vs serial reference for awkward shapes.
  const std::size_t shapes[][2] = {{1, 1}, {1, 9}, {3, 5}, {7, 7}, {13, 4}, {2, 50}, {100, 37}, {64, 64}, {256, 256}};
  for (auto& s : shapes) {
    const std::size_t m = s[0], n = s[1];
    Problem p;
    std::vector<double> yref, y;
    if (rank == 0) {
      p = make_problem(m, n, 77);
      yref.resize(m);
      y.assign(m, -1.0);
      multiply_serial(p.A.data(), p.x.data(), yref.data(), m, n);
    }
    MpiMatVec mm(MPI_COMM_WORLD, m, n);
    const double* A = rank == 0 ? p.A.data() : nullptr;
    const double* x = rank == 0 ? p.x.data() : nullptr;
    double* yp = rank == 0 ? y.data() : nullptr;
    mm.run(A, x, yp, true);
    if (rank == 0) check(max_relative_error(y.data(), yref.data(), m) <= kTolerance, "full run matches serial");
    if (rank == 0) std::fill(y.begin(), y.end(), -1.0);
    mm.run(A, x, yp, false);  // resident matrix (already scattered by the previous run)
    if (rank == 0) check(max_relative_error(y.data(), yref.data(), m) <= kTolerance, "resident run matches serial");
    check(mm.local_rows() == (std::size_t)block_row_counts(m, size)[rank], "local row count");
  }

  // Invalid configuration must throw on EVERY rank (no deadlock, no partial failure).
  int threw = 0;
  try { MpiMatVec bad(MPI_COMM_WORLD, 0, 10); } catch (const std::invalid_argument&) { threw = 1; }
  check(threw == 1, "rows == 0 rejected");
  threw = 0;
  try { MpiMatVec bad(MPI_COMM_WORLD, 10, 0); } catch (const std::invalid_argument&) { threw = 1; }
  check(threw == 1, "cols == 0 rejected");

  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) std::printf("test_matvec_mpi with %d process(es): %s\n", size, total_failures ? "FAILED" : "all checks passed");
  MPI_Finalize();
  return total_failures ? 1 : 0;
}
