#pragma once
// CSV schema shared by the OpenMP and MPI matrix-vector benchmarks.
#include <cstdio>
#include <fstream>
#include <string>

#include "hpc/timer.hpp"

namespace matvec {

struct Row {
  std::size_t rows = 0, cols = 0;
  std::string impl;         // serial | openmp | mpi_full | mpi_resident
  int workers = 1;
  std::string worker_type;  // none | threads | processes
  int repeats = 0, warmup = 0;
  hpc::Stats stats;
  double scatter_ms = 0, bcast_ms = 0, compute_ms = 0, gather_ms = 0;  // MPI phase medians
  bool oversubscribed = false, verified = false;
  double max_rel_err = 0;
  std::uint64_t seed = 0;
};

inline const char* csv_header() {
  return "rows,cols,impl,workers,worker_type,repeats,warmup,stat,time_ms,min_ms,mean_ms,stddev_ms,"
         "scatter_ms,bcast_ms,compute_ms,gather_ms,oversubscribed,verified,max_rel_err,seed\n";
}

// Appends one row; writes the header first if the file is new/empty.
inline void append_row(const std::string& path, const Row& r) {
  bool need_header = false;
  {
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    need_header = !probe || probe.tellg() == 0;
  }
  std::ofstream out(path, std::ios::app);
  if (!out) throw std::runtime_error("cannot write " + path);
  if (need_header) out << csv_header();
  char buf[512];
  std::snprintf(buf, sizeof buf, "%zu,%zu,%s,%d,%s,%d,%d,median,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.3e,%llu\n",
                r.rows, r.cols, r.impl.c_str(), r.workers, r.worker_type.c_str(), r.repeats, r.warmup, r.stats.median,
                r.stats.min, r.stats.mean, r.stats.stddev, r.scatter_ms, r.bcast_ms, r.compute_ms, r.gather_ms,
                r.oversubscribed ? 1 : 0, r.verified ? 1 : 0, r.max_rel_err, (unsigned long long)r.seed);
  out << buf;
}

}  // namespace matvec
