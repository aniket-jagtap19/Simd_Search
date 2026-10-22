#pragma once
#include <string>
#include <vector>

#include "simdsearch/dataset.hpp"

namespace simdsearch {

// Binary format ("SIMDSRC1"):  8-byte magic | uint64 n | uint64 dim | n*dim float32
// (little-endian, row-major).   CSV format (*.csv): one vector per line,
// comma/whitespace separated floats, '#' starts a comment line.
void save_binary(const Dataset& ds, const std::string& path);
void save_csv(const Dataset& ds, const std::string& path);
// Format chosen by extension (.csv => CSV, otherwise binary). Validates the result
// (finite values, consistent dimensions) and computes norms.
Dataset load_dataset(const std::string& path);

}  // namespace simdsearch
