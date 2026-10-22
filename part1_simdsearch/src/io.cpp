#include "simdsearch/io.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace simdsearch {
namespace {

constexpr char kMagic[8] = {'S', 'I', 'M', 'D', 'S', 'R', 'C', '1'};
constexpr std::uint64_t kHeaderBytes = 8 + 8 + 8;

bool is_little_endian() {
  const std::uint16_t x = 1;
  unsigned char b;
  std::memcpy(&b, &x, 1);
  return b == 1;
}

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

Dataset load_binary(const std::string& path) {
  if (!is_little_endian()) throw std::runtime_error("binary format requires a little-endian host");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  in.seekg(0, std::ios::end);
  const std::uint64_t file_size = static_cast<std::uint64_t>(in.tellg());
  in.seekg(0);
  if (file_size < kHeaderBytes) throw std::runtime_error("'" + path + "': file too small for header");
  char magic[8];
  std::uint64_t n = 0, dim = 0;
  in.read(magic, 8);
  in.read(reinterpret_cast<char*>(&n), 8);
  in.read(reinterpret_cast<char*>(&dim), 8);
  if (!in || std::memcmp(magic, kMagic, 8) != 0)
    throw std::runtime_error("'" + path + "': bad magic (not a SIMDSRC1 binary dataset; use .csv for text)");
  if (n == 0 || dim == 0) throw std::runtime_error("'" + path + "': n and dim must be > 0");
  if (n > std::numeric_limits<std::uint64_t>::max() / dim / sizeof(float))
    throw std::runtime_error("'" + path + "': header n*dim overflows");
  const std::uint64_t expected = kHeaderBytes + n * dim * sizeof(float);
  if (file_size != expected)
    throw std::runtime_error("'" + path + "': size " + std::to_string(file_size) + " != expected " +
                             std::to_string(expected) + " bytes (truncated or corrupt)");
  Dataset ds;
  ds.n = static_cast<std::size_t>(n);
  ds.dim = static_cast<std::size_t>(dim);
  ds.data.resize(ds.n * ds.dim);
  in.read(reinterpret_cast<char*>(ds.data.data()), static_cast<std::streamsize>(n * dim * sizeof(float)));
  if (!in) throw std::runtime_error("'" + path + "': read error");
  return ds;
}

Dataset load_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  Dataset ds;
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    for (char& c : line)
      if (c == ',' || c == ';') c = ' ';
    const char* p = line.c_str();
    char* end = nullptr;
    std::size_t count = 0;
    while (true) {
      while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
      if (!*p) break;
      float v = std::strtof(p, &end);
      if (end == p)
        throw std::runtime_error("'" + path + "' line " + std::to_string(lineno) + ": not a number near '" +
                                 std::string(p).substr(0, 12) + "'");
      ds.data.push_back(v);
      p = end;
      ++count;
    }
    if (ds.n == 0) ds.dim = count;
    else if (count != ds.dim)
      throw std::runtime_error("'" + path + "' line " + std::to_string(lineno) + ": expected " +
                               std::to_string(ds.dim) + " values, found " + std::to_string(count));
    ++ds.n;
  }
  if (ds.n == 0) throw std::runtime_error("'" + path + "': no vectors found");
  return ds;
}

}  // namespace

void save_binary(const Dataset& ds, const std::string& path) {
  if (!is_little_endian()) throw std::runtime_error("binary format requires a little-endian host");
  ds.validate();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  const std::uint64_t n = ds.n, dim = ds.dim;
  out.write(kMagic, 8);
  out.write(reinterpret_cast<const char*>(&n), 8);
  out.write(reinterpret_cast<const char*>(&dim), 8);
  out.write(reinterpret_cast<const char*>(ds.data.data()),
            static_cast<std::streamsize>(ds.data.size() * sizeof(float)));
  if (!out) throw std::runtime_error("write error on '" + path + "'");
}

void save_csv(const Dataset& ds, const std::string& path) {
  ds.validate();
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  out.precision(9);  // enough digits to round-trip float32
  for (std::size_t i = 0; i < ds.n; ++i) {
    for (std::size_t j = 0; j < ds.dim; ++j) out << (j ? "," : "") << ds.row(i)[j];
    out << "\n";
  }
  if (!out) throw std::runtime_error("write error on '" + path + "'");
}

Dataset load_dataset(const std::string& path) {
  Dataset ds = ends_with(path, ".csv") ? load_csv(path) : load_binary(path);
  ds.validate();
  ds.compute_norms();
  return ds;
}

}  // namespace simdsearch
