#pragma once
// Portable, platform-independent pseudo-random generator (SplitMix64).
// std::normal_distribution / std::uniform_real_distribution are implementation
// defined, so the same seed could produce different data on different standard
// libraries. These helpers give bit-identical data everywhere.
#include <cstdint>

namespace hpc {

inline std::uint64_t splitmix64_next(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline std::uint64_t mix64(std::uint64_t x) { return splitmix64_next(x); }

// Uniform in [-1, 1), exactly representable (24 random bits).
inline float uniform_pm1_f(std::uint64_t& s) {
  return static_cast<float>(splitmix64_next(s) >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

// Uniform in [-1, 1) with 53 random bits.
inline double uniform_pm1_d(std::uint64_t& s) {
  return static_cast<double>(splitmix64_next(s) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

}  // namespace hpc
