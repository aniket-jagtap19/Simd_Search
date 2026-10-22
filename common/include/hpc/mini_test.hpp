#pragma once
// Minimal dependency-free unit-test harness (no GoogleTest needed).
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace hpc_test {

struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };

inline int run_all() {
  int failed_cases = 0;
  for (auto& c : registry()) {
    int before = failures();
    try { c.fn(); }
    catch (const std::exception& e) { ++failures(); std::printf("    exception: %s\n", e.what()); }
    catch (...) { ++failures(); std::printf("    unknown exception\n"); }
    bool ok = failures() == before;
    if (!ok) ++failed_cases;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", c.name);
  }
  std::printf("%zu tests, %d failed\n", registry().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace hpc_test

#define TEST(name)                                                   \
  static void name();                                                \
  static ::hpc_test::Reg reg_##name(#name, name);                    \
  static void name()

#define CHECK(cond)                                                                    \
  do { if (!(cond)) { ++::hpc_test::failures();                                        \
       std::printf("    %s:%d CHECK failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
  do { double a_ = (a), b_ = (b);                                                      \
       if (!(std::fabs(a_ - b_) <= (tol))) { ++::hpc_test::failures();                 \
         std::printf("    %s:%d CHECK_NEAR failed: %s=%.17g vs %s=%.17g\n", __FILE__,  \
                     __LINE__, #a, a_, #b, b_); } } while (0)

#define CHECK_THROWS(expr)                                                             \
  do { bool thrown_ = false;                                                           \
       try { (void)(expr); } catch (const std::exception&) { thrown_ = true; }         \
       if (!thrown_) { ++::hpc_test::failures();                                       \
         std::printf("    %s:%d expected exception: %s\n", __FILE__, __LINE__, #expr); } } while (0)
