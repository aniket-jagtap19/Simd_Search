#include <cmath>
#include <vector>

#include "hpc/mini_test.hpp"
#include "matvec/matvec.hpp"

using namespace matvec;

TEST(known_values_2x3) {
  const double A[6] = {1, 2, 3, 4, 5, 6};
  const double x[3] = {1, 0, -1};
  double y[2];
  multiply_serial(A, x, y, 2, 3);
  CHECK(y[0] == -2.0 && y[1] == -2.0);
  double yo[2];
  multiply_openmp(A, x, yo, 2, 3, 4);
  CHECK(yo[0] == -2.0 && yo[1] == -2.0);
}

TEST(openmp_equals_serial_all_shapes_and_threads) {
  const std::size_t shapes[][2] = {{1, 1}, {1, 50}, {50, 1}, {7, 13}, {13, 7}, {100, 100}, {257, 31}, {256, 256}};
  for (auto& s : shapes) {
    Problem p = make_problem(s[0], s[1], 5);
    std::vector<double> ys(s[0]), yp(s[0]);
    multiply_serial(p.A.data(), p.x.data(), ys.data(), s[0], s[1]);
    for (int T : {1, 2, 3, 4, 8, 64}) {  // includes more threads than rows
      std::fill(yp.begin(), yp.end(), -999.0);
      int used = 0;
      multiply_openmp(p.A.data(), p.x.data(), yp.data(), s[0], s[1], T, &used);
      CHECK(used == T);
      CHECK(max_relative_error(yp.data(), ys.data(), s[0]) <= kTolerance);
    }
  }
}

TEST(serial_matches_long_double_oracle) {
  Problem p = make_problem(64, 200, 9);
  std::vector<double> y(64);
  multiply_serial(p.A.data(), p.x.data(), y.data(), 64, 200);
  for (std::size_t i = 0; i < 64; ++i) {
    long double s = 0;
    for (std::size_t j = 0; j < 200; ++j) s += (long double)p.A[i * 200 + j] * p.x[j];
    CHECK_NEAR(y[i], (double)s, 1e-12);
  }
}

TEST(generator_reproducible) {
  Problem a = make_problem(10, 10, 3), b = make_problem(10, 10, 3), c = make_problem(10, 10, 4);
  CHECK(a.A == b.A && a.x == b.x);
  CHECK(a.A != c.A);
}

TEST(validation) {
  double a = 1, x = 1, y = 0;
  CHECK_THROWS(make_problem(0, 5, 1));
  CHECK_THROWS(make_problem(5, 0, 1));
  CHECK_THROWS(validate_dims(~std::size_t(0) / 2, 4));
  CHECK_THROWS(multiply_serial(nullptr, &x, &y, 1, 1));
  CHECK_THROWS(multiply_openmp(&a, nullptr, &y, 1, 1, 2));
  CHECK_THROWS(multiply_serial(&a, &x, &y, 1, 0));
  multiply_serial(nullptr, nullptr, nullptr, 0, 5);  // zero rows is a harmless no-op
  const double nan_v[2] = {std::nan(""), 1.0}, ok_v[2] = {0.0, 1.0};
  CHECK(std::isinf(max_relative_error(nan_v, ok_v, 2)));
}

int main() { return hpc_test::run_all(); }
