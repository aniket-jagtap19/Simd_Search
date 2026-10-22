#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

#include "hpc/mini_test.hpp"
#include "simdsearch/dataset.hpp"
#include "simdsearch/io.hpp"
#include "simdsearch/search.hpp"

using namespace simdsearch;

static const double kTol = 1e-9;

static void expect_same(const std::vector<Hit>& a, const std::vector<Hit>& b) {
  MatchReport r = compare_hits(a, b, kTol);
  if (!r.ok) std::printf("    mismatch: %s\n", r.why.c_str());
  CHECK(r.ok);
}

TEST(generator_is_reproducible) {
  Dataset a = generate_dataset(100, 16, 7), b = generate_dataset(100, 16, 7), c = generate_dataset(100, 16, 8);
  CHECK(a.data == b.data);
  CHECK(a.data != c.data);
  CHECK(generate_query(16, 1) == generate_query(16, 1));
  CHECK(generate_query(16, 1) != generate_query(16, 2));
  // values in [-1,1)
  for (float v : a.data) CHECK(v >= -1.0f && v < 1.0f);
  // Row i does not depend on how many rows exist.
  Dataset small = generate_dataset(10, 16, 7);
  CHECK(std::equal(small.data.begin(), small.data.end(), a.data.begin()));
}

TEST(serial_matches_reference_both_metrics) {
  Dataset ds = generate_dataset(2000, 24, 11);
  auto q = generate_query(24, 5);
  for (Metric m : {Metric::Cosine, Metric::Euclidean})
    for (std::size_t k : {1u, 10u, 100u}) expect_same(search_serial(ds, q, k, m), search_reference(ds, q, k, m));
}

TEST(openmp_matches_serial_for_many_thread_counts) {
  Dataset ds = generate_dataset(3001, 33, 3);  // odd sizes: uneven static chunks
  auto q = generate_query(33, 9);
  for (Metric m : {Metric::Cosine, Metric::Euclidean})
    for (int T : {1, 2, 3, 4, 7, 8, 16}) {
      int used = 0;
      auto s = search_serial(ds, q, 25, m);
      auto p = search_openmp(ds, q, 25, m, T, &used);
      CHECK(used == T);
      expect_same(p, s);
    }
}

TEST(ordering_cosine_descending_euclid_ascending) {
  Dataset ds = generate_dataset(500, 8, 1);
  auto q = generate_query(8, 2);
  auto c = search_openmp(ds, q, 20, Metric::Cosine, 3);
  auto e = search_openmp(ds, q, 20, Metric::Euclidean, 3);
  for (std::size_t i = 1; i < c.size(); ++i) CHECK(c[i - 1].score >= c[i].score);
  for (std::size_t i = 1; i < e.size(); ++i) CHECK(e[i - 1].score <= e[i].score);
  for (const auto& h : c) CHECK(h.score >= -1.0 && h.score <= 1.0);
  for (const auto& h : e) CHECK(h.score >= 0.0);
}

TEST(known_small_example) {
  Dataset ds;
  ds.n = 4; ds.dim = 2;
  ds.data = {1, 0,   0, 1,   -1, 0,   2, 0};
  ds.compute_norms();
  std::vector<float> q = {1, 0};
  auto c = search_serial(ds, q, 4, Metric::Cosine);
  // ids 0 and 3 are both cosine 1.0 -> tie broken by smaller id; then id 1 (0), then id 2 (-1)
  CHECK(c[0].id == 0 && c[1].id == 3 && c[2].id == 1 && c[3].id == 2);
  CHECK_NEAR(c[0].score, 1.0, 1e-12);
  CHECK_NEAR(c[3].score, -1.0, 1e-12);
  auto e = search_serial(ds, q, 4, Metric::Euclidean);
  CHECK(e[0].id == 0 && e[1].id == 3 && e[2].id == 1 && e[3].id == 2);
  CHECK_NEAR(e[0].score, 0.0, 1e-12);
  CHECK_NEAR(e[1].score, 1.0, 1e-12);
  CHECK_NEAR(e[2].score, std::sqrt(2.0), 1e-12);
  CHECK_NEAR(e[3].score, 2.0, 1e-12);
}

TEST(deterministic_tie_breaking_with_duplicates) {
  // 200 identical vectors: every score ties; result must be the K smallest ids for any thread count.
  Dataset ds;
  ds.n = 200; ds.dim = 4;
  ds.data.assign(ds.n * ds.dim, 0.5f);
  ds.compute_norms();
  std::vector<float> q = {0.1f, 0.2f, 0.3f, 0.4f};
  for (Metric m : {Metric::Cosine, Metric::Euclidean})
    for (int T : {1, 2, 3, 5, 8}) {
      auto h = search_openmp(ds, q, 10, m, T);
      CHECK(h.size() == 10);
      for (std::size_t i = 0; i < h.size(); ++i) CHECK(h[i].id == i);
    }
}

TEST(zero_norm_vectors) {
  Dataset ds;
  ds.n = 3; ds.dim = 3;
  ds.data = {0, 0, 0,   1, 1, 0,   -1, -1, 0};
  ds.compute_norms();
  CHECK(ds.norms[0] == 0.0);
  // Non-zero query: zero vector scores 0 (finite, no NaN); ranking is 1.0, 0.0, -1.0.
  auto c = search_serial(ds, {1, 1, 0}, 3, Metric::Cosine);
  CHECK(c[0].id == 1 && c[1].id == 0 && c[2].id == 2);
  CHECK_NEAR(c[1].score, 0.0, 0.0);
  for (const auto& h : c) CHECK(std::isfinite(h.score));
  // Zero query: every cosine similarity is defined as 0 -> ids in ascending order.
  auto z = search_openmp(ds, {0, 0, 0}, 3, Metric::Cosine, 2);
  CHECK(z[0].id == 0 && z[1].id == 1 && z[2].id == 2);
  for (const auto& h : z) CHECK(h.score == 0.0);
  // Euclidean with a zero vector is perfectly well defined.
  auto e = search_serial(ds, {0, 0, 0}, 1, Metric::Euclidean);
  CHECK(e[0].id == 0 && e[0].score == 0.0);
}

TEST(k_edge_cases) {
  Dataset ds = generate_dataset(50, 6, 4);
  auto q = generate_query(6, 4);
  CHECK(search_serial(ds, q, 1000, Metric::Cosine).size() == 50);  // clamped to n
  CHECK(search_openmp(ds, q, 50, Metric::Euclidean, 8).size() == 50);
  expect_same(search_openmp(ds, q, 1, Metric::Cosine, 4), search_reference(ds, q, 1, Metric::Cosine));
  // more threads than vectors
  Dataset tiny = generate_dataset(3, 6, 4);
  expect_same(search_openmp(tiny, q, 3, Metric::Euclidean, 16), search_reference(tiny, q, 3, Metric::Euclidean));
  Dataset one = generate_dataset(1, 6, 4);
  CHECK(search_openmp(one, q, 5, Metric::Cosine, 4).size() == 1);
}

TEST(input_validation) {
  Dataset ds = generate_dataset(20, 5, 1);
  auto q = generate_query(5, 1);
  CHECK_THROWS(search_serial(ds, q, 0, Metric::Cosine));
  CHECK_THROWS(search_openmp(ds, q, 0, Metric::Cosine, 2));
  CHECK_THROWS(search_serial(ds, generate_query(4, 1), 3, Metric::Cosine));  // wrong dimension
  auto bad = q;
  bad[2] = std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS(search_serial(ds, bad, 3, Metric::Euclidean));
  bad[2] = std::numeric_limits<float>::infinity();
  CHECK_THROWS(search_openmp(ds, bad, 3, Metric::Cosine, 2));
  Dataset empty;
  CHECK_THROWS(search_serial(empty, q, 3, Metric::Cosine));
  Dataset no_norms = ds;
  no_norms.norms.clear();
  CHECK_THROWS(search_serial(no_norms, q, 3, Metric::Cosine));
  CHECK_THROWS(generate_dataset(0, 4, 1));
  CHECK_THROWS(generate_dataset(4, 0, 1));
  CHECK_THROWS(parse_metric("manhattan"));
  Dataset nan_ds = ds;
  nan_ds.data[7] = std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS(nan_ds.validate());
}

TEST(file_round_trip_binary_and_csv) {
  Dataset ds = generate_dataset(37, 9, 21);
  save_binary(ds, "test_tmp.bin");
  Dataset b = load_dataset("test_tmp.bin");
  CHECK(b.n == ds.n && b.dim == ds.dim && b.data == ds.data);
  CHECK(b.norms.size() == b.n);
  save_csv(ds, "test_tmp.csv");
  Dataset c = load_dataset("test_tmp.csv");
  CHECK(c.n == ds.n && c.dim == ds.dim && c.data == ds.data);  // 9 digits round-trips float32
  auto q = generate_query(9, 3);
  expect_same(search_serial(b, q, 5, Metric::Cosine), search_serial(ds, q, 5, Metric::Cosine));
  std::remove("test_tmp.bin");
  std::remove("test_tmp.csv");
}

TEST(malformed_files_are_rejected) {
  { std::ofstream f("bad1.bin", std::ios::binary); f << "not a dataset at all, definitely"; }
  CHECK_THROWS(load_dataset("bad1.bin"));
  Dataset ds = generate_dataset(10, 4, 1);
  save_binary(ds, "bad2.bin");
  { std::ifstream in("bad2.bin", std::ios::binary); std::string all((std::istreambuf_iterator<char>(in)), {});
    all.resize(all.size() - 5);
    std::ofstream out("bad3.bin", std::ios::binary); out << all; }
  CHECK_THROWS(load_dataset("bad3.bin"));  // truncated
  { std::ofstream f("bad4.csv"); f << "1,2,3\n4,5\n"; }
  CHECK_THROWS(load_dataset("bad4.csv"));  // ragged
  { std::ofstream f("bad5.csv"); f << "1,2,abc\n"; }
  CHECK_THROWS(load_dataset("bad5.csv"));
  { std::ofstream f("bad6.csv"); f << "1,2,nan\n"; }
  CHECK_THROWS(load_dataset("bad6.csv"));  // non-finite
  CHECK_THROWS(load_dataset("does_not_exist.bin"));
  for (const char* p : {"bad1.bin", "bad2.bin", "bad3.bin", "bad4.csv", "bad5.csv", "bad6.csv"}) std::remove(p);
}

int main() { return hpc_test::run_all(); }
