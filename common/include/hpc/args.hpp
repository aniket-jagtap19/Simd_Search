#pragma once
// Tiny command-line parser: --key value | --key=value | --flag, plus
// --config file (lines "key = value"; command-line values take precedence).
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpc {

class Args {
 public:
  Args(int argc, char** argv, int first = 1) {
    for (int i = first; i < argc; ++i) {
      std::string t = argv[i];
      if (t.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + t + "'");
      t = t.substr(2);
      auto eq = t.find('=');
      if (eq != std::string::npos) {
        kv_[t.substr(0, eq)] = t.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[t] = argv[++i];
      } else {
        kv_[t] = "true";
      }
    }
    if (has("config")) load_config(get("config", ""));
  }

  bool has(const std::string& k) const { return kv_.count(k) != 0; }
  bool flag(const std::string& k) const {
    auto it = kv_.find(k);
    return it != kv_.end() && it->second != "false" && it->second != "0";
  }
  std::string get(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  long long get_int(const std::string& k, long long def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : parse_int(k, it->second);
  }
  double get_double(const std::string& k, double def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    try {
      std::size_t pos = 0;
      double v = std::stod(it->second, &pos);
      if (pos != it->second.size()) throw std::invalid_argument("");
      return v;
    } catch (...) {
      throw std::invalid_argument("--" + k + ": '" + it->second + "' is not a number");
    }
  }
  std::vector<long long> get_int_list(const std::string& k, const std::vector<long long>& def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    std::vector<long long> out;
    for (const auto& tok : split(it->second)) out.push_back(parse_int(k, tok));
    if (out.empty()) throw std::invalid_argument("--" + k + ": empty list");
    return out;
  }
  std::vector<std::string> get_list(const std::string& k, const std::vector<std::string>& def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    auto out = split(it->second);
    if (out.empty()) throw std::invalid_argument("--" + k + ": empty list");
    return out;
  }
  // Throws if the user passed an option this program does not understand.
  void require_known(std::initializer_list<const char*> known) const {
    for (const auto& kv : kv_) {
      bool ok = (kv.first == "config");
      for (const char* k : known) ok = ok || kv.first == k;
      if (!ok) throw std::invalid_argument("unknown option --" + kv.first);
    }
  }

 private:
  std::map<std::string, std::string> kv_;

  static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
      while (!tok.empty() && tok.back() == ' ') tok.pop_back();
      if (!tok.empty()) out.push_back(tok);
    }
    return out;
  }
  static long long parse_int(const std::string& k, const std::string& s) {
    try {
      std::size_t pos = 0;
      long long v = std::stoll(s, &pos);
      if (pos != s.size()) throw std::invalid_argument("");
      return v;
    } catch (...) {
      throw std::invalid_argument("--" + k + ": '" + s + "' is not an integer");
    }
  }
  void load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot open config file '" + path + "'");
    std::string line;
    while (std::getline(in, line)) {
      auto h = line.find('#');
      if (h != std::string::npos) line.erase(h);
      auto eq = line.find('=');
      if (eq == std::string::npos) continue;
      auto trim = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
        return s;
      };
      std::string key = trim(line.substr(0, eq)), val = trim(line.substr(eq + 1));
      if (key.rfind("--", 0) == 0) key = key.substr(2);
      if (!key.empty() && !kv_.count(key)) kv_[key] = val;
    }
  }
};

}  // namespace hpc
