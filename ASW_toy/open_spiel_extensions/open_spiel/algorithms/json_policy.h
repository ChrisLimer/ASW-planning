#ifndef OPEN_SPIEL_ALGORITHMS_JSON_POLICY_H_
#define OPEN_SPIEL_ALGORITHMS_JSON_POLICY_H_

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

#include <fstream>

#include <sstream>
#include <iostream>
#include <tuple>
#include <regex>

#include <nlohmann/json.hpp>
// Alias for convenience
using json = nlohmann::json;


namespace open_spiel {
namespace algorithms {

// A type for tables holding 
using PolicyTable =
    std::unordered_map<std::string, ActionsAndProbs>;

// using PolicyTableAndReachProb =
//     std::unordered_map<std::string, std::tuple<ActionsAndProbs, std::vector<double>, std::vector<double>>>;


// using PolicyTableAndReachProb =
//     std::unordered_map<std::string, std::tuple<ActionsAndProbs, std::unordered_map<std::string, std::tuple<std::vector<double>, std::vector<double>>> >>;

using InnerMap = std::unordered_map<std::string, std::tuple<std::vector<double>, std::vector<double>>>;
using PolicyTableAndReachProb =std::unordered_map<std::string, 
    std::tuple< ActionsAndProbs, std::unordered_map<std::string, std::tuple<std::vector<double>, std::vector<double>>> >
    >;

// using PolicyTableAndReachProb =std::unordered_map<std::string, 
//     std::tuple< std::vector<double>, std::vector<double>, ActionsAndProbs, std::unordered_map<std::string, std::tuple<std::vector<double>, std::vector<double>>> >
//     >;

// A policy that extracts the current policy from the CFR table values.
class JsonPolicy : public Policy {

 public:
  JsonPolicy(const Game& game);
 
  ActionsAndProbs GetStatePolicy(const State& state) const override {
    return GetStatePolicy(state, state.CurrentPlayer());
  };
  ActionsAndProbs GetStatePolicy(const State& state, Player player) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

  void AddInfoStatePolicy(std::string info_state, ActionsAndProbs policy);

  PolicyTable& GetTabularPolicy();
  void PrintPolicy(std::string string_to_include="") const;
  void savePolicyToJson(const std::string& output_path) const;
  json savePolicyToJsonAndReturn(const std::string& output_path) const;

  void LoadPolicyFromJson(const std::string& json_path);


  PolicyTableAndReachProb ComputePolicyRewardAndReachProb();
  std::vector<double> ComputePolicyRewardAndReachProbRecursive(PolicyTableAndReachProb &prr_map, const State& state, std::vector<double> reach_prob) const;

  void savePolicyTableAndReachProbToJson_(const PolicyTableAndReachProb& table, const std::string& output_path);
  void savePolicyTableAndReachProbToJson(const std::string& output_path);

 private:
  std::shared_ptr<const Game> game_;
  PolicyTable info_states_;
  PolicyTableAndReachProb info_state_datas_;
};



}   // namespace algorithms
}   // namespace open_spiel

#endif // OPEN_SPIEL_ALGORITHMS_JSON_POLICY_H_