
#include "open_spiel/algorithms/json_policy.h"

#include <algorithm>
#include <array>
#include <random>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/charconv.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serialization.h"

#include <nlohmann/json.hpp>
// Alias for convenience
using json = nlohmann::json;

namespace open_spiel {
namespace algorithms {




JsonPolicy::JsonPolicy(const Game& game): game_(game.shared_from_this())
{

}



void JsonPolicy::LoadPolicyFromJson(const std::string& json_path)
{    
    std::ifstream file_in(json_path);
    if (!file_in.is_open()) {
        throw std::runtime_error("Could not open file: " + json_path);
    }

    json j;
    file_in >> j;

    info_states_.clear();  // Clear existing data if any

    for (auto& [key, policy_json] : j.items()) {
        ActionsAndProbs policy;
        for (auto& [action_str, prob_json] : policy_json.items()) {
            int action_id = std::stoi(action_str);
            double prob = prob_json.get<double>();
            policy.emplace_back(action_id, prob);
        }
        info_states_[key] = policy;
    }
}

json JsonPolicy::savePolicyToJsonAndReturn(const std::string& output_path) const
{
    json j;

    for (const auto& [key, policy] : info_states_) 
    {
        json policy_json;
        for (const auto& [action_id, prob] : policy) 
        {
            policy_json[std::to_string(action_id)] = prob;
        }
        j[key] = policy_json;
    }

    // Save to file
    std::ofstream file_out(output_path);
    file_out << j.dump(2);  // Pretty-print with 2-space indentation
    file_out.close();

    return j;
}


void JsonPolicy::savePolicyToJson(const std::string& output_path) const
{
    json j = savePolicyToJsonAndReturn(output_path);
}



void JsonPolicy::savePolicyTableAndReachProbToJson(const std::string& output_path)
{
    savePolicyTableAndReachProbToJson_(info_state_datas_, output_path);
}

void JsonPolicy::savePolicyTableAndReachProbToJson_(const PolicyTableAndReachProb& table, const std::string& output_path) {
    json j;

    for (const auto& [info_state, data_tuple] : table) {
        const ActionsAndProbs& actions_and_probs = std::get<0>(data_tuple);
        const InnerMap& inner_map = std::get<1>(data_tuple);

        json entry;

        // Store ActionsAndProbs
        json actions_json = json::array();
        for (const auto& [action, prob] : actions_and_probs) {
            actions_json.push_back({ {"action", action}, {"probability", prob} });
        }
        entry["actions_and_probs"] = actions_json;

        // Store InnerMap
        json inner_json;
        for (const auto& [subkey, subtuple] : inner_map) {
            const std::vector<double>& reward = std::get<0>(subtuple);
            const std::vector<double>& reach = std::get<1>(subtuple);

            inner_json[subkey] = {
                {"reward", reward},
                {"reach_prob", reach}
            };
        }
        entry["sub_info"] = inner_json;

        j[info_state] = entry;
    }

    // Save to file
    std::ofstream file_out(output_path);
    if (!file_out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    file_out << j.dump(2);  // Pretty-print with 2-space indentation
    file_out.close();
}




ActionsAndProbs JsonPolicy::GetStatePolicy(const State& state, Player player) const {
  ActionsAndProbs actions_and_probs = GetStatePolicy(state.InformationStateString(player));
  return actions_and_probs;
}


ActionsAndProbs JsonPolicy::GetStatePolicy(const std::string& info_state) const
{
    auto it = info_states_.find(info_state);
    if (it != info_states_.end()) {
        ActionsAndProbs retrieved_policy = it->second;
        // Use retrieved_policy...
        return retrieved_policy;
    } else {
        std::cout << "Info state not found.\n";
    }

    return ActionsAndProbs();
}

PolicyTable& JsonPolicy::GetTabularPolicy()
{
    return info_states_;
}

void JsonPolicy::PrintPolicy(std::string string_to_include) const
{
    for (const auto& [key, policy] : info_states_) 
    {
        if (key.find(string_to_include) != std::string::npos)
        {
            std::cout << " " << key << "  Policy: ";
            for (int i=0; i<policy.size(); i++)
            {
                std::cout << " (" << policy[i].first << ", " << policy[i].second << ")";
            }
            std::cout << std::endl;
        }

    }
}

void JsonPolicy::AddInfoStatePolicy(std::string info_state, ActionsAndProbs policy)
{
    info_states_.insert({info_state, policy});
}




PolicyTableAndReachProb JsonPolicy::ComputePolicyRewardAndReachProb()
{
    PolicyTableAndReachProb prr_map;

    std::vector<double> reach_prob = {1.0, 1.0, 1.0};
    const std::unique_ptr<State> root_state_ = game_->NewInitialState();
    std::vector<double> returns = ComputePolicyRewardAndReachProbRecursive(prr_map, *root_state_, reach_prob);
    info_state_datas_ = prr_map;
    return info_state_datas_;
}


std::vector<double> JsonPolicy::ComputePolicyRewardAndReachProbRecursive(PolicyTableAndReachProb &prr_map, const State& state, std::vector<double> reach_prob) const
{
    // std::cout << "ComputePolicyRewardAndReachProbRecursive: " << state.ToString() << std::endl;
    if (state.IsTerminal()) {
        std::vector<double> returns = state.Returns();
        // std::cout << "IsTerminal: " << returns[0] << std::endl;
        return state.Returns();
    }

    

    if (state.IsChanceNode()) {
        // std::cout << "IsChanceNode" <<std::endl;
        std::vector<double> returns = state.Returns();
        for (int i=0; i<returns.size(); i++)
            returns[i] = 0.0;

        ActionsAndProbs actions_and_probs = state.ChanceOutcomes();
        std::vector<double> dist(actions_and_probs.size(), 0);
        std::vector<Action> outcomes(actions_and_probs.size(), 0);
        for (int oidx = 0; oidx < actions_and_probs.size(); ++oidx) {

            Action action = actions_and_probs[oidx].first;
            double action_prob = actions_and_probs[oidx].second;
            const std::unique_ptr<State> new_state = state.Child(action);

            std::vector<double> new_reach_prob = reach_prob;
            new_reach_prob[2] *= action_prob;


            std::vector<double> return_i = ComputePolicyRewardAndReachProbRecursive(prr_map, *new_state, new_reach_prob);
            for (int i=0; i<return_i.size(); i++)
                returns[i] += return_i[i] * action_prob;
        }
        return returns;

    }
    else
    {

        std::vector<Action> legal_actions = state.LegalActions();
        std::string info_state_string = state.InformationStateString(state.CurrentPlayer());

        const ActionsAndProbs& actions_and_probs = info_states_.at(info_state_string);

        std::vector<double> returns = state.Returns();
        for (int i=0; i<returns.size(); i++)
            returns[i] = 0.0;

        for (int i=0; i<actions_and_probs.size(); i++)
        {
            Action action = actions_and_probs[i].first;
            double action_prob = actions_and_probs[i].second;
            const std::unique_ptr<State> new_state = state.Child(action);

            std::vector<double> new_reach_prob = reach_prob;
            new_reach_prob[state.CurrentPlayer()] *= action_prob;

            std::vector<double> return_i = ComputePolicyRewardAndReachProbRecursive(prr_map, *new_state, new_reach_prob);
            for (int i=0; i<return_i.size(); i++)
                returns[i] += return_i[i] * action_prob;
        }



        // std::string info_state_string = state.InformationStateString();
        std::string state_string = state.ToString();
        auto it = prr_map.find(info_state_string);
        if (it != prr_map.end()) {

        }
        else
        {
            std::tuple< ActionsAndProbs, std::unordered_map<std::string, std::tuple<std::vector<double>, std::vector<double>>> > state_data;
            
            std::get<0>(state_data) = actions_and_probs;
            prr_map[info_state_string] = state_data;
        }

        std::tuple<std::vector<double>, std::vector<double>> state_data(returns, reach_prob);
        std::get<1>(prr_map[info_state_string])[state_string] = state_data;                     // insert public-state to information state 

        return returns;

    }
    
}


}   // namespace algorithms
}   // namespace open_spiel