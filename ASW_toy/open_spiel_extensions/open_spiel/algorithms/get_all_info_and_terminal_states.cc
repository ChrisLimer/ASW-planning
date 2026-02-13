// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #include "open_spiel/algorithms/get_all_infostates.h"
#include "open_spiel/algorithms/get_all_info_and_terminal_states.h"

#include <algorithm>

namespace open_spiel {
namespace algorithms {
// namespace {



// Get all the information states. Note that there might contain duplicates.
void GetSubgameInformationStatesAndTerminalStates(
    State* state, std::vector<std::vector<std::string>>* all_info_states,
    std::vector<std::pair<std::string, std::vector<double>>>* all_terminal_tates,
    int depth_limit, int depth) {

  if (state->IsTerminal()) {
    // std::cout << "Terminal State: \n" << state->ToString() << std::endl;
    // std::cout << state->ToString() << std::endl;

    auto ret = state->Returns();
    (*all_terminal_tates).push_back({state->ToString(), ret});
    return;
  }

  if (depth_limit >= 0 && depth > depth_limit) {
    return;
  }


  for (auto action : state->LegalActions()) {
    auto next_state = state->Clone();
    next_state->ApplyAction(action);

    if (!state->IsChanceNode()) {
      int player = state->CurrentPlayer();
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, state->NumPlayers());
      // std::cout << "Information string: " << state->InformationStateString() << std::endl;
      (*all_info_states)[player].push_back(state->InformationStateString());
    }
    // else
    // {
    //   std::cout << "player: " << state->CurrentPlayer() << std::endl;
    // }

    GetSubgameInformationStatesAndTerminalStates(next_state.get(), all_info_states, all_terminal_tates, depth_limit,
                                depth + 1);
  }
}



std::pair<std::vector<std::vector<std::string>>, 
    std::vector<std::pair<std::string, std::vector<double>>>> GetAllInformationStatesAndTerminalStates(const Game& game,
  int depth_limit) {
  // std::cout << "In 'GetAllInformationStates'" << std::endl;
  // Get the root state.
  std::unique_ptr<State> state = game.NewInitialState();
  std::vector<std::vector<std::string>> all_infostates(game.NumPlayers());

  std::vector<std::pair<std::string, std::vector<double>>> all_terminaltates;


  // Then, do a recursive tree walk to fill up the vector.
  // std::cout << "Call 'GetSubgameInformationStates'" << std::endl;
  GetSubgameInformationStatesAndTerminalStates(state.get(), &all_infostates, &all_terminaltates, depth_limit, 0);

  // Remove duplicates by sorting the info states and calling std::unique.
  for (Player p = 0; p < all_infostates.size(); ++p) {
  absl::c_sort(all_infostates[p]);
  auto last = std::unique(all_infostates[p].begin(), all_infostates[p].end());
  all_infostates[p].erase(last, all_infostates[p].end());
  }

  return {all_infostates, all_terminaltates};
}


}  // namespace algorithms
}  // namespace open_spiel
