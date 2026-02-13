// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ASW_SMALL_H_
#define OPEN_SPIEL_GAMES_ASW_SMALL_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"


namespace open_spiel {
namespace asw_small {

// inline constexpr const int kNumInfoStatesP0 = 6;
// inline constexpr const int kNumInfoStatesP1 = 6;

class AswSmallGame;

class AswSmallState : public State {
 public:
  explicit AswSmallState(std::shared_ptr<const Game> game);
  AswSmallState(const AswSmallState&) = default;

  Player CurrentPlayer() const override;

  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;



 protected:
  void DoApplyAction(Action move) override;

 private:
  int max_turns = 6;

  uint turn=0;
  // int sub_pos = -1;
  int sub_pos = 0;
  int sv_pos = -2;
  bool detected_sub = false;

  std::vector<int> sub_history = {};
  std::vector<int> sv_history = {};
  std::array<bool, 13> sub_bool_history{{false}};
  std::array<bool, 13> sv_bool_history{{false}};
  // std::array<std::vector<Action>, 9> available_sub_destinations = {{
  //   {2,3}, {3,4},
  //   {5,6}, {6,7}, {7,8},
  //   {9}, {9,10}, {10,11}, {11}
  // }};
  // std::array<std::vector<Action>, 3> available_sv_dips = {{
  //   {2,3,4}, {5,6,7,8}, {9,10,11}
  // }};

  std::array<std::vector<Action>, 10> available_sub_destinations = {{
    {1,2},
    {3,4}, {4,5},
    {6,7}, {7,8}, {8,9},
    {10}, {10,11}, {11,12}, {12}
  }};
  std::array<std::vector<Action>, 3> available_sv_dips = {{
    {3,4,5}, {6,7,8,9}, {10,11,12}
  }};

  int current_player_ = kChancePlayerId;

};

class AswSmallGame : public Game {
 public:
  explicit AswSmallGame(const GameParameters& params);
  int NumDistinctActions() const override { return 10; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 3; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return 10; }
  int MaxChanceNodesInHistory() const override { return num_players_; }

 private:
  // Number of players.
  int num_players_=2;
};

}  // namespace asw_small
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ASW_SMALL_H_
