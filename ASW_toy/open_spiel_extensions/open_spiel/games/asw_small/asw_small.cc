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

#include "open_spiel/games/asw_small/asw_small.h"
// #include "ASW_toy/open_spiel_extensions/open_spiel/games/asw_small/asw_small.h"
// #include "asw_small.h"


#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/utils/tensor_view.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace asw_small {
namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"asw_small",
                         /*long_name=*/"ASW Small",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {},
                        //  /*default_loadable=*/true,
                        //  /*provides_factored_observation_string=*/true,
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AswSmallGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace



AswSmallState::AswSmallState(std::shared_ptr<const Game> game)
    : State(game) {}

int AswSmallState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void AswSmallState::DoApplyAction(Action move) {
  
  history_.push_back({CurrentPlayer(), move});
  if (CurrentPlayer()==0)
  {
    sub_pos = move;
    sub_history.push_back(sub_pos);
  }
  else if (CurrentPlayer()==1)
  {
    sv_pos = move;
    detected_sub = (sub_pos==sv_pos);
    sv_history.push_back(sv_pos);
  }
  else if (CurrentPlayer()==kChancePlayerId)
  {
    current_player_=0;
    sub_pos = move;
    sub_history.push_back(sub_pos);
    turn++;
    return;
  }
  else
  {
    std::cout << "\nError 'DoApplyAction'. CurrentPlayer(): " << CurrentPlayer() << "\n" << std::endl;
    return;
  }

  turn++;
  // turn += current_player_;
  current_player_ = 1-current_player_;

  if (turn>max_turns)
  {
    current_player_ = kInvalidPlayer;
  }

}

std::vector<Action> AswSmallState::LegalActions() const {
  if (IsTerminal()) return {};

  std::vector<Action> actions;
  if (IsChanceNode()) {
    actions = {};
    std::vector<std::pair<Action, double>> outcomes = ChanceOutcomes();
    for (int i=0; i<outcomes.size(); i++)
    {
      actions.push_back(outcomes[i].first);
    }

  } else {
    if (CurrentPlayer()==0)
    {
      actions = available_sub_destinations[sub_pos];
    }
    else if (CurrentPlayer()==1)
    {
      actions = available_sv_dips[turn/2-1];
    }
  }

  return actions;
}

std::string AswSmallState::ActionToString(Player player, Action move) const {
  if (CurrentPlayer()==0)
  {
    return "sub move to " + std::to_string(move);
  }
  else if (CurrentPlayer()==1)
  {
    return "sv dip in " + std::to_string(move);
  }
  else if (CurrentPlayer()==kChancePlayerId)
  {
    return "sub alloc to " + std::to_string(move);
  }

}

std::string AswSmallState::ToString() const {
  // // The deal: space separated card per player
  // std::string str;
  // str += "turn: " + std::to_string(turn) + "  pl: " + std::to_string(CurrentPlayer()) + "\n";
  // for (int i=0; i<sub_history.size(); i++)
  // {
  //   str += " " + std::to_string(sub_history[i]);
  // }
  // str += "\n";
  // for (int i=0; i<sv_history.size(); i++)
  // {
  //   str += " " + std::to_string(sv_history[i]);
  // }
  // str += "\n" + std::to_string(int(detected_sub));

  // return str;


  std::string str = "ToString turn: " + std::to_string(turn) + "  pl: " + std::to_string(CurrentPlayer())+" ";

  str += "Sub:";
  if (CurrentPlayer()==0 || true)
  {
    // str += "Sub:";
    for (int i=0; i<sub_history.size(); i++)
    {
      str += " " + std::to_string(sub_history[i]);
    }
  }
  str += "  SV:";
  for (int i=0; i<sv_history.size(); i++)
  {
    str += " " + std::to_string(sv_history[i]);
  }
  str += "  Detected: " + std::to_string(int(detected_sub));
  return str;
  
}

bool AswSmallState::IsTerminal() const 
{
  return detected_sub || turn>max_turns;
}

std::vector<double> AswSmallState::Returns() const {
  if (!IsTerminal()) {
    // std::cout << "Not IsTerminal" << std::endl;
    return std::vector<double>(num_players_, 0.0);
  }
  else if (detected_sub)
  {
    // return std::vector<double>{-1.0, 1.0};
    return std::vector<double>{0.0, 0.0}; 
  }

  double ret = 0.0;
  if (sub_pos==10)
    ret = 1.0;
  else if (sub_pos==11)
    ret = 2.0;
  else if (sub_pos==12)
    ret = 1.0;

  // double ret = (double(sub_pos>=9 && sub_pos<=11)) * 2.0 - 1.0;
  return std::vector<double>{ret, -ret};
}

std::string AswSmallState::InformationStateString(Player player) const {

  //// Multiline state string
  // std::string str = "turn: " + std::to_string(turn) + "  pl: " + std::to_string(CurrentPlayer())+"\n";
  // if (player==0)
  // {
  //   str += "Sub: ";
  //   for (int i=0; i<sub_history.size(); i++)
  //   {
  //     str += " " + std::to_string(sub_history[i]);
  //   }
  //   str += "\n";
  // }
  // str += "SV: ";
  // for (int i=0; i<sv_history.size(); i++)
  // {
  //   str += " " + std::to_string(sv_history[i]);
  // }

  // str += "\nDetected: " + std::to_string(int(detected_sub));
  // return str;


  //// Oneline state string
  std::string str = "turn: " + std::to_string(turn) + "  pl: " + std::to_string(CurrentPlayer()) + " ";
  str += "Sub:";
  if (CurrentPlayer()==0)
  {
    // str += "Sub:";
    for (int i=0; i<sub_history.size(); i++)
    {
      str += " " + std::to_string(sub_history[i]);
    }
  }
  str += "  SV:";
  for (int i=0; i<sv_history.size(); i++)
  {
    str += " " + std::to_string(sv_history[i]);
  }
  str += "  Detected: " + std::to_string(int(detected_sub));
  return str;
  
}

std::string AswSmallState::ObservationString(Player player) const {

    //// Oneline state string
  std::string str = "ObservationString turn: " + std::to_string(turn) + "  pl: " + std::to_string(CurrentPlayer())+" "+ std::to_string(player) + " ";
  str += "Sub:";
  if (CurrentPlayer()==0)
  {
    // str += "Sub:";
    for (int i=0; i<sub_history.size(); i++)
    {
      str += " " + std::to_string(sub_history[i]);
    }
  }
  str += "  SV:";
  for (int i=0; i<sv_history.size(); i++)
  {
    str += " " + std::to_string(sv_history[i]);
  }
  str += "  Detected: " + std::to_string(int(detected_sub));
  return str;


  // std::string str = "pl " + std::to_string(current_player_)+"\n";
  // str += "turn: " + std::to_string(turn) + "\nSub: ";
  // if (player==0)
  // {
  //   for (int i=0; i<sub_history.size(); i++)
  //   {
  //     str += " " + std::to_string(sub_history[i]);
  //   }
  // }
  // str += "\nSV: ";
  // for (int i=0; i<sv_history.size(); i++)
  // {
  //   str += " " + std::to_string(sv_history[i]);
  // }

  // str += "\n" + std::to_string(int(detected_sub));
  // return str;
}

void AswSmallState::InformationStateTensor(Player player,
                                       absl::Span<float> values) const {
  // ContiguousAllocator allocator(values);
  // const AswSmallGame& game = open_spiel::down_cast<const AswSmallGame&>(*game_);
  // game.info_state_observer_->WriteTensor(*this, player, &allocator);

  TensorView<2> view(values, {13, 2}, true);
  

  for (int i=0; i<sub_history.size(); i++)
  {
    view[{0, sub_history[i]}] = 1.0;
  }
  for (int i=0; i<sv_history.size(); i++)
  {
    view[{0, sv_history[i]}] = 1.0;
  }
  
}

void AswSmallState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  // ContiguousAllocator allocator(values);
  // const AswSmallGame& game = open_spiel::down_cast<const AswSmallGame&>(*game_);
  // game.default_observer_->WriteTensor(*this, player, &allocator);
  
  InformationStateTensor(player, values);
}

std::unique_ptr<State> AswSmallState::Clone() const {
  return std::unique_ptr<State>(new AswSmallState(*this));
}


std::vector<std::pair<Action, double>> AswSmallState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;  // [action, p]

  return std::vector<std::pair<Action, double>>{{1, 0.5}, {2, 0.5}};

  return outcomes;
}


AswSmallGame::AswSmallGame(const GameParameters& params)
    : Game(kGameType, params) {
  // SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  // SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

}

std::unique_ptr<State> AswSmallGame::NewInitialState() const {
  return std::unique_ptr<State>(new AswSmallState(shared_from_this()));
}

std::vector<int> AswSmallGame::InformationStateTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by 2 (n - 1 + n) bits for betting sequence (longest sequence:
  // everyone except one player can pass and then everyone can bet/pass).
  // n + n + 1 + 2 (n-1 + n) = 6n - 1.
  return {2,13};
}

std::vector<int> AswSmallGame::ObservationTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by the contribution of each player to the pot (n).
  // n + n + 1 + n = 3n + 1.
  return {2,13};
}

double AswSmallGame::MaxUtility() const {
  return 1.0;
}

double AswSmallGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // In AswSmall, the most any one player can lose is the single chip they paid
  // to play and the single chip they paid to raise/call.
  return -1.0;
}

}  // namespace asw_small
}  // namespace open_spiel
