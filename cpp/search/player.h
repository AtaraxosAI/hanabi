#pragma once

#include "cpp/search/game_sim.h"
#include "cpp/infra/tensor_dict.h"

namespace search {

class Player {
 public:
  Player(int index, std::shared_ptr<infra::BatchRunner> bpModel, infra::TensorDict bpHid)
      : index(index)
      , bpModel_(std::move(bpModel))
      , bpHid_(std::move(bpHid)) {
  }

  Player(const Player& p)
      : index(p.index)
      , bpModel_(p.bpModel_)
      , bpHid_(p.bpHid_) {
  }

//   void setLLMPrior(
//       const std::unordered_map<std::string, torch::Tensor>& llmPrior,
//       float piklLambda,
//       float piklBeta) {
//     assert(llmPrior_.size() == 0); // has not been previously set
//     assert(piklBeta == 1);
//     // for (const auto& kv : llmPrior) {
//     //   llmPrior_[kv.first] = kv.second, torch::kFloat32);
//     // }
//     llmPrior_ = llmPrior;
//     piklLambda_ = piklLambda;
//     piklBeta_ = piklBeta;
//   }

  // observe before act
  void observeBeforeAct(const GameSimulator& env);

  int decideAction(const GameSimulator& env);

  const int index;

 private:
  std::shared_ptr<infra::BatchRunner> bpModel_;
  infra::TensorDict bpHid_;
  infra::Future futBp_;

//   float piklLambda_ = -1;
//   float piklBeta_ = -1;
//   std::unordered_map<std::string, torch::Tensor> llmPrior_;
};
}  // namespace search
