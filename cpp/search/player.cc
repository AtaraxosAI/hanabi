#include "cpp/search/player.h"

namespace search {

// observe before act
void Player::observeBeforeAct(const GameSimulator& env) {
  assert(futBp_.isNull());
  auto input = observe(env.state(), index);
  addHid(input, bpHid_);
  futBp_ = bpModel_->call("act", input);
}

int Player::decideAction(const GameSimulator& env) {
  // first get results from the futures, to update hid
  auto bpReply = futBp_.get();
  moveHid(bpReply, bpHid_);
  int action = bpReply.at("a").item<int64_t>();

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }
  return action;
}
}  // namespace search
