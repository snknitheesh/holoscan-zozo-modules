// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the graph tests share: running a graph to its end, a source that sends
// prepared messages on a clock, and a sink that keeps what it receives.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan::messages::test {

using std::chrono_literals::operator""ms;

// Ground-truth tests lose nothing: a full queue holds its producer back.
inline holoscan::ConnectionOptions lossless() {
  return holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking};
}

struct RunOutcome {
  bool compiled{};
  bool timed_out{};
  holoscan::RunTermination termination{holoscan::RunTermination::kRunning};
  std::string diagnostics;

  // Completed on request, without a diagnostic.
  [[nodiscard]] bool ok() const {
    return compiled && !timed_out && diagnostics.empty() &&
           termination == holoscan::RunTermination::kRequestedCompleted;
  }
};

// Compiles and runs `graph` on a real-time clock until it ends, or stops it
// after `timeout`.
inline RunOutcome run_graph(holoscan::Graph& graph, holoscan::CompileOptions options = {},
                            std::chrono::seconds timeout = std::chrono::seconds{60}) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  RunOutcome outcome;
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    outcome.diagnostics = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return outcome;
  }
  outcome.compiled = true;
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  if (session.is_running()) {
    outcome.timed_out = true;
    session.request_stop();
  }
  session.wait();
  outcome.termination = session.termination();
  if (!session.diagnostics().empty()) {
    outcome.diagnostics = holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return outcome;
}

// Sends `messages` in order, one every `period`, each with its activation
// time as capture time; then stays idle.
template <typename T>
class ListSourceOp final : public holoscan::Operator<> {
 public:
  ListSourceOp(std::vector<T> messages, std::chrono::nanoseconds period)
      : messages_(std::move(messages)), period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }
  void start() override { next_ = 0; }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= messages_.size()) return {};
    auto sent = output.emit_ref(messages_[next_],
                                holoscan::EmitOptions{.capture_time = context.activation_time()});
    if (!sent) {
      // A consumer without room: the same message again on the next tick.
      if (sent.error().code == holoscan::ErrorCode::kBackpressured) return {};
      return sent;
    }
    ++next_;
    return {};
  }

  holoscan::Output<T> output;

 private:
  std::vector<T> messages_;
  std::chrono::nanoseconds period_;
  std::size_t next_{};
};

// One received message and its capture time.
template <typename T>
struct Received {
  T data;
  std::int64_t capture_timestamp_ns{};
};

// Keeps every message it receives; ends the run at the `count`th.
template <typename T>
class CaptureOp final : public holoscan::Operator<> {
 public:
  CaptureOp(std::shared_ptr<std::vector<Received<T>>> received, std::size_t count)
      : received_(std::move(received)), count_(count) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    received_->push_back({sample->data, sample->metadata.capture_timestamp_ns});
    if (received_->size() == count_) context.request_completion();
    return {};
  }

  holoscan::Input<T> input;

 private:
  std::shared_ptr<std::vector<Received<T>>> received_;
  std::size_t count_;
};

}  // namespace holoscan::messages::test
