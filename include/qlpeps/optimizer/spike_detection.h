// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-09-01
*
* Description: QuantumLiquids/PEPS project. Spike detection types and EMA tracker
*              for VMC optimization stability.
*/

#ifndef QLPEPS_OPTIMIZER_SPIKE_DETECTION_H
#define QLPEPS_OPTIMIZER_SPIKE_DETECTION_H

#include <string>
#include <vector>
#include <cmath>
#include <cstddef>

namespace qlpeps {

/**
 * @brief Signal types for spike detection in VMC optimization.
 *
 * - S1: Energy error bar anomalously large relative to EMA
 * - S2: Gradient norm anomalously large relative to EMA
 * - S3: Natural gradient norm anomaly (SR-only), or suspiciously few CG iterations
 * - S4: Energy spike upward relative to EMA (for rollback)
 */
enum class SpikeSignal {
  kNone,
  kS1_ErrorbarSpike,
  kS2_GradientNormSpike,
  kS3_NaturalGradientAnomaly,
  kS4_EMAEnergySpikeUpward
};

/**
 * @brief Action to take when a spike is detected.
 */
enum class SpikeAction {
  kAccept,
  kResample,
  kRollback,
  kAcceptWithWarning
};

/**
 * @brief Record of a single spike event for logging and diagnostics.
 */
struct SpikeEventRecord {
  size_t step;
  size_t attempt;
  SpikeSignal signal;
  SpikeAction action;
  double value;
  double threshold;
};

/**
 * @brief Aggregate statistics of spike events during an optimization run.
 */
struct SpikeStatistics {
  size_t total_resamples = 0;
  size_t total_rollbacks = 0;
  size_t total_forced_accepts = 0;
  std::vector<SpikeEventRecord> event_log;
};

/**
 * @brief Online Exponential Moving Average tracker with Welford-style variance estimate.
 *
 * Maintains running EMA mean and variance for a scalar time series.
 * Useful for adaptive spike detection thresholds.
 *
 * Algorithm:
 *   alpha = 2 / (window + 1)
 *   On first observation: ema = x, var = 0
 *   Subsequent: delta = x - ema; ema += alpha * delta; var = (1-alpha) * (var + alpha * delta^2)
 */
class EMATracker {
 public:
  explicit EMATracker(size_t window = 50)
      : alpha_(2.0 / (static_cast<double>(window) + 1.0)),
        ema_(0.0), var_(0.0), initialized_(false) {}

  void Update(double x) {
    if (!initialized_) {
      ema_ = x;
      var_ = 0.0;
      initialized_ = true;
      return;
    }
    double delta = x - ema_;
    ema_ += alpha_ * delta;
    var_ = (1.0 - alpha_) * (var_ + alpha_ * delta * delta);
  }

  double Mean() const { return ema_; }
  double Var() const { return var_; }
  double Std() const { return std::sqrt(var_); }
  bool IsInitialized() const { return initialized_; }

  void Reset() {
    ema_ = 0.0;
    var_ = 0.0;
    initialized_ = false;
  }

 private:
  double alpha_;
  double ema_;
  double var_;
  bool initialized_;
};

inline const char *SignalName(SpikeSignal s) {
  switch (s) {
    case SpikeSignal::kNone: return "NONE";
    case SpikeSignal::kS1_ErrorbarSpike: return "S1_ERRORBAR";
    case SpikeSignal::kS2_GradientNormSpike: return "S2_GRAD_NORM";
    case SpikeSignal::kS3_NaturalGradientAnomaly: return "S3_NGRAD_ANOMALY";
    case SpikeSignal::kS4_EMAEnergySpikeUpward: return "S4_ENERGY_SPIKE";
  }
  return "UNKNOWN";
}

inline const char *ActionName(SpikeAction a) {
  switch (a) {
    case SpikeAction::kAccept: return "ACCEPT";
    case SpikeAction::kResample: return "RESAMPLE";
    case SpikeAction::kRollback: return "ROLLBACK";
    case SpikeAction::kAcceptWithWarning: return "ACCEPT_WARN";
  }
  return "UNKNOWN";
}

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_SPIKE_DETECTION_H
