// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-24
*
* Description: QuantumLiquids/PEPS project. Learning rate schedulers.
*/

#ifndef QLPEPS_OPTIMIZER_LR_SCHEDULERS_H
#define QLPEPS_OPTIMIZER_LR_SCHEDULERS_H

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <numbers>

namespace qlpeps {

// =============================================================================
// LEARNING RATE SCHEDULER INTERFACE
// =============================================================================

class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;
  virtual double GetLearningRate(size_t iteration, double current_energy = 0.0) const = 0;
  virtual void Step() {}
  virtual std::unique_ptr<LearningRateScheduler> Clone() const = 0;
  virtual std::string Name() const = 0;
  virtual std::string Describe() const = 0;
};

// Constant LR
class ConstantLR : public LearningRateScheduler {
private:
  double learning_rate_;
public:
  explicit ConstantLR(double lr) : learning_rate_(lr) {}
  double GetLearningRate(size_t, double) const override { return learning_rate_; }
  std::unique_ptr<LearningRateScheduler> Clone() const override { return std::make_unique<ConstantLR>(learning_rate_); }
  std::string Name() const override { return "ConstantLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "lr=" << learning_rate_;
    return oss.str();
  }
};

// Exponential decay (smooth per-iteration): lr = initial_lr * decay_rate^(iteration/decay_steps)
class ExponentialDecayLR : public LearningRateScheduler {
private:
  double initial_lr_;
  double decay_rate_;
  size_t decay_steps_;
public:
  ExponentialDecayLR(double initial_lr, double decay_rate, size_t decay_steps)
    : initial_lr_(initial_lr), decay_rate_(decay_rate), decay_steps_(decay_steps) {}
  double GetLearningRate(size_t iteration, double) const override {
    return initial_lr_ * std::pow(decay_rate_, iteration / static_cast<double>(decay_steps_));
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<ExponentialDecayLR>(initial_lr_, decay_rate_, decay_steps_);
  }
  std::string Name() const override { return "ExponentialDecayLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "initial_lr=" << initial_lr_ << ", decay_rate=" << decay_rate_ << ", decay_steps=" << decay_steps_;
    return oss.str();
  }
};

// Step decay (discrete): lr = initial_lr * gamma^(floor(iteration/step_size))
class StepLR : public LearningRateScheduler {
private:
  double initial_lr_;
  double gamma_;
  size_t step_size_;
public:
  StepLR(double initial_lr, size_t step_size, double gamma = 0.1)
    : initial_lr_(initial_lr), gamma_(gamma), step_size_(step_size) {}
  double GetLearningRate(size_t iteration, double) const override {
    return initial_lr_ * std::pow(gamma_, iteration / step_size_);
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<StepLR>(initial_lr_, step_size_, gamma_);
  }
  std::string Name() const override { return "StepLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "initial_lr=" << initial_lr_ << ", step_size=" << step_size_ << ", gamma=" << gamma_;
    return oss.str();
  }
};

// Plateau-aware LR (energy plateau detection)
class PlateauLR : public LearningRateScheduler {
private:
  mutable double current_lr_;
  double factor_;
  size_t patience_;
  mutable size_t patience_counter_;
  mutable double best_energy_;
  double threshold_;
public:
  PlateauLR(double initial_lr, double factor = 0.5, size_t patience = 10, double threshold = 1e-4)
    : current_lr_(initial_lr), factor_(factor), patience_(patience),
      patience_counter_(0), best_energy_(std::numeric_limits<double>::max()), threshold_(threshold) {}
  double GetLearningRate(size_t, double current_energy) const override {
    if (current_energy < best_energy_ - threshold_) {
      best_energy_ = current_energy;
      patience_counter_ = 0;
    } else {
      patience_counter_++;
      if (patience_counter_ >= patience_) {
        current_lr_ *= factor_;
        patience_counter_ = 0;
        best_energy_ = current_energy;
      }
    }
    return current_lr_;
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    auto clone = std::make_unique<PlateauLR>(current_lr_, factor_, patience_, threshold_);
    clone->best_energy_ = best_energy_;
    clone->patience_counter_ = patience_counter_;
    return clone;
  }
  std::string Name() const override { return "PlateauLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "current_lr=" << current_lr_ << ", factor=" << factor_
        << ", patience=" << patience_ << ", threshold=" << threshold_;
    return oss.str();
  }
};

// Cosine annealing: lr = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi * t/T))
class CosineAnnealingLR : public LearningRateScheduler {
private:
  double eta_max_;
  double eta_min_;
  size_t T_max_;
public:
  CosineAnnealingLR(double eta_max, size_t T_max, double eta_min = 0.0)
    : eta_max_(eta_max), eta_min_(eta_min), T_max_(T_max) {}
  double GetLearningRate(size_t iteration, double) const override {
    if (T_max_ == 0) { return eta_min_; }
    double t = static_cast<double>(std::min(iteration, T_max_));
    double cosine = std::cos(std::numbers::pi * t / static_cast<double>(T_max_));
    return eta_min_ + 0.5 * (eta_max_ - eta_min_) * (1.0 + cosine);
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<CosineAnnealingLR>(eta_max_, T_max_, eta_min_);
  }
  std::string Name() const override { return "CosineAnnealingLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "eta_max=" << eta_max_ << ", T_max=" << T_max_ << ", eta_min=" << eta_min_;
    return oss.str();
  }
};

// Linear Warmup then Constant or target base rate
class WarmupLR : public LearningRateScheduler {
private:
  double base_lr_;
  size_t warmup_steps_;
  double start_lr_;
public:
  // Linear increase from start_lr to base_lr over warmup_steps; after that, clamp at base_lr
  WarmupLR(double base_lr, size_t warmup_steps, double start_lr = 0.0)
    : base_lr_(base_lr), warmup_steps_(warmup_steps), start_lr_(start_lr) {}
  double GetLearningRate(size_t iteration, double) const override {
    if (warmup_steps_ == 0) { return base_lr_; }
    if (iteration >= warmup_steps_) { return base_lr_; }
    double ratio = static_cast<double>(iteration + 1) / static_cast<double>(warmup_steps_);
    return start_lr_ + (base_lr_ - start_lr_) * ratio;
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<WarmupLR>(base_lr_, warmup_steps_, start_lr_);
  }
  std::string Name() const override { return "WarmupLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "base_lr=" << base_lr_ << ", warmup_steps=" << warmup_steps_ << ", start_lr=" << start_lr_;
    return oss.str();
  }
};

// MultiStep decay with milestones
class MultiStepLR : public LearningRateScheduler {
private:
  double initial_lr_;
  double gamma_;
  std::vector<size_t> milestones_;
public:
  MultiStepLR(double initial_lr, std::vector<size_t> milestones, double gamma)
    : initial_lr_(initial_lr), gamma_(gamma), milestones_(std::move(milestones)) {
    std::sort(milestones_.begin(), milestones_.end());
  }
  double GetLearningRate(size_t iteration, double) const override {
    // count milestones <= iteration
    size_t count = std::upper_bound(milestones_.begin(), milestones_.end(), iteration) - milestones_.begin();
    return initial_lr_ * std::pow(gamma_, static_cast<double>(count));
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<MultiStepLR>(initial_lr_, milestones_, gamma_);
  }
  std::string Name() const override { return "MultiStepLR"; }
  std::string Describe() const override {
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "initial_lr=" << initial_lr_ << ", gamma=" << gamma_ << ", milestones=[";
    for (size_t i = 0; i < milestones_.size(); ++i) {
      if (i) { oss << ","; }
      oss << milestones_[i];
    }
    oss << "]";
    return oss.str();
  }
};

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_LR_SCHEDULERS_H

