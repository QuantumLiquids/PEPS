// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Optimizer parameters structure.
*/

#ifndef QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H
#define QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H

#include <vector>
#include <variant>
#include <string>

namespace qlpeps {

/**
 * @enum WAVEFUNCTION_UPDATE_SCHEME
 * @brief Enumeration of different update schemes for variational wavefunction optimization.
 *
 * The update schemes correspond to different algorithms for updating the PEPS tensor network parameters
 * during variational Monte Carlo optimization. Each scheme can be described mathematically as follows:
 *
 * 1. StochasticGradient(SGD):
 *    \f[
 *      \theta_{k+1} = \theta_k - \eta \nabla_\theta E(\theta_k)
 *    \f]
 *    where \f$\theta\f$ are the variational parameters, \f$\eta\f$ is the step size, and \f$E(\theta)\f$ is the energy.
 *
 * 2. RandomStepStochasticGradient:
 *    \f[
 *      \theta_{k+1} = \theta_k - \eta \cdot r \cdot \nabla_\theta E(\theta_k)
 *    \f]
 *    where \f$r\f$ is a random number in (0,1), introducing stochasticity in the step size.
 *
 * 3. StochasticReconfiguration(SR):
 *    \f[
 *      S \cdot \Delta\theta = -\eta \nabla_\theta E(\theta)
 *    \f]
 *    where \f$S\f$ is the quantum Fisher information matrix (S-matrix), and \f$\Delta\theta\f$ is the natural gradient direction.
 *
 * 4. RandomStepStochasticReconfiguration:
 *    \f[
 *      S \cdot \Delta\theta = -\eta \cdot r \cdot \nabla_\theta E(\theta)
 *    \f]
 *    where \f$r\f$ is a random number in (0,1).
 *
 * 5. NormalizedStochasticReconfiguration:
 *    \f[
 *      S \cdot \Delta\theta = -\eta \nabla_\theta E(\theta), \quad \Delta\theta \leftarrow \frac{\Delta\theta}{\|\Delta\theta\|}
 *    \f]
 *    The natural gradient is normalized before applying the update.
 *
 * 6. RandomGradientElement:
 *    \f[
 *      \theta_{k+1} = \theta_k - \eta \cdot \text{RandomSign}(\nabla_\theta E(\theta_k))
 *    \f]
 *    where each element of the gradient is randomly multiplied by \f$\pm 1\f$.
 *
 * 7. BoundGradientElement:
 *    \f[
 *      \theta_{k+1} = \theta_k - \eta \cdot \text{clip}(\nabla_\theta E(\theta_k), -b, b)
 *    \f]
 *    where the gradient elements are bounded by a threshold \f$b\f$.
 *
 * 8. GradientLineSearch:
 *    \f[
 *      \theta_{k+1} = \theta_k + \alpha^* d
 *    \f]
 *    where \f$d\f$ is the search direction (typically the negative gradient), and \f$\alpha^*\f$ is the optimal step length found by line search.
 *
 * 9. NaturalGradientLineSearch:
 *    \f[
 *      \theta_{k+1} = \theta_k + \alpha^* d_{nat}
 *    \f]
 *    where \f$d_{nat}\f$ is the natural gradient direction, and \f$\alpha^*\f$ is determined by line search.
 *
 * 10. AdaGrad:
 *     \f[
 *       G_k = G_{k-1} + (\nabla_\theta E(\theta_k))^2
 *       \theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{G_k + \epsilon}} \nabla_\theta E(\theta_k)
 *     \f]
 *     where \f$G_k\f$ accumulates the squared gradients and \f$\epsilon\f$ is a small constant for numerical stability.
 */
enum WAVEFUNCTION_UPDATE_SCHEME {
  StochasticGradient,                     //0
  RandomStepStochasticGradient,           //1
  StochasticReconfiguration,              //2
  RandomStepStochasticReconfiguration,    //3
  NormalizedStochasticReconfiguration,    //4
  RandomGradientElement,                  //5
  BoundGradientElement,                   //6
  GradientLineSearch,                     //7
  NaturalGradientLineSearch,              //8
  AdaGrad                                 //9
};

/**
 * @brief Convert update scheme enum to string representation
 */
std::string WavefunctionUpdateSchemeString(WAVEFUNCTION_UPDATE_SCHEME scheme) {
  switch (scheme) {
    case StochasticGradient: return "StochasticGradient";
    case RandomStepStochasticGradient: return "RandomStepStochasticGradient";
    case StochasticReconfiguration: return "StochasticReconfiguration";
    case RandomStepStochasticReconfiguration: return "RandomStepStochasticReconfiguration";
    case NormalizedStochasticReconfiguration: return "NormalizedStochasticReconfiguration";
    case RandomGradientElement: return "RandomGradientElement";
    case BoundGradientElement: return "BoundGradientElement";
    case GradientLineSearch: return "GradientLineSearch";
    case NaturalGradientLineSearch: return "NaturalGradientLineSearch";
    case AdaGrad: return "AdaGrad";
    default: return "Unknown scheme";
  }
}

const std::vector<WAVEFUNCTION_UPDATE_SCHEME> stochastic_reconfiguration_methods({
                                                                                     StochasticReconfiguration,
                                                                                     RandomStepStochasticReconfiguration,
                                                                                     NormalizedStochasticReconfiguration,
                                                                                     NaturalGradientLineSearch
                                                                                 });

bool IsStochasticReconfiguration(WAVEFUNCTION_UPDATE_SCHEME scheme) {
  return std::find(stochastic_reconfiguration_methods.begin(), stochastic_reconfiguration_methods.end(), scheme)
      != stochastic_reconfiguration_methods.end();
}

/**
 * @struct ConjugateGradientParams
 * @brief Parameters for conjugate gradient solver used in stochastic reconfiguration.
 */
struct ConjugateGradientParams {
  size_t max_iter;
  double tolerance;
  int residue_restart_step;
  double diag_shift;

  ConjugateGradientParams(void) : max_iter(0), tolerance(0.0), residue_restart_step(0), diag_shift(0.0) {}
  ConjugateGradientParams(size_t max_iter, double tolerance, int residue_restart_step, double diag_shift)
      : max_iter(max_iter), tolerance(tolerance), residue_restart_step(residue_restart_step), diag_shift(diag_shift) {}

  bool IsDefault(void) const {
    return max_iter == 0 && tolerance == 0.0 && residue_restart_step == 0 && diag_shift == 0.0;
  }
};

/**
 * @struct AdaGradParams
 * @brief Parameters for AdaGrad optimization algorithm.
 */
struct AdaGradParams {
  double epsilon;
  double initial_accumulator_value;

  AdaGradParams(double epsilon = 1e-8, double initial_accumulator_value = 0.0)
      : epsilon(epsilon), initial_accumulator_value(initial_accumulator_value) {}
};

/**
 * @struct OptimizerParams
 * @brief Complete parameter structure for the optimizer.
 */
struct OptimizerParams {
  /**
 * @struct BaseParams
 * @brief Core optimization parameters that are independent of specific algorithms.
 */
  struct BaseParams {
    size_t max_iterations;
    double energy_tolerance;
    double gradient_tolerance;
    size_t plateau_patience;        // Stop after N iterations without improvement
    std::vector<double> step_lengths;

    BaseParams(size_t max_iter, double energy_tol = 1e-15, double grad_tol = 1e-30,
               size_t patience = 20, const std::vector<double> &steps = {0.1})
        : max_iterations(max_iter), energy_tolerance(energy_tol), gradient_tolerance(grad_tol),
          plateau_patience(patience), step_lengths(steps) {}
  };

  BaseParams base_params;
  WAVEFUNCTION_UPDATE_SCHEME update_scheme;
  std::variant<std::monostate, AdaGradParams> algorithm_params;
  ConjugateGradientParams cg_params;

  OptimizerParams(void) : base_params(0) { ; } // default we expect non-sense params.

  OptimizerParams(const BaseParams &core_params,
                  WAVEFUNCTION_UPDATE_SCHEME scheme,
                  const ConjugateGradientParams &cg = ConjugateGradientParams())
      : base_params(core_params), update_scheme(scheme), cg_params(cg) {
    if (IsStochasticReconfiguration(scheme)) {
      if (cg.IsDefault()) {
        throw std::invalid_argument("ConjugateGradientParams is not provided for stochastic reconfiguration methods");
      }
    }
  }

  OptimizerParams(const BaseParams &core_params,
                  const AdaGradParams &adagrad_params)
      : base_params(core_params), update_scheme(AdaGrad), algorithm_params(adagrad_params) {}

  AdaGradParams GetAdaGradParams() const {
    if (std::holds_alternative<AdaGradParams>(algorithm_params)) {
      return std::get<AdaGradParams>(algorithm_params);
    }
    return AdaGradParams();
  }

  static OptimizerParams CreateStochasticGradient(const std::vector<double> &step_lengths,
                                                  size_t max_iterations) {
    return OptimizerParams(BaseParams(max_iterations, 1e-15, 1e-15, 20, step_lengths),
                           StochasticGradient);
  }

  static OptimizerParams CreateAdaGrad(double step_length,
                                       double epsilon,
                                       size_t max_iterations) {
    return OptimizerParams(BaseParams(max_iterations, 1e-15, 1e-15, 20, {step_length}),
                           AdaGradParams(epsilon));
  }

  static OptimizerParams CreateStochasticReconfiguration(const std::vector<double> &step_lengths,
                                                         const ConjugateGradientParams &cg_params,
                                                         size_t max_iterations) {
    return OptimizerParams(BaseParams(max_iterations, 1e-15, 1e-15, 20, step_lengths),
                           StochasticReconfiguration, cg_params);
  }
};

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H 