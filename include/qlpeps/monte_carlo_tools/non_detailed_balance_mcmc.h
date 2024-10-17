/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-11-26
*
* Description: QuantumLiquids/PEPS project. PRL 105, 120603 (2010)
*/

#ifndef QLPEPS_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H
#define QLPEPS_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H

#include <cstddef>    //size_t
#include <vector>
#include <algorithm>  //max_element
#include <numeric>    //reduce
#include <cassert>

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC   __GNUC__ // probably
#endif

namespace qlpeps {

template<class RandGenerator>
size_t NonDBMCMCStateUpdate(size_t init_state,
                            std::vector<double> weights,
                            RandGenerator &generator) {
  const size_t n = weights.size();
//  std::vector<double> p(n, 0.0); //transition probabilities
#ifndef NDEBUG
  for (auto w : weights) {
    assert(w >= 0);
  }
  assert(weights[init_state] > 0);
#endif
  // Swap the first weight with the maximum weight
  auto max_weight_iter = std::max_element(weights.cbegin(), weights.cend());
  const size_t max_weight_id = max_weight_iter - weights.cbegin();
#ifndef NDEBUG
  assert(weights[max_weight_id] > 0);
#endif
  if (max_weight_id != 0) {
    std::swap(weights[0], weights[max_weight_id]);
  }
  if (init_state == max_weight_id) {
    init_state = 0;
  } else if (init_state == 0) {
    init_state = max_weight_id;
  }

  std::vector<double> s(n);
  s[0] = weights[0];
  for (size_t i = 1; i < n; i++) {
    s[i] = s[i - 1] + weights[i];
  }
  std::vector<double> delta(n);
  delta[0] = s[init_state] - s.back() + weights[0];
  for (size_t j = 1; j < n; j++) {
    delta[j] = s[init_state] - s[j - 1] + weights[0];
  }
  std::vector<double> v(n); // weights * transition probabilities

  for (size_t j = 0; j < n; j++) {
    v[j] = std::max(0.0,
                    std::min({delta[j], weights[j] - delta[j] + weights[init_state], weights[init_state], weights[j]}));
//    if (weights[j] != 0.0)
//      p[j] = v[j] / weights[init_state];
  }
#ifndef NDEBUG
#if defined(REAL_GCC) && (__GNUC__ < 9 || (__GNUC__ == 9 && __GNUC_MINOR__ < 1))
  double sum_v = std::accumulate(v.begin(), v.end(), 0.0);
#else
  double sum_v = std::reduce(v.begin(), v.end());
#endif
  assert(std::abs(sum_v - weights[init_state]) / std::abs(weights[init_state]) < 1e-13);
#endif
  double v_accumulate = 0.0;
  size_t final_state;
  std::uniform_real_distribution<double> uniform_dist(0.0, weights[init_state]);
  double rand_num = uniform_dist(generator);
  for (size_t j = 0; j < n; j++) {
    v_accumulate += v[j];
    if (rand_num < v_accumulate) {
      final_state = j;
      break;
    }
  }
  if (max_weight_id != 0) {
    if (final_state == 0) {
      final_state = max_weight_id;
    } else if (final_state == max_weight_id) {
      final_state = 0;
    }
  }
  return final_state;
}

}//qlpeps

#endif //QLPEPS_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H
