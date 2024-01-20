/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-11-26
*
* Description: GraceQ/VMC-PEPS project. PRL 105, 120603 (2010)
*/

#ifndef GRACEQ_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H
#define GRACEQ_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H

#include <cstddef>    //size_t
#include <vector>
#include <algorithm>  //max_element
#include <assert.h>

namespace gqpeps {

size_t NonDBMCMCStateUpdate(size_t init_state,
                            std::vector<double> weights,
                            const double rand_num) {
  const size_t n = weights.size();
  std::vector<double> p(n, 0.0); //transition probabilities

  // Swap the first weight with the maximum weight
  auto max_weight_iter = std::max_element(weights.cbegin(), weights.cend());
  size_t max_weight_id = max_weight_iter - weights.cbegin();
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
                    std::min({delta[j], weights[init_state] + weights[j] - delta[j], weights[init_state], weights[j]}));
    if (weights[j] != 0.0)
      p[j] = v[j] / weights[init_state];
  }
#ifndef NDEBUG
  double sum_p = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum_p += p[i];
  }
  assert(std::abs(sum_p - 1.0) < 1e-13);
#endif
  double p_accumulate = 0.0;
  size_t final_state;
  for (size_t j = 0; j < n; j++) {
    p_accumulate += p[j];
    if (rand_num < p_accumulate) {
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

}//gqpeps

#endif //GRACEQ_VMC_PEPS_NON_DETAILED_BALANCE_MCMC_H
