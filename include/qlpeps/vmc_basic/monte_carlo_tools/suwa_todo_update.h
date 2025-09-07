/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-11-26
*
* Description: QuantumLiquids/PEPS project. PRL 105, 120603 (2010)
*/

#ifndef QLPEPS_MONTE_CARLO_TOOLS_SUWA_TODO_UPDATE_H
#define QLPEPS_MONTE_CARLO_TOOLS_SUWA_TODO_UPDATE_H

#include <cstddef>    //size_t
#include <vector>
#include <algorithm>  //max_element
#include <cassert>
#include <random>
#include <cmath>
#include <cstdio>

namespace qlpeps {

/**
 * @brief Single-state update using the Suwa–Todo algorithm (geometric overlap on a ring).
 *
 * Mathematics (ring construction): Let \( S = \sum_{j=0}^{n-1} w_j \),
 * prefix sums \( s_k = \sum_{t=0}^{k} w_t \) with \( s_{-1}=0 \), and target intervals
 * \( I_j = [ s_{j-1}, s_j ) \) on \([0, S)\).
 * Internally, we swap the maximum weight to index 0 (pure relabeling) and take a fixed offset
 * \( C = w_0 \). For current state \( i \), define \( s_{i-1} = (i==0?0:s[i-1]) \) and
 * source interval \( J_i = [ (s_{i-1}+C)\bmod S,\ (s_{i-1}+C+w_i)\bmod S ) \).
 * Sampling \( x \sim U(J_i) \) and returning the unique \( j \) with \( x \in I_j \)
 * yields the Suwa–Todo transition probabilities, which are equivalent to the classical
 * v-min/max formula but numerically simpler and rejection-free.
 *
 * Implementation notes:
 * - Weights are received by value intentionally to allow internal swapping (no side effects).
 * - Intervals are half-open on the right; we use std::nextafter to avoid sampling the upper bound.
 * - The temporary swap of the maximal weight to index 0 is fully mapped back before return,
 *   so the external state ordering remains unchanged and detailed-balance-free stationarity holds.
 *
 * Preconditions/assumptions:
 * - weights.size() > 0, init_state < weights.size().
 * - All weights are non-negative and weights[init_state] > 0.
 * - The global order of states is fixed across the whole simulation and does NOT depend on init_state.
 *   Violating this breaks the balance condition and biases the results.
 *
 * @param init_state The state in the last step (index into weights).
 * @param weights Non-normalized, non-negative relative weights of states (by-value for internal swap).
 * @param generator Random number generator.
 * @return The updated state index.
 *
 * Reference: H. Suwa and S. Todo, Phys. Rev. Lett. 105, 120603 (2010).
 */
template<class RandGenerator>
size_t SuwaTodoStateUpdate(size_t init_state,
                            std::vector<double> weights,
                            RandGenerator &generator) {
  const size_t n = weights.size();
//  std::vector<double> p(n, 0.0); //transition probabilities
#ifndef NDEBUG
  assert(n > 0);
  assert(init_state < n);
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

  std::vector<long double> s(n);
  s[0] = static_cast<long double>(weights[0]);
  for (size_t i = 1; i < n; i++) {
    s[i] = s[i - 1] + static_cast<long double>(weights[i]);
  }
  // Geometric method: direct sampling on [start, start + w_i)
  const long double S = s.back();
  #ifndef NDEBUG
  assert(S > 0.0L);
  #endif

  const long double s_im1 = (init_state == 0) ? 0.0L : s[init_state - 1];
  long double start = s_im1 + static_cast<long double>(weights[0]);
  if (start >= S) start -= S; // one wrap is enough

  std::uniform_real_distribution<long double> uniform_dist_x(
      start,
      std::nextafter(start + static_cast<long double>(weights[init_state]), start)
  );
  long double x = uniform_dist_x(generator); // directly sample on [start, start + w_i)
  if (x >= S) x -= S; // modulo S (at most one wrap)

  size_t final_state = static_cast<size_t>(std::upper_bound(s.begin(), s.end(), x) - s.begin());
  if (max_weight_id != 0) {
    if (final_state == 0) {
      final_state = max_weight_id;
    } else if (final_state == max_weight_id) {
      final_state = 0;
    }
  }
  return final_state;
}

// Backward compatibility wrapper
template<class RandGenerator>
[[deprecated("NonDBMCMCStateUpdate is deprecated; use SuwaTodoStateUpdate instead.")]]
size_t NonDBMCMCStateUpdate(size_t init_state,
                            std::vector<double> weights,
                            RandGenerator &generator) {
  return SuwaTodoStateUpdate(init_state, std::move(weights), generator);
}

}//qlpeps

#endif // QLPEPS_MONTE_CARLO_TOOLS_SUWA_TODO_UPDATE_H
