/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: Enum for single-site state in t-J model (spin up, spin down, empty).
*/

#ifndef QLPEPS_VMC_BASIC_TJ_SINGLE_SITE_STATE_H
#define QLPEPS_VMC_BASIC_TJ_SINGLE_SITE_STATE_H

namespace qlpeps {

/**
 * Single-site state for t-J model:
 * 0: SpinUp
 * 1: SpinDown
 * 2: Empty (hole)
 */
enum class tJSingleSiteState {
  SpinUp = 0,
  SpinDown = 1,
  Empty = 2
};

} // namespace qlpeps

#endif // QLPEPS_VMC_BASIC_TJ_SINGLE_SITE_STATE_H