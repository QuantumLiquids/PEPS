// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-09-05
*
* Description: QuantumLiquids/PEPS project. Explicit, named conversion APIs
* between PEPS, TPS, and SplitIndexTPS to avoid implicit conversions.
*/

#ifndef QLPEPS_API_CONVERSIONS_H
#define QLPEPS_API_CONVERSIONS_H

// Intentionally avoid including heavy headers here to prevent cycles.
// Forward declare templates and define conversion wrappers.

namespace qlpeps {

template<typename TenElemT, typename QNT>
class TPS;

template<typename TenElemT, typename QNT>
class SplitIndexTPS;

template<typename TenElemT, typename QNT>
class SquareLatticePEPS;

} // namespace qlpeps

namespace qlpeps {

/**
 * Convert SquareLatticePEPS to TPS explicitly.
 * Prefer this over implicit operator conversions.
 */
template<typename TenElemT, typename QNT>
inline TPS<TenElemT, QNT>
ToTPS(const SquareLatticePEPS<TenElemT, QNT> &peps) {
  return peps.ToTPS();
}

/**
 * Convert TPS to SplitIndexTPS explicitly.
 */
template<typename TenElemT, typename QNT>
inline SplitIndexTPS<TenElemT, QNT>
ToSplitIndexTPS(const TPS<TenElemT, QNT> &tps) {
  return SplitIndexTPS<TenElemT, QNT>::FromTPS(tps);
}

/**
 * Convert SquareLatticePEPS to SplitIndexTPS explicitly.
 * Implemented by composition: PEPS -> TPS -> SplitIndexTPS.
 */
template<typename TenElemT, typename QNT>
inline SplitIndexTPS<TenElemT, QNT>
ToSplitIndexTPS(const SquareLatticePEPS<TenElemT, QNT> &peps) {
  return ToSplitIndexTPS<TenElemT, QNT>(ToTPS<TenElemT, QNT>(peps));
}

/**
 * Alias: Split physical index (same as ToSplitIndexTPS)
 * TPS -> SplitIndexTPS
 */
template<typename TenElemT, typename QNT>
inline SplitIndexTPS<TenElemT, QNT>
SplitPhyIndex(const TPS<TenElemT, QNT> &tps) {
  return ToSplitIndexTPS<TenElemT, QNT>(tps);
}

} // namespace qlpeps

#endif // QLPEPS_API_CONVERSIONS_H


