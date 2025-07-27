/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-07-22
*
* Description: Jastrow factor class for square lattice with OBC.
*              Stores only v_{ij} for i < j (no diagonal, no double counting).
*/

#ifndef QLPEPS_VMC_BASIC_JASTROW_FACTOR_H
#define QLPEPS_VMC_BASIC_JASTROW_FACTOR_H

#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include "qlpeps/basic.h" // for SiteIdx

namespace qlpeps {

/*
 * same data structure, but different explanation in physics.
 * In figure, maybe density can be negative or double, at that time, update the data structure.
*/
using DensityConfig = Configuration;

/**
 * JastrowFactor: stores v_{ij} for i < j 
 * (OBC, no translational symmetry. No diagonal i==j since it can be absorbed into the PEPS).
 * Access via operator()(i, j) or operator()(SiteIdx, SiteIdx).
 *
 * The convention of Jastrow factor we use is
 *  exp(\sum_ij v_ij * n_i * n_j) with i < j, and no any double counting and no 1/2 factor.
 */
class JastrowFactor {
 public:
  JastrowFactor(size_t Ly, size_t Lx)
      : Ly_(Ly), Lx_(Lx), N_(Ly * Lx), v_ij_((N_ * (N_ - 1)) / 2, 0.0) {}

  // Set v_{ij} for i < j
  void Set(size_t i, size_t j, double v) {
    assert(i < N_ && j < N_ && i != j);
    if (i < j) v_ij_[PackedUpperIndex(i, j)] = v;
    else v_ij_[PackedUpperIndex(j, i)] = v;
  }

  // Convert SiteIdx to linear site index
  size_t SiteIdxToLinear(const SiteIdx &s) const {
    assert(s.row() < Ly_ && s.col() < Lx_);
    return s.row() * Lx_ + s.col();
  }

  void Set(const SiteIdx &si, const SiteIdx &sj, double v) {
    Set(SiteIdxToLinear(si), SiteIdxToLinear(sj), v);
  }

  double operator()(const SiteIdx &si, const SiteIdx &sj) const {
    return (*this)(SiteIdxToLinear(si), SiteIdxToLinear(sj));
  }

  double &operator()(const SiteIdx &si, const SiteIdx &sj) {
    size_t i = SiteIdxToLinear(si), j = SiteIdxToLinear(sj);
    assert(i != j);
    return i < j ? v_ij_[PackedUpperIndex(i, j)] : v_ij_[PackedUpperIndex(j, i)];
  }

  template<typename Func>
  void Fill(Func f) {
    for (size_t i = 0; i < N_; ++i)
      for (size_t j = i + 1; j < N_; ++j)
        v_ij_[PackedUpperIndex(i, j)] = f(i, j);
  }

  double operator()(size_t i, size_t j) const {
    assert(i < N_ && j < N_);
    if (i == j) return 0.0;
    return i < j ? v_ij_[PackedUpperIndex(i, j)] : v_ij_[PackedUpperIndex(j, i)];
  }

  double &operator()(size_t i, size_t j) {
    assert(i < N_ && j < N_ && i != j);
    return i < j ? v_ij_[PackedUpperIndex(i, j)] : v_ij_[PackedUpperIndex(j, i)];
  }

  size_t NumSites() const { return N_; }

  size_t SiteToIndex(const SiteIdx &s) const {
    assert(s.row() < Ly_ && s.col() < Lx_);
    return s.row() * Lx_ + s.col();
  }

  /**
  * Calculate the Jastrow field at site i:
  *   sum_j v_{ij} * n_j, where n_j is the density at site j.
  *   Only i != j is considered.
  * @param density_config  Configuration object, n_j = config(j)
  * @param site_i          Linear site index (0 ... N-1)
  * @return                double, the Jastrow field at site i
  */
  double JastrowFieldAtSite(const DensityConfig &density_config, const SiteIdx &site_i) const {
    double sum = 0.0;
    size_t linear_idx_i = SiteIdxToLinear(site_i);
    for(size_t row = 0; row < Ly_; ++row) {
      for (size_t col = 0; col < Lx_; ++col) {
        size_t j = row * Lx_ + col;
        if (j == linear_idx_i) continue; // skip self
        sum += (*this)(linear_idx_i, j) * density_config({row, col});
      }
    }
    return sum;
  }

 private:
  size_t Ly_, Lx_, N_;
  // Map (i, j) with i < j to packed upper triangle index
  size_t PackedUpperIndex(size_t i, size_t j) const {
    assert(i < j && j < N_);
    return i * N_ + j - ((i + 1) * (i + 2)) / 2;
  }

  std::vector<double> v_ij_; // packed upper triangle, no diagonal
};

} // namespace qlpeps

#endif // QLPEPS_VMC_BASIC_JASTROW_FACTOR_H