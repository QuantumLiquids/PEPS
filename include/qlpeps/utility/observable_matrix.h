// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xing Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-04
 *
 * Description: Lightweight row-major matrix wrapper for observable accumulation.
 */

#ifndef QLPEPS_UTILITY_OBSERVABLE_MATRIX_H
#define QLPEPS_UTILITY_OBSERVABLE_MATRIX_H

#include <vector>
#include <stdexcept>

#include "qlpeps/two_dim_tn/framework/site_idx.h"

namespace qlpeps {

/**
 * @brief Helper container that stores lattice observables in canonical row-major order.
 *
 * It provides semantic indexing via (row, col) or SiteIdx, hides stride math,
 * and exposes Flatten/Extract APIs for registry dump compatibility.
 */
template<typename T>
class ObservableMatrix {
 public:
  ObservableMatrix() = default;

  ObservableMatrix(size_t rows, size_t cols, const T &init_value = T())
      : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

  void Resize(size_t rows, size_t cols, const T &init_value = T()) {
    rows_ = rows;
    cols_ = cols;
    data_.assign(rows * cols, init_value);
  }

  [[nodiscard]] size_t rows() const { return rows_; }
  [[nodiscard]] size_t cols() const { return cols_; }
  [[nodiscard]] size_t size() const { return data_.size(); }

  T &operator()(size_t row, size_t col) {
    return data_.at(Index(row, col));
  }

  const T &operator()(size_t row, size_t col) const {
    return data_.at(Index(row, col));
  }

  T &operator()(const SiteIdx &site) {
    return (*this)(site.row(), site.col());
  }

  const T &operator()(const SiteIdx &site) const {
    return (*this)(site.row(), site.col());
  }

  void Add(size_t row, size_t col, const T &delta) {
    auto &ref = (*this)(row, col);
    ref += delta;
  }

  void Add(const SiteIdx &site, const T &delta) {
    Add(site.row(), site.col(), delta);
  }

  [[nodiscard]] const std::vector<T> &Flatten() const { return data_; }

  std::vector<T> Extract() {
    return std::move(data_);
  }

 private:
  [[nodiscard]] size_t Index(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("ObservableMatrix index out of range");
    }
    return row * cols_ + col;
  }

  size_t rows_{0};
  size_t cols_{0};
  std::vector<T> data_;
};

}  // namespace qlpeps

#endif  // QLPEPS_UTILITY_OBSERVABLE_MATRIX_H

