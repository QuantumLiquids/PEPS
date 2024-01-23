// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: GraceQ/VMC-PEPS project. A fix size matrix which supports maintaining
*              elements using a reference (the memory managed by this class) or
*              a pointer (the memory managed by user themselves).
*/

/**
 * @file duomatrix.h
 * @brief A fix size matrix supporting elements maintaining by reference or pointer.
 */

#ifndef GQPEPS_VMC_PEPS_TWO_DIM_TN_FRAMEWORK_DUOMATRIX_H
#define GQPEPS_VMC_PEPS_TWO_DIM_TN_FRAMEWORK_DUOMATRIX_H

#include <vector>     // vector
#include <array>      // array
#include <utility>    // move
#include <cstddef>    // size_t
#include "gqpeps/basic.h"      //BondOrientation

namespace gqpeps {

/**
 * A fixed-size 2D matrix supporting elements maintained by reference or pointer.
 * @tparam ElemT Type of the elements.
 */
template<typename ElemT>
class DuoMatrix {
 public:
  /**
   * Default constructor.
   */
  DuoMatrix(void) = default;

  /**
   * Create a DuoMatrix using its size.
   * @param rows Number of rows in the matrix.
   * @param cols Number of columns in the matrix.
   */
  DuoMatrix(const size_t rows, const size_t cols) : raw_data_(rows, std::vector<ElemT *>(cols, nullptr)) {}

  /**
   * Create a DuoMatrix by copying another DuoMatrix.
   * @param duomat A DuoMatrix instance.
   */
  DuoMatrix(const DuoMatrix<ElemT> &duomat) : raw_data_(duomat.rows(),
                                                        std::vector<ElemT *>(duomat.cols(), nullptr)) {
    for (size_t i = 0; i < duomat.rows(); ++i) {
      for (size_t j = 0; j < duomat.cols(); ++j) {
        if (duomat(i, j) != nullptr) {
          raw_data_[i][j] = new ElemT(*duomat(i, j));
        }
      }
    }
  }

  /**
   * Copy a DuoMatrix.
   * @param rhs A DuoMatrix instance.
   */
  DuoMatrix<ElemT> &operator=(const DuoMatrix<ElemT> &rhs) {
    for (auto &row : raw_data_) {
      for (auto &elem : row) {
        if (elem != nullptr) {
          delete elem;
        }
      }
    }

    const size_t rows = rhs.rows();
    const size_t cols = rhs.cols();
    raw_data_ = std::vector<std::vector<ElemT * >>(rows, std::vector<ElemT *>(cols, nullptr));

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        if (rhs(i, j) != nullptr) {
          raw_data_[i][j] = new ElemT(*rhs(i, j));
        }
      }
    }

    return *this;
  }

  /**
   * Create a DuoMatrix by moving raw data from another DuoMatrix instance.
   * @param duomat A DuoMatrix instance.
   */
  DuoMatrix(DuoMatrix<ElemT> &&duomat)

  noexcept: raw_data_(std::move(duomat.raw_data_)) {
    duomat.raw_data_ = std::vector<std::vector<
        ElemT * >>(duomat.rows(), std::vector<ElemT *>(duomat.cols(), nullptr));
  }

  /**
   * Move a DuoMatrix with the same size.
   * @param rhs A DuoMatrix instance.
   */
  DuoMatrix<ElemT> &operator=(DuoMatrix<ElemT> &&rhs) noexcept {
    for (auto &row : raw_data_) {
      for (auto &elem : row) {
        if (elem != nullptr) {
          delete elem;
        }
      }
    }

    raw_data_ = std::move(rhs.raw_data_);
    rhs.raw_data_ =
        std::vector<std::vector<ElemT * >>(rhs.rows(), std::vector<ElemT *>(rhs.cols(), nullptr));

    return *this;
  }

  /**
   * Destruct a DuoMatrix. Release memory it maintained.
   */
  virtual ~DuoMatrix(void) {
    for (auto &row : raw_data_) {
      for (auto &elem : row) {
        if (elem != nullptr) {
          delete elem;
        }
      }
    }
  }

  // Data access methods.

  /**
* Element getter.
* @param row Row index of the element.
* @param col Column index of the element.
*/
  const ElemT &operator()(const std::array<size_t, 2> coordinate) const {
    const size_t row = coordinate[0];
    const size_t col = coordinate[1];
    return *raw_data_[row][col];
  }

  /**
   * Element setter. If the corresponding memory has not been allocated, allocate it first.
   * @param row Row index of the element.
   * @param col Column index of the element.
   */
  ElemT &operator()(const std::array<size_t, 2> coordinate) {
    const size_t row = coordinate[0];
    const size_t col = coordinate[1];
    if (raw_data_[row][col] == nullptr) {
      raw_data_[row][col] = new ElemT;
    }
    return *raw_data_[row][col];
  }

  /**
   * Pointer-to-element setter.
   * @param row Row index of the element.
   * @param col Column index of the element.
   */
  ElemT *&operator()(size_t row, size_t col) {
    return raw_data_[row][col];
  }

  /**
   * Read-only pointer-to-element getter.
   * @param row Row index of the element.
   * @param col Column index of the element.
   */
  const ElemT *operator()(size_t row, size_t col) const {
    return raw_data_[row][col];
  }

  /**
   * Read-only raw data access.
   */
  const std::vector<std::vector<const ElemT *>> cdata(void) const {
    std::vector<std::vector<const ElemT *>> craw_data;
    for (const auto &row : raw_data_) {
      std::vector<const ElemT *> crow;
      for (const auto &elem : row) {
        crow.push_back(elem);
      }
      craw_data.push_back(crow);
    }
    return craw_data;
  }

  // Memory management methods
  /**
   * Allocate memory of the element at the given rows_ and column indices. If the given place has a non-nullptr,
   * release the memory which points to first.
   * @param row Row index of the element.
   * @param col Column index of the element.
   */
  void alloc(const size_t row, const size_t col) {
    if (raw_data_[row][col] != nullptr) {
      delete raw_data_[row][col];
    }
    raw_data_[row][col] = new ElemT;
  }

  /**
   * Deallocate memory of the element at the given rows_ and column indices.
   * @param row Row index of the element.
   * @param col Column index of the element.
   */
  void dealloc(const size_t row, const size_t col) {
    if (raw_data_[row][col] != nullptr) {
      delete raw_data_[row][col];
      raw_data_[row][col] = nullptr;
    }
  }

  void has_alloc(const size_t row, const size_t col) {
    return raw_data_[row][col] != nullptr;
  }

  /**
   * Deallocate all elements.
   */
  void clear(void) {
    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j) {
        dealloc(i, j);
      }
    }
  }

  // Property methods
  /**
   * Get the number of rows in the DuoMatrix.
   */
  size_t rows(void) const { return raw_data_.size(); }

  /**
   * Get the number of columns in the DuoMatrix.
   */
  size_t cols(void) const { return (raw_data_.empty() ? 0 : raw_data_[0].size()); }

  size_t length(BondOrientation orientation) const {
    return (orientation == HORIZONTAL) ? cols() : rows();
  }

  /**
   * Get the number of elements in the DuoMatrix.
   */
  size_t size(void) const { return rows() * cols(); }

  /**
   * Check whether the matrix is empty.
   */
  bool empty(void) const {
    for (const auto &row : raw_data_) {
      for (const auto &elem : row) {
        if (elem != nullptr) {
          return false;
        }
      }
    }
    return true;
  }

/**
 * Extracts a specified column from the DuoMatrix as a plain std::vector<ElemT*>.
 * @param col Column index to extract.
 * @return std::vector containing the pointers to the elements from the specified column.
 */
  std::vector<ElemT *> get_col(size_t col) const {
    std::vector<ElemT *> result(rows(), nullptr);
    for (size_t row = 0; row < rows(); ++row) {
      result[row] = raw_data_[row][col];
    }
    return result;
  }

/**
 * Extracts a specified rows_ from the DuoMatrix as a plain std::vector<ElemT*>.
 * @param row Row index to extract.
 * @return std::vector containing the pointers to the elements from the specified rows_.
 */
  const std::vector<ElemT *> &get_row(size_t row) const {
    return raw_data_[row];
  }

  std::vector<ElemT *> get_slice(size_t num, BondOrientation orient) const {
    if (orient == HORIZONTAL) {
      return get_row(num);
    } else {
      return get_col(num);
    }
  }

  // Define an iterator class for DuoMatrix
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ElemT;
    using difference_type = std::ptrdiff_t;
    using pointer = ElemT *;
    using reference = ElemT &;

    Iterator(DuoMatrix<ElemT> &matrix, size_t row = 0, size_t col = 0)
        : matrix_(matrix), row_(row), col_(col) {}

    // Dereference operator
    reference operator*() const {
      return matrix_({row_, col_});
    }

    // Pre-increment operator
    Iterator &operator++() {
      incrementPosition();
      return *this;
    }

    // Post-increment operator
    Iterator operator++(int) {
      Iterator temp = *this;
      incrementPosition();
      return temp;
    }

    // Equality operator
    bool operator==(const Iterator &other) const {
      return (&matrix_ == &other.matrix_) && (row_ == other.row_) && (col_ == other.col_);
    }

    // Inequality operator
    bool operator!=(const Iterator &other) const {
      return !(*this == other);
    }

   private:
    void incrementPosition() {
      if (col_ < matrix_.cols() - 1) {
        col_++;
      } else if (row_ < matrix_.rows() - 1) {
        row_++;
        col_ = 0;
      } else {
        // Reached the end of the matrix, set to invalid position
        row_ = matrix_.rows();
        col_ = matrix_.cols();
      }
    }

    DuoMatrix<ElemT> &matrix_;
    size_t row_;
    size_t col_;
  };

  // Begin iterator for DuoMatrix
  Iterator begin() {
    return Iterator(*this);
  }

  // End iterator for DuoMatrix
  Iterator end() {
    return Iterator(*this, rows(), cols());
  }

  // Define a const iterator class for DuoMatrix
  class ConstIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const ElemT;
    using difference_type = std::ptrdiff_t;
    using pointer = const ElemT *;
    using reference = const ElemT &;

    ConstIterator(const DuoMatrix<ElemT> &matrix, size_t row = 0, size_t col = 0)
        : matrix_(matrix), row_(row), col_(col) {}

    // Dereference operator
    reference operator*() const {
      return matrix_({row_, col_});
    }

    // Pre-increment operator
    ConstIterator &operator++() {
      incrementPosition();
      return *this;
    }

    // Post-increment operator
    ConstIterator operator++(int) {
      ConstIterator temp = *this;
      incrementPosition();
      return temp;
    }

    // Equality operator
    bool operator==(const ConstIterator &other) const {
      return (&matrix_ == &other.matrix_) && (row_ == other.row_) && (col_ == other.col_);
    }

    // Inequality operator
    bool operator!=(const ConstIterator &other) const {
      return !(*this == other);
    }

   private:
    void incrementPosition() {
      if (col_ < matrix_.cols() - 1) {
        col_++;
      } else if (row_ < matrix_.rows() - 1) {
        row_++;
        col_ = 0;
      } else {
        // Reached the end of the matrix, set to invalid position
        row_ = matrix_.rows();
        col_ = matrix_.cols();
      }
    }

    const DuoMatrix<ElemT> &matrix_;
    size_t row_;
    size_t col_;
  };

  // Begin const iterator for DuoMatrix
  ConstIterator cbegin() const {
    return ConstIterator(*this);
  }

  // End const iterator for DuoMatrix
  ConstIterator cend() const {
    return ConstIterator(*this, rows(), cols());
  }

  // Begin const iterator for DuoMatrix (const overload)
  ConstIterator begin() const {
    return cbegin();
  }

  // End const iterator for DuoMatrix (const overload)
  ConstIterator end() const {
    return cend();
  }

 private:
  std::vector<std::vector<ElemT *>> raw_data_;
};
}//gqpeps


#endif //GQPEPS_VMC_PEPS_TWO_DIM_TN_FRAMEWORK_DUOMATRIX_H
