// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: QuantumLiquids/PEPS project. A fix size matrix tensor matrix class
*/


#ifndef QLPEPS_TWO_DIM_TN_FRAMEWORK_TEN_MATRIX_H
#define QLPEPS_TWO_DIM_TN_FRAMEWORK_TEN_MATRIX_H

#include "qlpeps/two_dim_tn/framework/duomatrix.h"    //DuoMatrix
#include <fstream>                                      //ifs, ofs

namespace qlpeps {
/**
 * A fixed-size 2D matrix supporting elements maintained by reference or pointer, specialized for tensor elements.
 * @tparam TenT Type of the tensor elements.
 */
template<typename TenT>
class TenMatrix : public DuoMatrix<TenT> {
 public:
  /**
   * Default constructor.
   */
  TenMatrix(void) = default;

  /**
   * Create a TenMatrix using its size.
   * @param rows Number of rows in the matrix.
   * @param cols Number of columns in the matrix.
   */
  TenMatrix(const size_t rows, const size_t cols) : DuoMatrix<TenT>(rows, cols) {}

  /**
   * Create a TenMatrix by copying another TenMatrix.
   * @param tenmat A TenMatrix instance.
   */
  TenMatrix(const TenMatrix<TenT> &tenmat) : DuoMatrix<TenT>(tenmat) {}

  /**
   * Copy a TenMatrix.
   * @param rhs A TenMatrix instance.
   */
  TenMatrix<TenT> &operator=(const TenMatrix<TenT> &rhs) {
    DuoMatrix<TenT>::operator=(rhs);
    return *this;
  }

  /**
   * Create a TenMatrix by moving raw data from another TenMatrix instance.
   * @param tenmat A TenMatrix instance.
   */
  TenMatrix(TenMatrix<TenT> &&tenmat)
  noexcept: DuoMatrix<TenT>(std::move(tenmat)) {}

  /**
   * Move a TenMatrix.
   * @param rhs A TenMatrix instance.
   */
  TenMatrix<TenT> &operator=(TenMatrix<TenT> &&rhs)
  noexcept {
    DuoMatrix<TenT>::operator=(std::move(rhs));
    return *this;
  }

  /**
    * Load tensor element from a file.
    * @param row Row index of the element.
    * @param col Column index of the element.
    * @param file The file which contains the tensor to be loaded.
    * @return True if the tensor was loaded successfully, false otherwise.
    */
  bool LoadTen(const size_t row, const size_t col, const std::string &file) {
    this->alloc(row, col);
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) {
      return false; // Failed to open the file
    }
    ifs >> (*this)({row, col});
    if (!ifs) {
      return false; // Failed to read the tensor from the file
    }
    ifs.close();
    return true; // Successfully loaded the tensor
  }

  /**
   * Dump tensor element to a file.
   * @param row Row index of the element.
   * @param col Column index of the element.
   * @param file The element tensor will be dumped to this file.
   * @return True if the tensor was dumped successfully, false otherwise.
   */
  bool DumpTen(const size_t row, const size_t col, const std::string &file) const {
    std::ofstream ofs(file, std::ofstream::binary);
    if (!ofs) {
      return false; // Failed to open the file
    }
    ofs << (*this)({row, col});
    if (!ofs) {
      return false; // Failed to write the tensor to the file
    }
    ofs.close();
    return true; // Successfully dumped the tensor
  }

  /**
   * Dump tensor element to a file.
   * @param row Row index of the element.
   * @param col Column index of the element.
   * @param file The element tensor will be dumped to this file.
   * @param release_mem Whether to release memory after dump.
   * @return True if the tensor was dumped successfully, false otherwise.
   */
  bool DumpTen(const size_t row, const size_t col, const std::string &file, const bool release_mem = false) {
    std::ofstream ofs(file, std::ofstream::binary);
    if (!ofs) {
      return false; // Failed to open the file
    }
    ofs << (*this)({row, col});
    if (!ofs) {
      return false; // Failed to write the tensor to the file
    }
    ofs.close();
    if (release_mem) {
      this->dealloc(row, col);
    }
    return true; // Successfully dumped the tensor
  }


  /**
   * Extracts a specified column from the TenMatrix as a plain std::vector<Tensor*>.
   * @param col Column index to extract.
   * @return std::vector containing the pointers to the tensor elements from the specified column.
   */
  using DuoMatrix<TenT>::get_col;

  /**
   * Extracts a specified rows_ from the TenMatrix as a plain std::vector<Tensor*>.
   * @param row Row index to extract.
   * @return std::vector containing the pointers to the tensor elements from the specified rows_.
   */
  using DuoMatrix<TenT>::get_row;
};

}

#endif //QLPEPS_TWO_DIM_TN_FRAMEWORK_TEN_MATRIX_H
