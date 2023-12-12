/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-06
*
* Description: GraceQ/VMC-PEPS project. The split index TPS class, where the tensor are stored in tensors splited in physical index
*/

#ifndef GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H
#define GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqpeps/two_dim_tn/tps/tps.h"                  // TPS

namespace gqpeps {
using namespace gqten;

// Helpers
inline std::string GenSplitIndexTPSTenName(const std::string &tps_path,
                                           const size_t row, const size_t col,
                                           const size_t compt) {
  return tps_path + "/" +
      kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "_" + std::to_string(compt) + "." +
      kGQTenFileSuffix;
}

template<typename TenElemT, typename QNT>
class SplitIndexTPS : public TenMatrix<std::vector<GQTensor<TenElemT, QNT>>> {
  using Tensor = GQTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using TN2D = TensorNetwork2D<TenElemT, QNT>;
 public:

  //constructor
  SplitIndexTPS(void) = default;    // default constructor for MPI

  SplitIndexTPS(const size_t rows, const size_t cols) : TenMatrix<std::vector<Tensor>>(rows, cols) {}

  SplitIndexTPS(const size_t rows, const size_t cols, const size_t phy_dim) : SplitIndexTPS(rows, cols) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      }
    }
  }

  SplitIndexTPS(const SplitIndexTPS &brotps) : TenMatrix<std::vector<GQTensor<TenElemT, QNT>>>(brotps) {}

  SplitIndexTPS(const TPST &tps) : TenMatrix<std::vector<Tensor>>(tps.rows(), tps.cols()) {
    const size_t phy_idx = 4;
    for (size_t row = 0; row < tps.rows(); row++) {
      for (size_t col = 0; col < tps.cols(); col++) {
        const Index<QNT> local_hilbert = tps({row, col}).GetIndex(phy_idx);
        const auto local_hilbert_inv = InverseIndex(local_hilbert);
        const size_t dim = local_hilbert.dim();
        (*this)({row, col}) = std::vector<Tensor>(dim);
        for (size_t i = 0; i < dim; i++) {
          Tensor project_ten = Tensor({local_hilbert_inv});
          project_ten({i}) = TenElemT(1.0);
          Contract(tps(row, col), {phy_idx}, &project_ten, {0}, &(*this)({row, col})[i]);
        }
      }
    }
  }

//  using TenMatrix<std::vector<Tensor>>::operator=;
  ///< using below explicitly definition to be compatible with lower version of g++
  SplitIndexTPS &operator=(const SplitIndexTPS &rhs) {
    TenMatrix<std::vector<Tensor>>::operator=(rhs);
    return *this;
  }

  // TODO
  operator TPST() {
    TPST tps(this->rows(), this->cols());

  }

  size_t PhysicalDim(const SiteIdx &site = {0, 0}) const {
    return (*this)(site).size();
  }

  TN2D Project(const Configuration &config) const {
    assert(config.rows() == this->rows());
    assert(config.cols() == this->cols());
    return TN2D((*this), config);
  }

  size_t GetMinBondDimension(void) const;
  size_t GetMaxBondDimension(void) const;

  std::pair<size_t, size_t> GetMinMaxBondDimension(void) const;

  std::vector<size_t> GetAllInnerBondDimensions(void) const;

  bool IsBondDimensionEven(void) const;

  SplitIndexTPS operator*(const TenElemT scalar) const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i] * scalar;
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator*=(const TenElemT scalar) {
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            (*this)({row, col})[i] *= scalar;
        }
      }
    }
    return *this;
  }

  SplitIndexTPS operator+(const SplitIndexTPS &right) const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault() && !right({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i] + right({row, col})[i];
          else if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i];
          else if (!right({row, col})[i].IsDefault())
            res({row, col})[i] = right({row, col})[i];
        }
      }
    }
    return res;
  }

  SplitIndexTPS &operator+=(const SplitIndexTPS &right) {
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if ((*this)({row, col})[i].IsDefault())
            (*this)({row, col})[i] = right({row, col})[i];
          else if (!right({row, col})[i].IsDefault())
            (*this)({row, col})[i] += right({row, col})[i];
        }
      }
    }
    return *this;
  }

  ///< Inner product
  TenElemT operator*(const SplitIndexTPS &right) const {
    TenElemT res(0);
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if ((*this)({row, col})[i].IsDefault() || right({row, col})[i].IsDefault()) {
            continue;
          }
          Tensor ten_dag = Dag((*this)({row, col})[i]);
          Tensor scalar;
          Contract(&ten_dag, {0, 1, 2, 3}, &right({row, col})[i], {0, 1, 2, 3}, &scalar);
          res += scalar();
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator-() const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = -(*this)({row, col})[i];
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator-(const SplitIndexTPS &right) const {
    return (*this) + (-right);
  }

  ///< NB! not the wave function norm, it's the summation of tensor norm square.
  ///< definition: summation of tensor element squares
  double NormSquare() const;

  /**
   * Normalize the site tensors by treating all tensors in one site as one tps tensor.
   * @param site
   * @return
   */
  double NormalizeSite(const SiteIdx &site);

  ///< normalize all site by function NormalizeSite.
  void NormalizeAllSite();

  bool DumpTen(const size_t row, const size_t col, const size_t compt, const std::string &file) const;

  ///< assume has alloc memory
  bool LoadTen(const size_t row, const size_t col, const size_t compt, const std::string &file);

  void Dump(const std::string &tps_path = kTpsPath, const bool release_mem = false);

  bool Load(const std::string &tps_path = kTpsPath);
};

}//gqpeps

#include "gqpeps/two_dim_tn/tps/split_index_tps_impl.h"
#endif //GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H
