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
  SplitIndexTPS(const size_t rows, const size_t cols) : TenMatrix<std::vector<Tensor>>(rows, cols) {}

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

  using TenMatrix<std::vector<Tensor>>::operator=;

  // TODO
  operator TPST() {
    TPST tps(this->rows(), this->cols());

  }

  TN2D Project(const Configuration &config) const {
    assert(config.rows() == this->rows());
    assert(config.cols() == this->cols());
    return TN2D((*this), config);
  }

  size_t GetMaxBondDimension(void) const {
    size_t dmax = 0;
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        const Tensor &tensor = (*this)({row, col})[0];
        dmax = std::max(dmax, tensor.GetShape()[0]);
        dmax = std::max(dmax, tensor.GetShape()[1]);
      }
    }
    return dmax;
  };

  bool DumpTen(const size_t row, const size_t col, const size_t compt, const std::string &file) const {
    std::ofstream ofs(file, std::ofstream::binary);
    if (!ofs) {
      return false; // Failed to open the file
    }
    ofs << (*this)({row, col})[compt];
    if (!ofs) {
      return false; // Failed to write the tensor to the file
    }
    ofs.close();
    return true; // Successfully dumped the tensor
  }


  ///< assume has alloc memory
  bool LoadTen(const size_t row, const size_t col, const size_t compt, const std::string &file) {
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) {
      return false; // Failed to open the file
    }
    ifs >> (*this)({row, col})[compt];
    if (!ifs) {
      return false; // Failed to read the tensor from the file
    }
    ifs.close();
    return true; // Successfully loaded the tensor
  }

  // note don't confilt the name of tensors with original tps
  void Dump(const std::string &tps_path = kTpsPath, const bool release_mem = false);

  bool Load(const std::string &tps_path = kTpsPath);
};

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::Dump(const std::string &tps_path, const bool release_mem) {
  if (!gqmps2::IsPathExist(tps_path)) { gqmps2::CreatPath(tps_path); }
  std::string file;
  const size_t phy_dim = (*this)({0, 0}).size();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      for (size_t compt = 0; compt < phy_dim; compt++) {
        file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        this->DumpTen(row, col, compt, file);
      }
      if (release_mem) {
        this->dealloc(row, col);
      }
    }
  }
  file = tps_path + "/" + "phys_dim";
  std::ofstream ofs(file, std::ofstream::binary);
  ofs << phy_dim;
  ofs.close();
}

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::Load(const std::string &tps_path) {

  if (!gqmps2::IsPathExist(tps_path)) {
    std::cout << "No path " << tps_path << std::endl;
    return false;
  }
  std::string file = tps_path + "/" + "phys_dim";
  std::ifstream ifs(file, std::ofstream::binary);
  size_t phy_dim(0);
  ifs >> phy_dim;
  ifs.close();
  if (phy_dim == 0) {
    return false;
  }
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t compt = 0; compt < phy_dim; compt++) {
        file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        if (!(this->LoadTen(row, col, compt, file))) {
          return false;
        }
      }
    }
  }
  return true;
}
}


#endif //GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H
