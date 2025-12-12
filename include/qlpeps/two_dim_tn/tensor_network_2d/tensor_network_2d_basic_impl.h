// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class, implementation.
*/


#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H

#include <stdexcept>
#include <string>

namespace qlpeps {

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const size_t rows, const size_t cols, const BoundaryCondition bc)
    : TenMatrix<qlten::QLTensor<TenElemT, QNT>>(rows, cols), boundary_condition_(bc) {
}

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config)
    :TensorNetwork2D(tps.rows(), tps.cols(), tps.GetBoundaryCondition()) {
  assert(config.rows() == tps.rows());
  assert(config.cols() == tps.cols());
  
  // Input validation - detect uninitialized TPS early
  for (size_t row = 0; row < tps.rows(); row++) {
    for (size_t col = 0; col < tps.cols(); col++) {
      const auto& site_tensors = tps({row, col});
      
      size_t config_val = config({row, col});
      
      // Check for empty tensor vector first
      if (site_tensors.empty()) {
        throw std::invalid_argument("TensorNetwork2D: TPS at site (" + std::to_string(row) + 
                                   ", " + std::to_string(col) + ") has empty tensor vector. " +
                                   "TPS must be properly initialized before creating TensorNetwork2D.");
      }
      
      // Check configuration bounds
      if (config_val >= site_tensors.size()) {
        throw std::out_of_range("TensorNetwork2D: Configuration value " + std::to_string(config_val) + 
                               " at site (" + std::to_string(row) + ", " + std::to_string(col) + 
                               ") exceeds TPS physical dimension " + std::to_string(site_tensors.size()));
      }
      
      // Check if the specific tensor component needed is default
      if (site_tensors[config_val].IsDefault()) {
        throw std::invalid_argument("TensorNetwork2D: TPS tensor at site (" + std::to_string(row) + 
                                   ", " + std::to_string(col) + ") component " + std::to_string(config_val) + 
                                   " is default (uninitialized). TPS must be properly initialized before creating TensorNetwork2D.");
      }
      
#ifndef NDEBUG
      (*this)({row, col}) = tps({row, col}).at(config({row, col}));
#else
      (*this)({row, col}) = tps({row, col})[config({row, col})];
#endif
    }
  }
}

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT> &TensorNetwork2D<TenElemT, QNT>::operator=(const TensorNetwork2D<TenElemT, QNT> &tn) {
  TenMatrix<Tensor>::operator=(tn);
  boundary_condition_ = tn.boundary_condition_;
  return *this;
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::UpdateSiteTensor(const qlpeps::SiteIdx &site, const size_t update_config,
                                                      const SplitIndexTPS<TenElemT, QNT> &sitps) {
  (*this)(site) = sitps(site)[update_config];
}

}///qlpeps

#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H

