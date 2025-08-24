/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25 
*
* Description: QuantumLiquids/PEPS project. Statistical functions for tensor operations.
*/

#ifndef QLPEPS_VMC_BASIC_STATISTICS_TENSOR_H
#define QLPEPS_VMC_BASIC_STATISTICS_TENSOR_H

#include <vector>
#include <memory>
#include "qlten/qlten.h"
#include "mpi.h"

namespace qlpeps {
using namespace qlten;

/**
 * Calculate mean of a list of tensors
 * @tparam TenElemT tensor element type
 * @tparam QNT quantum number type
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> Mean(const std::vector<QLTensor<TenElemT, QNT> *> &tensor_list,
                             const size_t length) {
  std::vector<TenElemT> coefs(tensor_list.size(), TenElemT(1.0));
  QLTensor<TenElemT, QNT> sum;
  LinearCombine(coefs, tensor_list, TenElemT(0.0), &sum);
  return sum * (1.0 / double(length));
}

/**
 * Calculate MPI distributed mean of a tensor
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> MPIMeanTensor(const QLTensor<TenElemT, QNT> &tensor,
                                      const MPI_Comm &comm) {
  using Tensor = QLTensor<TenElemT, QNT>;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    std::vector<std::unique_ptr<Tensor>> ten_list;
    ten_list.reserve(mpi_size);
    
    // Gather tensors from all processes
    for (size_t proc = 0; proc < mpi_size; proc++) {
      if (proc != qlten::hp_numeric::kMPIMasterRank) {
        ten_list.emplace_back(std::make_unique<Tensor>());
        ten_list.back()->MPI_Recv(proc, 2 * proc, comm);
      } else {
        ten_list.emplace_back(std::make_unique<Tensor>(tensor));
      }
    }
    
    // Convert to raw pointers for Mean calculation
    std::vector<Tensor*> tensor_ptrs;
    tensor_ptrs.reserve(mpi_size);
    for (const auto& ptr : ten_list) {
      tensor_ptrs.push_back(ptr.get());
    }
    
    return Mean(tensor_ptrs, mpi_size);
  } else {
    tensor.MPI_Send(qlten::hp_numeric::kMPIMasterRank, 2 * rank, comm);
    return Tensor();
  }
}

/**
 * Calculate variance of a list of tensors
 * @note TODO: Implement if needed
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> Variance(const std::vector<QLTensor<TenElemT, QNT> *> &tensor_list,
                                 const size_t length) {
  // TODO: Implement variance calculation
  return QLTensor<TenElemT, QNT>();
}

} // namespace qlpeps

#endif // QLPEPS_VMC_BASIC_STATISTICS_TENSOR_H