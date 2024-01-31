// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-11-06
*
* Description: QuantumLiquids/PEPS project. Implementation for the axis update.
*/

#ifndef GRACEQ_VMC_PEPS_AXIS_UPDATE_H
#define GRACEQ_VMC_PEPS_AXIS_UPDATE_H


namespace qlpeps {

using namespace qlten;

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param split_index_tens
 * @return The indexes order: 0: physical index
 *         other indexes:  remain as before. For the default setting of inputs, the output indexes order is
 *            4
 *            |
 *         1--T--3  and 0 the physical index.
 *            |
 *            2
 * @note the function only suitable for the dense tensor
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GrowPhyIndexFromSplitIndexTensorsWithouQN_(
    const std::vector<QLTensor<TenElemT, QNT> *> &split_index_tens
) {
  const size_t phy_dim = split_index_tens.size();
  const QNT qn0 = split_index_tens[0]->GetIndex(0).GetQNSct(0).GetQn();
  const Index<QNT> pb_out = IndexT({QNSctT(qn0, phy_dim)},
                                   qlten::TenIndexDirType::OUT
  );
  const std::vector<Index<QNT>> &virtual_bond = split_index_tens[0]->GetIndexes();
  std::vector<Index<QNT>> whole_indexes = {pb_out};
  whole_indexes.insert(whole_indexes.end(), virtual_bond.cbegin(), virtual_bond.cend());
  QLTensor<TenElemT, QNT> res(whole_indexes);
  res({0, 0, 0, 0, 0}) = 1.0;
  TenElemT *pres_data = res.GetRawDataPtr();
  size_t stride = split_index_tens[0]->GetActualDataSize();
  for (size_t i = 0; i < phy_dim; i++) {
    hp_numeric::VectorCopy(split_index_tens[i]->GetRawDataPtr(), stride, pres_data + stride);
  }
  return res;
}


}//qlpeps
#endif //GRACEQ_VMC_PEPS_AXIS_UPDATE_H
