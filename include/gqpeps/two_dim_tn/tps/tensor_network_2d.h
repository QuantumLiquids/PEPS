// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-24
*
* Description: GraceQ/VMC-PEPS project. The 2-dimensional tensor network class.
*/

#ifndef VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
#define VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "gqpeps/basic.h"           //TruncatePara
namespace gqpeps {
using namespace gqten;

using BTenPOSITION = BMPSPOSITION;

/**
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<GQTensor<TenElemT, QNT>> {
  using Tensor = GQTensor<TenElemT, QNT>;
  using TransferMPO = std::vector<Tensor *>;
  using BMPST = BMPS<TenElemT, QNT>;
 public:

  const std::vector<BMPS<TenElemT, QNT>> &GetBMPS(const BMPSPOSITION position) const {
    return position_bmps_set_map[position];
  }

  void GrowBMPSStep(const BMPSPOSITION position) {
    std::vector<BMPS<TenElemT, QNT>> &bmps_set = position_bmps_set_map[position];
    size_t existed_bmps_size = bmps_set.size();
    assert(existed_bmps_size > 0);
    const BMPS<TenElemT, QNT> &last_bmps = bmps_set.back();
    size_t rows = this->rows();
    size_t cols = this->cols();
    switch (position) {
      case DOWN: {
        const TransferMPO &mpo = this->get_row(rows - existed_bmps_size);
        bmps_set.push_back(last_bmps.MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        break;
      }
      case UP: {
        const TransferMPO &mpo = this->get_row(existed_bmps_size - 1);
        bmps_set.push_back(last_bmps.MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        break;
      }
      case LEFT: {
        const TransferMPO &mpo = this->get_col(existed_bmps_size - 1);
        bmps_set.push_back(last_bmps.MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        break;
      }
      case RIGHT: {
        const TransferMPO &mpo = this->get_col(cols - existed_bmps_size);
        bmps_set.push_back(last_bmps.MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        break;
      }
    }
  }

  void GrowBMPSForRow(const size_t row) {
    const size_t rows = this->rows();
    std::vector<BMPS<TenElemT, QNT>> &bmps_set_down = position_bmps_set_map[DOWN];
    for (size_t row_bmps = rows - bmps_set_down.size(); row_bmps > row; row_bmps--) {
      const TransferMPO &mpo = this->get_row(row_bmps);
      bmps_set_down.push_back(bmps_set_down.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
    }

    std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = position_bmps_set_map[UP];
    for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
      const TransferMPO &mpo = this->get_row(row_bmps);
      bmps_set_up.push_back(bmps_set_up.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
    }
  }

  const std::pair<BMPST &, BMPST &> GetBoundaryMPSForRow(const size_t row) {
    const size_t rows = this->rows();
    GrowBMPSForRow(row);
    BMPST &down_bmps = position_bmps_set_map[DOWN][rows - 1 - row];
    BMPST &up_bmps = position_bmps_set_map[UP][row];

    return std::make_pair(down_bmps, up_bmps);
  }

  void GrowFullBMPS(const BMPSPOSITION position) {
    std::vector<BMPS<TenElemT, QNT>> &bmps_set = position_bmps_set_map[position];
    size_t existed_bmps_size = bmps_set.size();
    assert(existed_bmps_size > 0);
    size_t rows = this->rows();
    size_t cols = this->cols();
    switch (position) {
      case DOWN: {
        for (size_t row = rows - existed_bmps_size; row > 0; row--) {
          const TransferMPO &mpo = this->get_row(row);
          bmps_set.push_back(bmps_set.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        }
        break;
      }
      case UP: {
        for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
          const TransferMPO &mpo = this->get_row(row);
          bmps_set.push_back(bmps_set.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        }
        break;
      }
      case LEFT: {
        for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
          const TransferMPO &mpo = this->get_col(col);
          bmps_set.push_back(bmps_set.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        }
      }
      case RIGHT: {
        for (size_t col = cols - existed_bmps_size; col > 0; col--) {
          const TransferMPO &mpo = this->get_col(col);
          bmps_set.push_back(bmps_set.back().MultipleMPO(mpo, truncate_para.D_min, truncate_para.D_max));
        }
        break;
      }
    }
  }

  std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> position_bmps_set_map;
  std::map<BTenPOSITION, std::vector<Tensor>> position_bten_set_map;
  TruncatePara truncate_para;

 private:

};

}//gqpeps

#endif //VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
