// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"

namespace qlpeps {

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::Evolve(const TransferMPO& mpo, const BMPSTruncateParams<RealT> &trunc_para) {
  // MultiplyMPO requires non-const ref due to internal alignment logic; make a local copy
  TransferMPO mpo_copy = mpo;
  bmps_ = bmps_.MultiplyMPO(mpo_copy, trunc_para.compress_scheme,
                            trunc_para.D_min, trunc_para.D_max, trunc_para.trunc_err,
                            trunc_para.convergence_tol,
                            trunc_para.iter_max);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::EvolveStep(const BMPSTruncateParams<RealT> &trunc_para) {
  assert(stack_size_ > 0);
  size_t mpo_num;
  if (pos_ == UP || pos_ == LEFT) {
    // If stack_size is 1 (vacuum), next is row 0. mpo_num = 0.
    mpo_num = stack_size_ - 1;
  } else if (pos_ == DOWN) {
    // If stack_size is 1 (vacuum), next is row rows-1.
    // mpo_num = rows - 1.
    mpo_num = tn_.rows() - stack_size_;
  } else { // RIGHT
    mpo_num = tn_.cols() - stack_size_;
  }

  // Safety check for boundary over-evolution
  if (pos_ == UP && mpo_num >= tn_.rows() - 1) return; 
  if (pos_ == LEFT && mpo_num >= tn_.cols() - 1) return;
  // For DOWN and RIGHT, index goes 0..N-1, so check underflow/bounds if size_t wasn't unsigned
  
  const TransferMPO &mpo = tn_.get_slice(mpo_num, Rotate(Orientation(pos_)));
  Evolve(mpo, trunc_para);
  stack_size_++;
}

template<typename TenElemT, typename QNT>
typename BMPSContractor<TenElemT, QNT>::BMPSWalker 
BMPSContractor<TenElemT, QNT>::GetWalker(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION position) const {
  const auto& stack = bmps_set_.at(position);
  assert(!stack.empty() && "Cannot create Walker from empty BMPS stack");
  // Copy the top BMPS
  return BMPSWalker(tn, stack.back(), position, stack.size());
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::ContractRow(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary) const {
  // Contract <bmps_ | mpo | opposite_boundary> to get a scalar overlap.
  //
  // Tensor network structure (for UP walker, DOWN opposite):
  //   top[col] (UP BMPS tensor)
  //      |
  //   mpo[col] (TN site tensor)  
  //      |
  //   bot[col] (DOWN BMPS tensor)
  //
  // Index conventions:
  //   BMPS tensor (boson): 0:left, 1:physical, 2:right
  //   TN site tensor: 0:left, 1:down, 2:right, 3:up
  //
  // CRITICAL: Storage order differs by direction!
  //   - UP BMPS (bmps_): Reversed. bmps_[0] = rightmost column (col=N-1).
  //   - DOWN BMPS (opposite_boundary): Normal. opposite[0] = leftmost column (col=0).
  //   - site MPO (mpo): Normal. mpo[0] = leftmost column (col=0).
  //
  // Due to reversed UP storage, we contract from col=N-1 to col=0 (right-to-left)
  // to properly match virtual bonds between adjacent BMPS tensors.
  //
  // Connection pattern when contracting column (left) with accumulator (right):
  //   - acc.top_R connects column.top_L (UP BMPS internal bond)
  //   - column.site_R connects acc.site_L (site tensor horizontal bond)
  //   - column.bot_R connects acc.bot_L (DOWN BMPS internal bond)
  //
  // For OBC:
  //   - col=0: left bonds (top_L, site_L, bot_L) are trivial (dim=1)
  //   - col=N-1: right bonds (top_R, site_R, bot_R) are trivial (dim=1)
  
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::ContractRow: Size mismatch. N=" + std::to_string(N) + 
                             ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }
  
  // Only handle UP-DOWN case for horizontal row contraction
  if (pos_ != UP || opposite_boundary.Direction() != DOWN) {
    throw std::runtime_error("BMPSWalker::ContractRow: Unsupported direction pair. "
                             "Walker must be UP, opposite must be DOWN.");
  }
  
  // Build accumulator by contracting RIGHT to LEFT (following UP BMPS storage order)
  // UP BMPS is stored reversed: bmps_[0] = rightmost column (N-1), bmps_[N-1] = leftmost column (0)
  // For UP BMPS: bmps_[i].right (idx2) connects bmps_[i+1].left (idx0)
  // For DOWN BMPS (normal order): opposite[i].right (idx2) connects opposite[i+1].left (idx0)
  //
  // To match both, we contract from col=N-1 to col=0 (right to left)
  // - UP BMPS: bmps_[0] to bmps_[N-1], i.e., storage order
  // - DOWN BMPS: opposite[N-1] to opposite[0], i.e., reverse storage order
  //
  // Each column tensor after contraction has indices:
  //   (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
  // Left bonds: (0, 2, 4), Right bonds: (1, 3, 5)

  Tensor accumulator;
  
  // Contract from col=N-1 (rightmost) to col=0 (leftmost)
  for (size_t i = 0; i < N; ++i) {
    size_t col = N - 1 - i;  // col goes from N-1 down to 0
    
    // UP BMPS: bmps_[i] corresponds to col = N-1-i
    // So for col, we use bmps_[N-1-col] = bmps_[N-1-(N-1-i)] = bmps_[i]
    const Tensor& top = bmps_[i];
    const Tensor& site = *mpo[col];
    const Tensor& bot = opposite_boundary[col];
    
    // Contract top[1] with site[3] (physical bond)
    Tensor top_site;
    qlten::Contract(&top, {1}, &site, {3}, &top_site);
    // top_site indices: (top_L, top_R, site_L, site_D, site_R) = (0, 1, 2, 3, 4)
    
    // Contract top_site[3] with bot[1] (physical bond)
    Tensor column;
    qlten::Contract(&top_site, {3}, &bot, {1}, &column);
    // column indices: (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
    
    if (i == 0) {
      // First iteration (rightmost column): just store the column tensor
      // At OBC right boundary: top_L (idx0), site_R (idx3), bot_R (idx5) are trivial
      accumulator = std::move(column);
    } else if (i == 1) {
      // Second iteration: contract to connect the two columns
      // Connection pattern (col is to the LEFT of acc in space):
      //   - UP BMPS: acc.top_R connects col.top_L (bmps_[i-1].idx2 connects bmps_[i].idx0)
      //   - site: col.site_R connects acc.site_L (site[col].idx2 connects site[col+1].idx0)
      //   - DOWN BMPS: col.bot_R connects acc.bot_L (opposite[col].idx2 connects opposite[col+1].idx0)
      // So: acc{top_R, site_L, bot_L} = acc{1, 2, 4} connects col{top_L, site_R, bot_R} = col{0, 3, 5}
      Tensor new_acc;
      qlten::Contract(&accumulator, {1, 2, 4}, &column, {0, 3, 5}, &new_acc);
      // Result indices: (acc:0, acc:3, acc:5, col:1, col:2, col:4)
      //               = (top_L_first, site_R_first, bot_R_first, top_R_col, site_L_col, bot_L_col)
      //               = (0, 1, 2, 3, 4, 5)
      // For next iteration: left bonds are (3, 4, 5), right bonds are (0, 1, 2)
      accumulator = std::move(new_acc);
    } else {
      // Subsequent iterations: 
      // acc indices: (top_L_first, site_R_first, bot_R_first, top_R_prev, site_L_prev, bot_L_prev)
      //            = (0, 1, 2, 3, 4, 5)
      // Connect: acc{top_R, site_L, bot_L} = acc{3, 4, 5} with col{top_L, site_R, bot_R} = col{0, 3, 5}
      Tensor new_acc;
      qlten::Contract(&accumulator, {3, 4, 5}, &column, {0, 3, 5}, &new_acc);
      // Result indices: (acc:0, acc:1, acc:2, col:1, col:2, col:4)
      //               = (top_L_first, site_R_first, bot_R_first, top_R_col, site_L_col, bot_L_col)
      accumulator = std::move(new_acc);
    }
  }
  
  // After all columns, accumulator has 6 indices:
  // (top_L_0, site_L_0, bot_L_0, top_R_{N-1}, site_R_{N-1}, bot_R_{N-1})
  // For OBC, all 6 indices are trivial (dim=1)
  
  size_t rank = accumulator.Rank();
  if (rank == 0) {
    return accumulator();
  }
  
  // Verify all indices are trivial and extract the scalar
  std::vector<size_t> coords(rank, 0);
  bool all_trivial = true;
  for (size_t i = 0; i < rank; ++i) {
    if (accumulator.GetIndex(i).dim() != 1) {
      all_trivial = false;
#ifndef NDEBUG
      std::cerr << "[ContractRow] Non-trivial index at position " << i 
                << ", dim=" << accumulator.GetIndex(i).dim() 
                << ", total rank=" << rank << std::endl;
#endif
    }
  }
  
  if (!all_trivial) {
    std::string err_msg = "BMPSWalker::ContractRow: Resulting accumulator is not a scalar. "
                          "Residual indices found.\n"
                          "Debug info: N=" + std::to_string(N) + "\n";
    if (N > 0) {
      err_msg += "bmps_[0] rank=" + std::to_string(bmps_[0].Rank()) + "\n" +
                 "opposite[0] rank=" + std::to_string(opposite_boundary[0].Rank());
    }
    throw std::runtime_error(err_msg);
  }
  
  return accumulator.GetElem(coords);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::InitBTenLeft(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary, 
    size_t target_col) {
  using IndexT = qlten::Index<QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::InitBTenLeft: Size mismatch. N=" + std::to_string(N) + 
                             ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }
  
  bten_left_.clear();
  bten_left_col_ = 0;
  
  // Following the same pattern as BMPSContractor::InitBTen(LEFT, row):
  // vacuum BTen connects via:
  //   index0: UP BMPS last tensor (col=0 in reversed) RIGHT index
  //   index1: site tensor at col=0 LEFT index
  //   index2: DOWN BMPS first tensor (col=0) LEFT index
  
  const Tensor& up_ten_col0 = bmps_[N - 1];  // UP reversed: bmps_[N-1] = col 0
  const Tensor& site_col0 = *mpo[0];
  const Tensor& down_ten_col0 = opposite_boundary[0];
  
  // Match InitBTen LEFT pattern: index0 from UP[R], index1 from site[L], index2 from DOWN[L]
  IndexT index0 = qlten::InverseIndex(up_ten_col0.GetIndex(2));    // UP col0 RIGHT
  IndexT index1 = qlten::InverseIndex(site_col0.GetIndex(0));      // site col0 LEFT
  IndexT index2 = qlten::InverseIndex(down_ten_col0.GetIndex(0));  // DOWN col0 LEFT
  
#ifndef NDEBUG
  std::cerr << "[InitBTenLeft] N=" << N << ", target_col=" << target_col << std::endl;
  std::cerr << "[InitBTenLeft] up_ten_col0: "; up_ten_col0.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenLeft] site_col0: "; site_col0.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenLeft] down_ten_col0: "; down_ten_col0.ConciseShow(0); std::cerr << std::endl;
#endif
  
  Tensor vacuum_bten({index0, index1, index2});
  vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  
#ifndef NDEBUG
  std::cerr << "[InitBTenLeft] vacuum_bten: "; vacuum_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_left_.push_back(std::move(vacuum_bten));
  
  // Grow to target_col
  while (bten_left_col_ < target_col && bten_left_col_ < N) {
    GrowBTenLeftStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::InitBTenRight(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary, 
    size_t target_col) {
  using IndexT = qlten::Index<QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::InitBTenRight: Size mismatch. N=" + std::to_string(N) + 
                             ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }
  
  bten_right_.clear();
  bten_right_col_ = N;  // Right edge starts at N (nothing absorbed yet)
  
  // Following the same pattern as BMPSContractor::InitBTen(RIGHT, row):
  // vacuum BTen connects via:
  //   index0: DOWN BMPS last tensor (col=N-1) RIGHT index
  //   index1: site tensor at col=N-1 RIGHT index
  //   index2: UP BMPS first tensor (col=N-1 in reversed) LEFT index
  
  const Tensor& down_ten_colN1 = opposite_boundary[N - 1];
  const Tensor& site_colN1 = *mpo[N - 1];
  const Tensor& up_ten_colN1 = bmps_[0];  // UP reversed: bmps_[0] = col N-1
  
  // Match InitBTen RIGHT pattern
  IndexT index0 = qlten::InverseIndex(down_ten_colN1.GetIndex(2));  // DOWN col=N-1 RIGHT
  IndexT index1 = qlten::InverseIndex(site_colN1.GetIndex(2));      // site col=N-1 RIGHT
  IndexT index2 = qlten::InverseIndex(up_ten_colN1.GetIndex(0));    // UP col=N-1 LEFT
  
#ifndef NDEBUG
  std::cerr << "[InitBTenRight] N=" << N << ", target_col=" << target_col << std::endl;
  std::cerr << "[InitBTenRight] down_ten_colN1: "; down_ten_colN1.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenRight] site_colN1: "; site_colN1.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenRight] up_ten_colN1: "; up_ten_colN1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  Tensor vacuum_bten({index0, index1, index2});
  vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  
#ifndef NDEBUG
  std::cerr << "[InitBTenRight] vacuum_bten: "; vacuum_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_right_.push_back(std::move(vacuum_bten));
  
  // Grow to target_col (towards left)
  // target_col means we want RIGHT BTen to cover (target_col, N-1]
  // So we need bten_right_col_ == target_col + 1
  while (bten_right_col_ > target_col + 1 && bten_right_col_ > 0) {
    GrowBTenRightStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::GrowBTenLeftStep(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary) {
  const size_t N = bmps_.size();

  // Auto-initialize vacuum if empty and at start
  if (bten_left_.empty()) {
    if (bten_left_col_ == 0 && N > 0 && mpo.size() == N && opposite_boundary.size() == N) {
      using IndexT = qlten::Index<QNT>;
      const Tensor& up_ten_col0 = bmps_[N - 1];  // UP reversed: bmps_[N-1] = col 0
      const Tensor& site_col0 = *mpo[0];
      const Tensor& down_ten_col0 = opposite_boundary[0];

      IndexT index0 = qlten::InverseIndex(up_ten_col0.GetIndex(2));    // UP col0 RIGHT
      IndexT index1 = qlten::InverseIndex(site_col0.GetIndex(0));      // site col0 LEFT
      IndexT index2 = qlten::InverseIndex(down_ten_col0.GetIndex(0));  // DOWN col0 LEFT

      Tensor vacuum_bten({index0, index1, index2});
      vacuum_bten({0, 0, 0}) = TenElemT(1.0);
      bten_left_.push_back(std::move(vacuum_bten));
    } else {
      throw std::runtime_error("BMPSWalker::GrowBTenLeftStep: Auto-initialization failed. "
                               "Ensure N > 0 and input dimensions match (mpo=" + 
                               std::to_string(mpo.size()) + ", opposite=" + 
                               std::to_string(opposite_boundary.size()) + ", N=" + std::to_string(N) + ")");
    }
  }

  if (bten_left_col_ >= N) {
    throw std::runtime_error("BMPSWalker::GrowBTenLeftStep: Cannot grow further right. Current col is " + 
                             std::to_string(bten_left_col_) + ", N is " + std::to_string(N));
  }
  
  // Absorb column at bten_left_col_
  const size_t col = bten_left_col_;
  
  // Get tensors for this column (same as GrowFullBTen LEFT case)
  const Tensor& up_mps_ten = bmps_[N - 1 - col];  // UP reversed
  const Tensor& mpo_ten = *mpo[col];
  const Tensor& down_mps_ten = opposite_boundary[col];
  const Tensor& left_bten = bten_left_.back();

#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] col=" << col << std::endl;
  std::cerr << "[GrowBTenLeftStep] up_mps_ten: "; up_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] mpo_ten: "; mpo_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] down_mps_ten: "; down_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] left_bten: "; left_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Use EXACT same contraction pattern as GrowFullBTen LEFT case (bosonic):
  // Contract<TenElemT, QNT, true, true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
  // Contract<TenElemT, QNT, false, false>(tmp1, *mpo[i], 1, 3, 2, tmp2);
  // Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &tmp3);
  
  Tensor tmp1, tmp2, next_bten;
  qlten::Contract<TenElemT, QNT, true, true>(up_mps_ten, left_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] tmp1 after up*bten: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 3, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] tmp2 after tmp1*mpo: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &next_bten);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] next_bten: "; next_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_left_.push_back(std::move(next_bten));
  bten_left_col_++;
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::GrowBTenRightStep(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary) {
  const size_t N = bmps_.size();
  if (bten_right_.empty()) {
    throw std::runtime_error("BMPSWalker::GrowBTenRightStep: Right BTen cache is empty. Call InitBTenRight first.");
  }
  if (bten_right_col_ == 0) {
    throw std::runtime_error("BMPSWalker::GrowBTenRightStep: Cannot grow further left. Current col is 0.");
  }
  
  // Absorb column at bten_right_col_ - 1 (moving left)
  const size_t col = bten_right_col_ - 1;
  
  // Get tensors for this column (same indexing as GrowFullBTen RIGHT case)
  // In GrowFullBTen RIGHT: i-th iteration processes col = N-1-i
  // Here we use col directly, so we need:
  // up_mps_ten = up_bmps[N-1-col] = bmps_[N-1-col] but wait...
  // Actually in GrowFullBTen RIGHT: up_mps_ten = up_bmps[i] where col=N-1-i
  // So when col=N-1, i=0, up_mps_ten = up_bmps[0]
  // Since UP BMPS is reversed: up_bmps[0] corresponds to col=N-1. Correct!
  // When col=N-2, i=1, up_mps_ten = up_bmps[1] corresponds to col=N-2. Correct!
  
  const Tensor& up_mps_ten = bmps_[N - 1 - col];  // UP reversed
  const Tensor& down_mps_ten = opposite_boundary[col];
  const Tensor& mpo_ten = *mpo[col];
  const Tensor& right_bten = bten_right_.back();
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] col=" << col << std::endl;
  std::cerr << "[GrowBTenRightStep] down_mps_ten: "; down_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] mpo_ten: "; mpo_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] up_mps_ten: "; up_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] right_bten: "; right_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Use EXACT same contraction pattern as GrowFullBTen RIGHT case (bosonic):
  // Contract<TenElemT, QNT, true, true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
  // Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
  // Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &tmp3);
  
  Tensor tmp1, tmp2, next_bten;
  qlten::Contract<TenElemT, QNT, true, true>(down_mps_ten, right_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] tmp1 after down*bten: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] tmp2 after tmp1*mpo: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &next_bten);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] next_bten: "; next_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_right_.push_back(std::move(next_bten));
  bten_right_col_--;
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::TraceWithBTen(
    const Tensor& site, 
    size_t site_col, 
    const BMPS<TenElemT, QNT>& opposite_boundary) const {
  const size_t N = bmps_.size();
  
  // Check if BTen caches are available and cover the required columns
  // LEFT BTen should cover [0, site_col), so bten_left_col_ >= site_col
  // RIGHT BTen should cover (site_col, N-1], so bten_right_col_ <= site_col + 1
  if (bten_left_.empty() || bten_right_.empty()) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: BTen caches not initialized. "
                             "Call InitBTenLeft/Right first.");
  }
  
  if (bten_left_col_ < site_col) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Left BTen cache insufficient. "
                             "Current left edge: " + std::to_string(bten_left_col_) + 
                             ", required: " + std::to_string(site_col));
  }

  if (bten_right_col_ > site_col + 1) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Right BTen cache insufficient. "
                             "Current right edge: " + std::to_string(bten_right_col_) + 
                             ", required: " + std::to_string(site_col + 1));
  }
  
  // Get the appropriate BTen
  // bten_left_[k] covers [0, k-1] after k grow steps, so for [0, site_col) we need bten_left_[site_col]
  const Tensor& left_bten = bten_left_[site_col];
  
  // bten_right_[0] is vacuum (after col N-1), bten_right_[k] has absorbed cols [N-k, N-1]
  // For (site_col, N-1], we need to have absorbed cols [site_col+1, N-1]
  // Number of cols = N-1 - site_col = N - 1 - site_col
  // So we need bten_right_[N - 1 - site_col]
  const size_t right_idx = N - 1 - site_col;
  if (right_idx >= bten_right_.size()) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Right BTen index out of bounds. "
                             "Idx: " + std::to_string(right_idx) + 
                             ", Size: " + std::to_string(bten_right_.size()));
  }
  const Tensor& right_bten = bten_right_[right_idx];
  
  // Get tensors at site_col
  const Tensor& up_ten = bmps_[N - 1 - site_col];  // UP reversed
  const Tensor& down_ten = opposite_boundary[site_col];
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] site_col=" << site_col << ", right_idx=" << right_idx << std::endl;
  std::cerr << "[TraceWithBTen] left_bten: "; left_bten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] up_ten: "; up_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] site: "; site.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] down_ten: "; down_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] right_bten: "; right_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // The BTen structure from GrowFullBTen (LEFT case):
  // After contracting up_mps_ten[2] with bten[0], then mpo, then down_mps_ten,
  // the result has indices connecting to the RIGHT side of the current column.
  // So left_bten's indices should connect to column's LEFT sides, not the MPS tensors' L indices.
  //
  // Looking at the contraction in GrowFullBTen more carefully:
  // Contract<true,true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1)
  // This contracts up_mps_ten[R=2] with bten[0].
  // So bten[0] should be compatible with up_mps_ten[R], meaning bten[0] connects from the RIGHT.
  //
  // This is confusing. Let me just follow the same contraction pattern as GrowFullBTen
  // but for calculating the full trace.
  //
  // For a single-column trace, we need to contract:
  // left_bten -- up_mps_ten -- right_bten
  //            |-- mpo_ten --|
  //            -- down_mps_ten --
  //
  // Using GrowFullBTen pattern:
  // Step 1: Contract up_mps_ten with left_bten (same as GrowBTenLeftStep)
  // Step 2: Contract result with mpo_ten
  // Step 3: Contract result with down_mps_ten
  // Step 4: Contract result with right_bten
  
  Tensor tmp1, tmp2, tmp3, result;
  
  // Follow GrowBTenLeftStep pattern exactly for first 3 contractions
  qlten::Contract<TenElemT, QNT, true, true>(up_ten, left_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp1: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, site, 1, 3, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp2: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &down_ten, {0, 1}, &tmp3);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp3: "; tmp3.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Now tmp3 has the same structure as a LEFT BTen after absorbing one column
  // tmp3 indices should match what right_bten expects to contract with
  // right_bten from GrowFullBTen RIGHT case has indices after Contract pattern:
  // Contract<true,true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1)
  // So right_bten[0] connects to down_mps_ten[R]
  // This means tmp3 needs to be contracted with right_bten appropriately
  
  // Contract all remaining indices
  qlten::Contract(&tmp3, {0, 1, 2}, &right_bten, {2, 1, 0}, &result);
  
  // Result should be a scalar
  if (result.Rank() != 0) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Contraction result is not a scalar. Rank: " + 
                             std::to_string(result.Rank()));
  }
  
  return result();
}

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H

