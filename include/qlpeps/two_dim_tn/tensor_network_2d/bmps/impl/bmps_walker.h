// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_helpers.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bten_operations.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"

namespace qlpeps {

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::Evolve(const TransferMPO& mpo) {
  // MultiplyMPO requires non-const ref due to internal alignment logic; make a local copy
  TransferMPO mpo_copy = mpo;
  bmps_ = bmps_.MultiplyMPO(mpo_copy, trunc_params_.compress_scheme,
                            trunc_params_.D_min, trunc_params_.D_max, trunc_params_.trunc_err,
                            trunc_params_.convergence_tol,
                            trunc_params_.iter_max);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::EvolveStep() {
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
  Evolve(mpo);
  stack_size_++;
}

template<typename TenElemT, typename QNT>
typename BMPSContractor<TenElemT, QNT>::BMPSWalker 
BMPSContractor<TenElemT, QNT>::GetWalker(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION position) const {
  const auto& stack = bmps_set_.at(position);
  assert(!stack.empty() && "Cannot create Walker from empty BMPS stack");
  // Copy the top BMPS
  return BMPSWalker(tn, stack.back(), position, stack.size(), GetTruncateParams());
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
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::InitBTenLeft: Size mismatch. N=" + std::to_string(N) + 
                             ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }
  
  bten_left_.clear();
  bten_left_col_ = 0;
  
  // Create vacuum BTen using shared function
  // UP BMPS reversed: bmps_[N-1] = col 0
  Tensor vacuum = bten_ops::CreateVacuumBTenLeft<TenElemT, QNT>(
      bmps_[N - 1], *mpo[0], opposite_boundary[0]);
  bten_left_.push_back(std::move(vacuum));
  
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
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::InitBTenRight: Size mismatch. N=" + std::to_string(N) + 
                             ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }
  
  bten_right_.clear();
  bten_right_col_ = N;  // Right edge starts at N (nothing absorbed yet)
  
  // Create vacuum BTen using shared function
  // UP BMPS reversed: bmps_[0] = col N-1
  Tensor vacuum = bten_ops::CreateVacuumBTenRight<TenElemT, QNT>(
      opposite_boundary[N - 1], *mpo[N - 1], bmps_[0]);
  bten_right_.push_back(std::move(vacuum));
  
  // Grow to target_col (towards left)
  while (bten_right_col_ > target_col + 1 && bten_right_col_ > 0) {
    GrowBTenRightStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::GrowBTenLeftStep(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary) {
  const size_t N = bmps_.size();

  // Validate inputs
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    throw std::runtime_error("BMPSWalker::GrowBTenLeftStep: Size mismatch. N=" + 
                             std::to_string(N) + ", mpo=" + std::to_string(mpo.size()) + 
                             ", opposite=" + std::to_string(opposite_boundary.size()));
  }

  // Auto-initialize vacuum BTen if empty
  if (bten_left_.empty()) {
    if (bten_left_col_ != 0) {
      throw std::runtime_error("BMPSWalker::GrowBTenLeftStep: bten_left_ is empty but col != 0");
    }
    // Use shared function to create vacuum BTen
    // UP BMPS reversed: bmps_[N-1] = col 0
    Tensor vacuum = bten_ops::CreateVacuumBTenLeft<TenElemT, QNT>(
        bmps_[N - 1], *mpo[0], opposite_boundary[0]);
    bten_left_.push_back(std::move(vacuum));
  }

  if (bten_left_col_ >= N) {
    throw std::runtime_error("BMPSWalker::GrowBTenLeftStep: Cannot grow beyond N. col=" + 
                             std::to_string(bten_left_col_) + ", N=" + std::to_string(N));
  }
  
  // Absorb column using shared function
  const size_t col = bten_left_col_;
  Tensor next_bten = bten_ops::GrowBTenLeftStep<TenElemT, QNT>(
      bten_left_.back(),
      bmps_[N - 1 - col],    // UP reversed
      *mpo[col],
      opposite_boundary[col]
  );
  
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
    throw std::runtime_error("BMPSWalker::GrowBTenRightStep: Cannot grow further left. col is already 0.");
  }
  
  // Absorb column at bten_right_col_ - 1 (moving left)
  const size_t col = bten_right_col_ - 1;
  
  // Use shared function
  // UP BMPS reversed: bmps_[N-1-col] corresponds to physical col
  Tensor next_bten = bten_ops::GrowBTenRightStep<TenElemT, QNT>(
      bten_right_.back(),
      opposite_boundary[col],  // DOWN
      *mpo[col],               // site
      bmps_[N - 1 - col]       // UP reversed
  );
  
  bten_right_.push_back(std::move(next_bten));
  bten_right_col_--;
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::ShiftBTenWindow(
    const TransferMPO& mpo,
    const BMPS<TenElemT, QNT>& opposite_boundary,
    BTenPOSITION position) {
  if (position == LEFT) {
    // Shift window to the left: pop left, grow right
    if (bten_left_.empty()) {
      throw std::runtime_error("BMPSWalker::ShiftBTenWindow: Left BTen cache is empty.");
    }
    bten_left_.pop_back();
    bten_left_col_--;
    GrowBTenRightStep(mpo, opposite_boundary);
  } else {  // RIGHT
    // Shift window to the right: pop right, grow left
    if (bten_right_.empty()) {
      throw std::runtime_error("BMPSWalker::ShiftBTenWindow: Right BTen cache is empty.");
    }
    bten_right_.pop_back();
    bten_right_col_++;
    GrowBTenLeftStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::TraceWithBTen(
    const Tensor& site, 
    size_t site_col, 
    const BMPS<TenElemT, QNT>& opposite_boundary) const {
  const size_t N = bmps_.size();
  
  // Validate BTen caches
  if (bten_left_.empty() || bten_right_.empty()) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: BTen caches not initialized.");
  }
  
  if (bten_left_col_ < site_col) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Left BTen insufficient. "
                             "edge=" + std::to_string(bten_left_col_) + ", need=" + std::to_string(site_col));
  }

  if (bten_right_col_ > site_col + 1) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Right BTen insufficient. "
                             "edge=" + std::to_string(bten_right_col_) + ", need=" + std::to_string(site_col + 1));
  }
  
  // Get BTen at correct positions
  const Tensor& left_bten = bten_left_[site_col];
  const size_t right_idx = N - 1 - site_col;
  if (right_idx >= bten_right_.size()) {
    throw std::runtime_error("BMPSWalker::TraceWithBTen: Right BTen index out of bounds.");
  }
  const Tensor& right_bten = bten_right_[right_idx];
  
  // Use shared function for trace computation
  return bten_ops::TraceBTen<TenElemT, QNT>(
      bmps_[N - 1 - site_col],    // UP reversed
      left_bten,
      site,
      opposite_boundary[site_col], // DOWN
      right_bten
  );
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::TraceWithTwoSiteBTen(
    const Tensor& site_a, const Tensor& site_b, size_t site_col,
    const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary) const {
  const size_t N = bmps_.size();
  
  // Validate site_col
  if (site_col + 1 >= N) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: site_col+1 out of bounds.");
  }
  
  // Validate BTen caches
  if (bten_left_.empty() || bten_right_.empty()) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: BTen caches not initialized.");
  }
  
  // LEFT BTen must cover [0, site_col)
  if (bten_left_col_ < site_col) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: Left BTen insufficient. "
                             "edge=" + std::to_string(bten_left_col_) + ", need=" + std::to_string(site_col));
  }
  
  // RIGHT BTen must cover (site_col+1, N-1]
  if (bten_right_col_ > site_col + 2) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: Right BTen insufficient. "
                             "edge=" + std::to_string(bten_right_col_) + ", need=" + std::to_string(site_col + 2));
  }
  
  // Get LEFT BTen at site_col (covers [0, site_col))
  if (site_col >= bten_left_.size()) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: Left BTen index out of bounds.");
  }
  const Tensor& left_bten = bten_left_[site_col];
  
  // Get RIGHT BTen at site_col+1 (covers (site_col+1, N-1])
  const size_t right_idx = N - 2 - site_col;  // = N - 1 - (site_col + 1)
  if (right_idx >= bten_right_.size()) {
    throw std::runtime_error("BMPSWalker::TraceWithTwoSiteBTen: Right BTen index out of bounds. "
                             "idx=" + std::to_string(right_idx) + ", size=" + std::to_string(bten_right_.size()));
  }
  const Tensor& right_bten = bten_right_[right_idx];
  
  // Contract: left_bten + up[col] + site_a + down[col] -> intermediate
  const size_t col_a = site_col;
  const size_t col_b = site_col + 1;
  
  Tensor tmp[4];
  
  // Contract left_bten with up[col_a], site_a, down[col_a]
  Tensor site_a_copy = site_a;
  Tensor intermediate = bten_ops::GrowBTenLeftStep<TenElemT, QNT>(
      left_bten,
      bmps_[N - 1 - col_a],    // UP reversed
      site_a_copy,
      opposite_boundary[col_a] // DOWN
  );
  
  // Contract intermediate with up[col_b], site_b, down[col_b], right_bten
  return bten_ops::TraceBTen<TenElemT, QNT>(
      bmps_[N - 1 - col_b],     // UP reversed
      intermediate,
      site_b,
      opposite_boundary[col_b], // DOWN
      right_bten
  );
}

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_WALKER_H
