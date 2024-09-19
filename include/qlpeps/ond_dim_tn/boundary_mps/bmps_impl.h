/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-01
*
* Description: QuantumLiquids/PEPS project. Boundary MPS
*/

#ifndef QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H
#define QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H

namespace qlpeps {
using namespace qlten;
using namespace qlmps;

///< Initial the bmps of the boundary of the OBC tensor network
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>::BMPS(const BMPSPOSITION position, const size_t size,
                          const BMPS::IndexT &local_hilbert_space) :
    TenVec<Tensor>(size),
    position_(position),
    center_(0),
    tens_cano_type_(size, MPSTenCanoType::RIGHT) {
  assert(local_hilbert_space.dim() == 1);
  if constexpr (Tensor::IsFermionic()) {
    assert(local_hilbert_space.GetQNSct(0).GetQn().IsFermionParityEven());
    Tensor mps_ten = Tensor({index0_in_, local_hilbert_space, index0_out_, index0_out_});
    mps_ten({0, 0, 0, 0}) = 1.0;
    for (size_t i = 0; i < size; i++) {
      this->alloc(i);
      (*this)[i] = mps_ten;
    }
  } else { //bosonic
    Tensor mps_ten = Tensor({index0_in_, local_hilbert_space, index0_out_});
    mps_ten({0, 0, 0}) = 1.0;
    for (size_t i = 0; i < size; i++) {
      this->alloc(i);
      (*this)[i] = mps_ten;
    }
  }
}

///< Initial the bmps of the boundary of the OBC tensor network
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>::BMPS(const BMPSPOSITION position,
                          const std::vector<BMPS::IndexT> &hilbert_spaces) :
    TenVec<Tensor>(hilbert_spaces.size()),
    position_(position),
    center_(0),
    tens_cano_type_(hilbert_spaces.size(), MPSTenCanoType::RIGHT) {
  if constexpr (Tensor::IsFermionic()) {
    for (size_t i = 0; i < hilbert_spaces.size(); i++) {
      assert(hilbert_spaces.at(i).dim() == 1);
      assert(hilbert_spaces.at(i).GetQNSct(0).GetQn().IsFermionParityEven());
      this->alloc(i);
      (*this)[i] = Tensor({index0_in_, hilbert_spaces.at(i), index0_out_, index0_out_});
      (*this)[i]({0, 0, 0, 0}) = 1.0;
    }
  } else {
    for (size_t i = 0; i < hilbert_spaces.size(); i++) {
      this->alloc(i);
      (*this)[i] = Tensor({index0_in_, hilbert_spaces.at(i), index0_out_});
      assert(hilbert_spaces.at(i).dim() == 1);
      (*this)[i]({0, 0, 0}) = 1.0;
    }
  }
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) {
  return DuoVector<Tensor>::operator[](idx);
}

template<typename TenElemT, typename QNT>
const QLTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) const {
  return DuoVector<Tensor>::operator[](idx);
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *&BMPS<TenElemT, QNT>::operator()(const size_t idx) {
  return DuoVector<Tensor>::operator()(idx);
}

template<typename TenElemT, typename QNT>
const QLTensor<TenElemT, QNT> *BMPS<TenElemT, QNT>::operator()(const size_t idx) const {
  return DuoVector<Tensor>::operator()(idx);
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::Centralize(const int target_center) {
  assert(target_center >= 0);
  auto mps_tail_idx = this->size() - 1;
  if (target_center != 0) { LeftCanonicalize(target_center - 1); }
  if (target_center != mps_tail_idx) {
    RightCanonicalize(target_center + 1);
  }
  center_ = target_center;
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::LeftCanonicalize(const size_t stop_idx) {
  size_t start_idx;
  for (size_t i = 0; i <= stop_idx; ++i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::LEFT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are left canonical, do nothing.
  }
  for (size_t i = start_idx; i <= stop_idx; ++i) { LeftCanonicalizeTen(i); }
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::RightCanonicalize(const size_t stop_idx) {
  auto mps_tail_idx = this->size() - 1;
  size_t start_idx;
  for (size_t i = mps_tail_idx; i >= stop_idx; --i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::RIGHT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are right canonical, do nothing.
  }
  for (size_t i = start_idx; i >= stop_idx; --i) { RightCanonicalizeTen(i); }
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::LeftCanonicalizeTen(const size_t site_idx) {
  assert(site_idx < this->size() - 1);
  size_t ldims((*this)(site_idx)->Rank() - 1);
  auto pq = new Tensor;
  Tensor r;
  if constexpr (Tensor::IsFermionic()) {
    (*this)(site_idx)->Transpose({0, 1, 3, 2});
  }
  QR((*this)(site_idx), ldims, Div((*this)[site_idx]), pq, &r);
  delete (*this)(site_idx);
  (*this)(site_idx) = pq;
  if constexpr (Tensor::IsFermionic()) {
    (*this)(site_idx)->Transpose({0, 1, 3, 2});
  }
  auto pnext_ten = new Tensor;
  Contract(&r, (*this)(site_idx + 1), {{1}, {0}}, pnext_ten);
  delete (*this)(site_idx + 1);
  (*this)(site_idx + 1) = pnext_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::LEFT;
  tens_cano_type_[site_idx + 1] = MPSTenCanoType::NONE;
}

template<typename TenElemT, typename QNT>
QLTensor<QLTEN_Double, QNT> BMPS<TenElemT, QNT>::RightCanonicalizeTen(const size_t site_idx) {
  ///< TODO: using LU decomposition
  assert(site_idx > 0);
  size_t ldims = 1;
  Tensor u;
  QLTensor<QLTEN_Double, QNT> s;
  auto pvt = new Tensor;
  auto qndiv = Div((*this)[site_idx]);
  mock_qlten::SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  Tensor temp_ten;
  Contract(&u, &s, {{1}, {0}}, &temp_ten);
  std::vector<std::vector<size_t>> ctrct_axes = {{2}, {0}};
  auto pprev_ten = new Tensor;
  Contract((*this)(site_idx - 1), &temp_ten, ctrct_axes, pprev_ten);
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = pprev_ten;
  if constexpr (Tensor::IsFermionic()) {
    (*this)(site_idx)->Transpose({0, 1, 3, 2});
  }
  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
  return s;
}

template<typename TenElemT, typename QNT>
std::pair<size_t, double>
BMPS<TenElemT, QNT>::RightCanonicalizeTruncate(const size_t site, const size_t Dmin,
                                               const size_t Dmax, const double trunc_err) {

  QLTensor<QLTEN_Double, QNT> s;
  auto pvt = new Tensor;
  Tensor u;
  double actual_trunc_err;
  size_t D;
  SVD((*this)(site),
      1, qn0_, trunc_err, Dmin, Dmax,
      &u, &s, pvt, &actual_trunc_err, &D
  );
//  std::cout << "Truncate MPS bond " << std::setw(4) << site
//            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
//            << " D = " << std::setw(5) << D;
//  std::cout << std::scientific << std::endl;

  delete (*this)(site);
  (*this)(site) = pvt;

  Tensor temp_ten;
  Contract(&u, &s, {{1}, {0}}, &temp_ten);
  auto pnext_ten = new Tensor;
  Contract((*this)(site - 1), &temp_ten, {{2}, {0}}, pnext_ten);
  delete (*this)(site - 1);
  (*this)(site - 1) = pnext_ten;
  if constexpr (Tensor::IsFermionic()) {
    (*this)(site - 1)->Transpose({0, 1, 3, 2});
  }
  //set ten canonical type
  return std::make_pair(D, actual_trunc_err);
}

template<typename TenElemT, typename QNT>
std::vector<double> BMPS<TenElemT, QNT>::GetEntanglementEntropy(size_t
                                                                n) {
  size_t N = this->size();
  std::vector<double> ee_list(N - 1);
  Centralize(N
                 - 1);
  (*this)[N - 1].
      Normalize();

  for (
      size_t i = N - 1;
      i >= 1; --i) { // site
    auto s = RightCanonicalizeTen(i);
    double ee = 0;
    double sum_of_p2n = 0.0;
    for (
        size_t k = 0;
        k < s.
            GetShape()[0];
        ++k) { // indices of singular value matrix
      double singular_value = s(k, k);
      double p = singular_value * singular_value;
      if (n == 1) {
        ee += (-
            p * std::log(p)
        );
      } else {
        double p_to_n = p;
        for (
            size_t j = 1;
            j < n;
            j++) { // order of power
          p_to_n *=
              p;
        }
        sum_of_p2n +=
            p_to_n;
      }
    }
    if (n == 1) {
      ee_list[i - 1] =
          ee;
    } else {
      ee_list[i - 1] = -
          std::log(sum_of_p2n)
          / (double) (n - 1);
// note above formula must be in form of n-1 rather 1-n because of n is type of size_t
    }
  }
  return
      ee_list;
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::Reverse() {
  size_t N = this->size();
  if (N % 2 == 0) {
    for (size_t i = 0; i < N / 2; i++) {
      std::swap((*this)(i), (*this)(N - 1 - i));
    }
  } else {
    for (size_t i = 0; i < (N - 1) / 2; i++) {
      std::swap((*this)(i), (*this)(N - 1 - i));
    }
  }
  for (size_t i = 0; i < N; i++) {
    if constexpr (Tensor::IsFermionic()) {
      (*this)(i)->Transpose({2, 1, 0, 3});
    } else {
      (*this)(i)->Transpose({2, 1, 0});
    }
  }
  if (center_ != kUncentralizedCenterIdx) {
    center_ = N - 1 - center_;
    for (size_t i = 0; i < N; i++) {
      if (tens_cano_type_[i] == MPSTenCanoType::RIGHT) {
        tens_cano_type_[i] = MPSTenCanoType::LEFT;
      } else if (tens_cano_type_[i] == MPSTenCanoType::LEFT) {
        tens_cano_type_[i] = MPSTenCanoType::RIGHT;
      }
    }
  }
  //note the indices directions will be changed.
}

template<typename TenElemT, typename QNT>
void
BMPS<TenElemT, QNT>::InplaceMultipleMPO(BMPS::TransferMPO &mpo,
                                        const size_t Dmin, const size_t Dmax,
                                        const double trunc_err, const size_t iter_max,
                                        const CompressMPSScheme &scheme) {
  auto res = this->MultipleMPO(mpo, Dmin, Dmax, trunc_err, iter_max, scheme);
  (*this) = res;
}

/**
 * MultipleMPOResCheck_
 *
 * Check the physical index and quantum number of the result boundary-MPS after MPO multiplication
 *
 * @param mpo the original
 * @return
 */
template<typename TenElemT, typename QNT>
bool MultipleMPOResCheck_(const typename BMPS<TenElemT, QNT>::TransferMPO &mpo,
                          const bool has_mpo_been_reversed,
                          const BMPS<TenElemT, QNT> &mps,   //input mps
                          const BMPS<TenElemT, QNT> &res,
                          const BMPSPOSITION post) {

  size_t mpo_remain_idx = static_cast<size_t>(Opposite(post));
  for (size_t i = 0; i < res.size(); i++) {
    size_t mpo_idx;
    if (has_mpo_been_reversed || MPOIndex(post) < 2) {
      mpo_idx = i;
    } else {
      mpo_idx = res.size() - 1 - i;
    }
    assert(res[i].GetIndex(1) == mpo[mpo_idx]->GetIndex(mpo_remain_idx));
  }
  assert(res.front().GetIndex(0).dim() == 1);
  assert(res.back().GetIndex(2).dim() == 1);

  QNT qn_mpo = mpo[0]->Div(), qn_mps = mps[0].Div(), qn_res = res[0].Div();
  for (size_t i = 1; i < res.size(); i++) {
    qn_mpo += mpo[i]->Div();
    qn_mps += mps[i].Div();
    qn_res += res[i].Div();
  }
  assert(qn_mpo + qn_mps == qn_res);
  return true;
}

/**
 * BMPS<TenElemT, QNT>::MultipleMPO
 *
 * transfer-MPO cut from 2D tensor network multiples on the boundary MPS.
 *
 *
 * @param mpo
 *  for bosonic tensor network, the transfer-MPO tensor has the following order of legs:
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 *  Please also refer to the boundary-MPS tensor order in different direction
 *  in the annotation of the BMPS class.
 *
 * @param scheme
 *      SVD_COMPRESS: SVD compress, O(D^7) time complexity for square TN
 *      VARIATION2Site: 2-site update variational method, O(D^6) time complexity for square TN
 *      VARIATION1Site: 1-site update variational method, also O(D^6) time complexity, but usually faster than VARIATION2Site;
 *                      but the quantum number in virtual bond of boundary-MPS will be locked so that it will be trapped to local minimal.
 *
 * @return the multiplied boundary-MPS
 */
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPO(BMPS::TransferMPO &mpo, const CompressMPSScheme &scheme,
                                 const size_t Dmin, const size_t Dmax,
                                 const double trunc_err,
                                 const std::optional<double> variational_converge_tol,
                                 const std::optional<size_t> max_iter //only valid for variational methods
) const {
  const size_t N = this->size();
  assert(mpo.size() == N);
  AlignTransferMPOTensorOrder_(mpo);
  if (N == 2 || scheme == CompressMPSScheme::SVD_COMPRESS) {
    double actual_trunc_err_max;
    size_t actual_D_max;
    return MultipleMPOSVDCompress_(mpo, Dmin, Dmax, trunc_err, actual_D_max, actual_trunc_err_max);
  }
  switch (scheme) {
    case CompressMPSScheme::VARIATION2Site: {
      return MultipleMPO2SiteVariationalCompress_(mpo, Dmin, Dmax, trunc_err,
                                                  variational_converge_tol.value(),
                                                  max_iter.value());
    }
    case CompressMPSScheme::VARIATION1Site: {
      return MultipleMPO1SiteVariationalCompress_(mpo, Dmin, Dmax, trunc_err,
                                                  variational_converge_tol.value(),
                                                  max_iter.value());
    }
    default: {
      std::cerr << "Do not support MPO multiplication method." << std::endl;
      exit(1);
    }
  }
}//MultipleMPO

/**
 *         4
 *         |
 *      1--t--3, and phy index 0
 *         |
 *         2
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param mpo
 * @param scheme
 * @return note mpo will also be changed according position in output
 */
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPOWithPhyIdx(BMPS::TransferMPO &mpo,
                                           const size_t Dmin, const size_t Dmax,
                                           const double trunc_err, const size_t iter_max,
                                           const CompressMPSScheme &scheme) const {
  assert(mpo.size() == this->size());
  size_t pre_post = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
  size_t next_post = ((size_t) (position_) + 1) % 4;
  switch (scheme) {
    case CompressMPSScheme::SVD_COMPRESS: {
      BMPS<TenElemT, QNT> res(position_, this->size());
      IndexT idx1;
      if (position_ > 1) {
        std::reverse(mpo.begin(), mpo.end());
      }
      switch (position_) {
        case DOWN: {
          for (size_t i = 0; i < mpo.size(); i++) {
            mpo[i]->Transpose({1, 2, 3, 4, 0});
          }
          break;
        }
        case UP: {
          for (size_t i = 0; i < mpo.size(); i++) {
            mpo[i]->Transpose({3, 4, 1, 2, 0});
          }
          break;
        }
        case LEFT: {
          for (size_t i = 0; i < mpo.size(); i++) {
            mpo[i]->Transpose({4, 1, 2, 3, 0});
          }
          break;
        }
        case RIGHT: {
          for (size_t i = 0; i < mpo.size(); i++) {
            mpo[i]->Transpose({2, 3, 4, 1, 0});
          }
          break;
        }
      }

      idx1 = InverseIndex(mpo[0]->GetIndex(0));

      IndexT idx2 = InverseIndex((*this)[0].GetIndex(0));
      Tensor r = IndexCombine<TenElemT, QNT>(idx1, idx2, IN);
      r.Transpose({2, 0, 1});
      for (size_t i = 0; i < this->size(); i++) {
        Tensor tmp1, tmp2;
        Contract<TenElemT, QNT, false, false>((*this)[i], r, 0, 2, 1, tmp1);
        Contract<TenElemT, QNT, true, true>(tmp1, *mpo[i], 3, 0, 2, tmp2);
        res.alloc(i);
        if (i < this->size() - 1) {
          tmp2.Transpose({1, 3, 4, 2, 0});
          QNT mps_div = (*this)[i].Div();
          r = Tensor();
          QR(&tmp2, 3, mps_div, res(i), &r);
        } else {
          auto trivial_idx = tmp2.GetIndex(0);
          Tensor tmp3({InverseIndex(trivial_idx)});
          tmp3({0}) = 1.0;
          Contract(&tmp3, {0}, &tmp2, {0}, res(i));
          res(i)->Transpose({0, 2, 3, 1});
        }
      }
#ifndef NDEBUG
      for (size_t i = 0; i < this->size(); i++) {
        assert(res[i].GetIndex(1) == mpo[i]->GetIndex(3));
        assert(res[i].GetIndex(2) == mpo[i]->GetIndex(4)); //phy index
      }
      assert(res[0].GetIndex(0).dim() == 1);
      assert(res[res.size() - 1].GetIndex(3).dim() == 1);
#endif
      for (size_t i = res.size() - 1; i > 0; --i) {
        res.RightCanonicalizeTruncateWithPhyIdx_(i, Dmin, Dmax, trunc_err);
      }
      assert(res[0].GetIndex(0).dim() == 1);
      assert(res[res.size() - 1].GetIndex(3).dim() == 1);
      return res;
    }
    case CompressMPSScheme::VARIATION2Site: {
      const double converge_tol = 1e-15;
      size_t N = this->size();
      if (N == 2) {
        return MultipleMPOWithPhyIdx(mpo, Dmin, Dmax,
                                     trunc_err, iter_max, CompressMPSScheme::SVD_COMPRESS);
      }
      BMPS<TenElemT, QNT> res_init = InitGuessForVariationalMPOMultiplicationWithPhyIdx_(mpo, Dmin, Dmax, trunc_err);
      BMPS<TenElemT, QNT> res_dag(res_init);
      for (size_t i = 0; i < res_dag.size(); i++) {
        res_dag[i].Dag();
      } //initial guess for the result

      std::vector<Tensor> lenvs, renvs;  // from the view of down mps
      lenvs.reserve(N - 1);
      renvs.reserve(N - 1);
      IndexT index2 = InverseIndex((*this)[0].GetIndex(0));
      IndexT index1 = InverseIndex(mpo[0]->GetIndex(0));
      IndexT index0 = InverseIndex(res_dag[0].GetIndex(0));
      auto lenv0 = Tensor({index0, index1, index2});
      lenv0({0, 0, 0}) = 1;
      lenvs.push_back(lenv0);
      index0 = InverseIndex((*this)[N - 1].GetIndex(2));
      index1 = InverseIndex(mpo[N - 1]->GetIndex(2));
      index2 = InverseIndex(res_dag[N - 1].GetIndex(2));
      auto renv0 = Tensor({index0, index1, index2});
      renv0({0, 0, 0}) = 1;
      renvs.push_back(renv0);

      //initially grow the renvs
      for (size_t i = N - 1; i > 1; i--) {
        Tensor renv_next, temp_ten, temp_ten2;
        Contract<TenElemT, QNT, true, true>((*this)[i], renvs.back(), 2, 0, 1, temp_ten);
        Contract<TenElemT, QNT, false, false>(temp_ten, *mpo[i], 1, 1, 2, temp_ten2);
        Contract(&temp_ten2, {0, 2, 3}, res_dag(i), {3, 1, 2}, &renv_next);
        renvs.emplace_back(renv_next);
      }

      Tensor s12bond_last;
      for (size_t iter = 0; iter < iter_max; iter++) {
        //left move
        QLTensor<QLTEN_Double, QNT> s;
        for (size_t i = 0; i < N - 2; i++) {
          Tensor tmp[6];
          Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
          Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, 0, 2, tmp[1]);

          Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
          Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, 1, 2, tmp[3]);
          Contract(tmp + 1, {2, 0}, tmp + 3, {4, 1}, tmp + 4);
          tmp[4].Dag();
          Tensor *pu = new Tensor(), vt;
          s = QLTensor<QLTEN_Double, QNT>();
          double actual_trunc_err;
          size_t D;
          SVD(tmp + 4,
              3, res_dag[i].Div(), trunc_err, Dmin, Dmax,
              pu, &s, &vt, &actual_trunc_err, &D
          );

          delete res_dag(i);
          res_dag(i) = pu;

          //grow left_tensor
          Contract(tmp + 1, {1, 3, 4}, res_dag(i), {0, 1, 2}, tmp + 5);
          tmp[5].Transpose({2, 1, 0});
          lenvs.emplace_back(tmp[5]);
          renvs.pop_back();
        }
        //right move
        for (size_t i = N - 2; i > 0; i--) {
          Tensor tmp[6];
          Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
          Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, 0, 2, tmp[1]);

          Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
          Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, 1, 2, tmp[3]);
          Contract(tmp + 1, {2, 0}, tmp + 3, {4, 1}, tmp + 4);
          tmp[4].Dag();
          Tensor u, *pvt = new Tensor();
          s = QLTensor<QLTEN_Double, QNT>();
          double actual_trunc_err;
          size_t D;
          SVD(tmp + 4,
              3, res_dag[i].Div(), trunc_err, Dmin, Dmax,
              &u, &s, pvt, &actual_trunc_err, &D
          );

          delete res_dag(i + 1);
          pvt->Transpose({0, 2, 3, 1});
          res_dag(i + 1) = pvt;
          Contract(&tmp[3], {0, 2, 3}, res_dag(i + 1), {3, 1, 2}, &tmp[5]);
          renvs.emplace_back(tmp[5]);
          lenvs.pop_back();
        }
        if (iter == 0 || s.GetActualDataSize() != s12bond_last.GetActualDataSize()) {
          s12bond_last = s;
          continue;
        }
        double diff = 0.0;
        const double *s_data = s.GetRawDataPtr();
        const double *s_last_data = s12bond_last.GetRawDataPtr();
        for (size_t k = 0; k < s.GetActualDataSize(); k++) {
          diff += std::fabs(*(s_data + k) - *(s_last_data + k));
        }
        if (diff < converge_tol) {
          break;
        } else {
          s12bond_last = s;
        }
      }
      size_t i = 0;
      Tensor tmp[6];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, 0, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, 1, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {4, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor u, *pvt = new Tensor();
      QLTensor<QLTEN_Double, QNT> s;
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          3, res_dag[i].Div(), trunc_err, Dmin, Dmax,
          &u, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i);
      res_dag(i) = new Tensor();
      Contract<TenElemT, QNT, true, true>(u, s, 3, 0, 1, res_dag[i]);
      pvt->Transpose({0, 2, 3, 1});
      delete (res_dag(i + 1));
      res_dag(i + 1) = pvt;

      BMPS<TenElemT, QNT> res(std::move(res_dag));
      for (size_t i = 0; i < res.size(); i++) {
        res[i].Dag();
        res.tens_cano_type_[i] = MPSTenCanoType::RIGHT;
      }
      res.center_ = 0;
#ifndef NDEBUG
      QNT qn_mpo = qn0_, qn_mps = qn0_, qn_res = qn0_;
      for (size_t i = 0; i < res.size(); i++) {
        qn_mpo += mpo[i]->Div();
        qn_mps += (*this)[i].Div();
        qn_res += res[i].Div();
      }
      assert(qn_mpo + qn_mps == qn_res);
      for (size_t i = 0; i < this->size(); i++) {
        assert(res[i].GetIndex(1) == mpo[i]->GetIndex(3));
        assert(res[i].GetIndex(2) == mpo[i]->GetIndex(4)); //phy index
      }
      assert(res[0].GetIndex(0).dim() == 1);
      assert(res[res.size() - 1].GetIndex(3).dim() == 1);
#endif
      return res;
    }
    case CompressMPSScheme::VARIATION1Site: {

    }
  }
}

template<typename TenElemT, typename QNT>
void BMPS<TenElemT, QNT>::AlignTransferMPOTensorOrder_(TransferMPO &mpo) const {
  if (MPOIndex(position_) > 1) {  //RIGHT or UP
    std::reverse(mpo.begin(), mpo.end());
  }
}

template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPOSVDCompress_(const TransferMPO &mpo,
                                             const size_t Dmin, const size_t Dmax, const double trunc_err,
                                             size_t &actual_Dmax, double &actual_trunc_err_max) const {
  const size_t N = this->size();
#ifndef NDEBUG
  assert(mpo.size() == N);
#endif
  size_t pre_post = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
  BMPS<TenElemT, QNT> res_mps(position_, N);
  IndexT idx1 = InverseIndex(mpo[0]->GetIndex(pre_post));
  IndexT idx2 = InverseIndex((*this)[0].GetIndex(0));
  Tensor r = IndexCombine<TenElemT, QNT>(idx1, idx2, IN);
  r.Transpose({2, 0, 1});
  /*
   *      ----1 (connected to mpo tensor)
   *      |
   *  0---r---2 (connected to mps tensor)
   */
  for (size_t i = 0; i < N; i++) {
    Tensor tmp1, tmp2;
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, false, false>(r, (*this)[i], 2, 0, 1, tmp1);
      Tensor mpo_trans = *mpo[i];
      switch (this->position_) {
        case DOWN: {
          break;
        }
        case LEFT: {
          mpo_trans.Transpose({3, 0, 1, 2, 4});
          break;
        }
        case UP: {
          mpo_trans.Transpose({2, 3, 0, 1, 4});
          break;
        }
        case RIGHT: {
          mpo_trans.Transpose({1, 2, 3, 0, 4});
          break;
        }
      }
      Contract<TenElemT, QNT, false, true>(tmp1, mpo_trans, 1, 0, 2, tmp2);
      assert(tmp2.GetIndex(1).dim() == 1);
      assert(tmp2.GetIndex(5).dim() == 1);
      tmp2.FuseIndex(1, 5);
      assert(tmp2.GetIndex(0).dim() == 1);
    } else {
      Contract<TenElemT, QNT, false, false>((*this)[i], r, 0, 2, 1, tmp1);
      Contract<TenElemT, QNT, false, true>(tmp1, *mpo[i], 3, pre_post, 2, tmp2);
    }
    res_mps.alloc(i);
    if (i < this->size() - 1) {
      std::vector<size_t> transpose_order;
      if constexpr (Tensor::IsFermionic()) {
        transpose_order = {2, 4, 0, 3, 1};
      } else {
        transpose_order = {1, 3, 2, 0};
      }
      tmp2.Transpose(transpose_order);
      QNT mps_div = (*this)[i].Div();
      r = Tensor();
      constexpr size_t ldim = 2 + Tensor::IsFermionic();
      QR(&tmp2, ldim, mps_div, res_mps(i), &r);
      if constexpr (Tensor::IsFermionic()) {
        res_mps(i)->Transpose({0, 1, 3, 2});
        assert(res_mps(i)->GetIndex(3).dim() == 1);
      }
    } else {
      auto trivial_idx1 = InverseIndex(tmp2.GetIndex(0));
      auto trivial_idx2 = InverseIndex(tmp2.GetIndex(2 + Tensor::IsFermionic()));
      assert(trivial_idx1.dim() == 1);
      assert(trivial_idx2.dim() == 1);
      Tensor right_boundary = IndexCombine<TenElemT, QNT>(trivial_idx1, trivial_idx2, OUT);
      std::vector<size_t> ctrct_axes;
      if constexpr (Tensor::IsFermionic()) {
        ctrct_axes = {0, 3};
      } else {
        ctrct_axes = {0, 2};
      }
      Contract(&tmp2, ctrct_axes, &right_boundary, {0, 1}, res_mps(i));
      assert(res_mps[i].GetRawDataPtr() != nullptr);
      if constexpr (Tensor::IsFermionic()) {
        res_mps(i)->Transpose({1, 2, 3, 0});
      }
    }
  }
  actual_Dmax = 1;
  actual_trunc_err_max = 0.0;
  for (size_t i = N - 1; i > 0; --i) {
    auto [D, actual_trunc_err] = res_mps.RightCanonicalizeTruncate(i, Dmin, Dmax, trunc_err);
    actual_Dmax = std::max(actual_Dmax, D);
    actual_trunc_err_max = std::max(actual_trunc_err, actual_trunc_err_max);
  }
#ifndef NDEBUG
  MultipleMPOResCheck_(mpo, true, *this, res_mps, position_);
#endif
  return res_mps;
}

template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPO2SiteVariationalCompress_(const TransferMPO &mpo,
                                                          const size_t Dmin,
                                                          const size_t Dmax,
                                                          const double trunc_err,
                                                          const double variational_converge_tol,
                                                          const size_t max_iter) const {
//  static_assert(!Tensor::IsFermionic());
  assert(!Tensor::IsFermionic());
  const size_t N = this->size();
  size_t pre_post = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
  size_t next_post = ((size_t) (position_) + 1) % 4;

  BMPS<TenElemT, QNT> res_init = InitGuessForVariationalMPOMultiplication_(mpo, Dmin, Dmax, trunc_err);
  BMPS<TenElemT, QNT> res_dag(res_init);
  for (size_t i = 0; i < res_dag.size(); i++) {
    res_dag[i].Dag();
  }

  std::vector<Tensor> lenvs, renvs;  // from the view of down mps
  lenvs.reserve(N - 1);
  renvs.reserve(N - 1);
  IndexT index2 =
      InverseIndex((*this)[0].GetIndex(0)); // for fermion case, the boundary index should always be even qn trivial index.
  IndexT index1 = InverseIndex(mpo[0]->GetIndex(pre_post));
  IndexT index0 = InverseIndex(res_dag[0].GetIndex(0));
  auto lenv0 = Tensor({index0, index1, index2});
  lenv0({0, 0, 0}) = 1;
  lenvs.push_back(lenv0);
  index0 = InverseIndex((*this)[N - 1].GetIndex(2));
  index1 = InverseIndex(mpo[N - 1]->GetIndex(next_post));
  index2 = InverseIndex(res_dag[N - 1].GetIndex(2));
  auto renv0 = Tensor({index0, index1, index2});
  renv0({0, 0, 0}) = 1;
  renvs.push_back(renv0);

  //initially grow the renvs
  for (size_t i = N - 1; i > 1; i--) {
    Tensor renv_next, temp_ten, temp_ten2;
    Contract<TenElemT, QNT, true, true>((*this)[i], renvs.back(), 2, 0, 1, temp_ten);
    Contract<TenElemT, QNT, false, false>(temp_ten, *mpo[i], 1, position_, 2, temp_ten2);
    Contract(&temp_ten2, {2, 0}, res_dag(i), {1, 2}, &renv_next);
    renvs.emplace_back(renv_next);
  }

  QLTensor<QLTEN_Double, QNT> s12bond_last;
  for (size_t iter = 0; iter < max_iter; iter++) {
    //left move
    QLTensor<QLTEN_Double, QNT> s;
    for (size_t i = 0; i < N - 2; i++) {
      Tensor tmp[6];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, pre_post, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, position_, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor *pu = new Tensor(), *pvt = new Tensor();
      s = QLTensor<QLTEN_Double, QNT>();
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          2, res_dag[i].Div(), trunc_err, Dmin, Dmax,
          pu, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i);
      res_dag(i) = pu;

      //grow left_tensor
      Contract(tmp + 1, {1, 3}, res_dag(i), {0, 1}, tmp + 5);
      tmp[5].Transpose({2, 1, 0});
      lenvs.emplace_back(tmp[5]);
      renvs.pop_back();
      delete pvt;
    }
    //right move
    for (size_t i = N - 2; i > 0; i--) {
      Tensor tmp[6];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, pre_post, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, position_, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor *pu = new Tensor(), *pvt = new Tensor();
      s = QLTensor<QLTEN_Double, QNT>();
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          2, res_dag[i].Div(), trunc_err, Dmin, Dmax,
          pu, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i + 1);
      pvt->Transpose({0, 2, 1});
      res_dag(i + 1) = pvt;
      Contract(&tmp[3], {2, 0}, res_dag(i + 1), {1, 2}, &tmp[5]);
      renvs.emplace_back(tmp[5]);
      lenvs.pop_back();
      delete pu;
    }
    if (iter == 0 || s.GetIndex(0) != s12bond_last.GetIndex(0)) {
      s12bond_last = s;
      continue;
    }
    decltype(s) diff_ten = s + (-s12bond_last);
    double diff = 0.0;
    for (size_t k = 0; k < diff_ten.GetShape().front(); k++) {
      diff += std::fabs(diff_ten({k, k}));
    }
    if (diff / s({0, 0}) < variational_converge_tol) {
      break;
    } else {
      s12bond_last = s;
    }
  }
  Tensor tmp[6];
  Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[0], 2, 0, 1, tmp[0]);
  Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[0], 1, pre_post, 2, tmp[1]);

  Contract<TenElemT, QNT, true, true>((*this)[1], renvs.back(), 2, 0, 1, tmp[2]);
  Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[1], 1, position_, 2, tmp[3]);
  Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
  tmp[4].Dag();
  Tensor u, *pvt = new Tensor();
  QLTensor<QLTEN_Double, QNT> s;
  double actual_trunc_err;
  size_t D;
  SVD(tmp + 4,
      2, res_dag[0].Div(), trunc_err, Dmin, Dmax,
      &u, &s, pvt, &actual_trunc_err, &D
  );
#ifndef NDEBUG
  //      if (actual_trunc_err > trunc_err) {
  //      std::cout << "actual_trunc_err in BMPS : " << actual_trunc_err << std::endl;
  //      }
#endif

  delete res_dag(0);
  res_dag(0) = new Tensor();
  Contract(&u, &s, {{2}, {0}}, res_dag(0));
  pvt->Transpose({0, 2, 1});
  delete (res_dag(1));
  res_dag(1) = pvt;

  BMPS<TenElemT, QNT> res(std::move(res_dag));
  for (size_t i = 0; i < res.size(); i++) {
    res[i].Dag();
    res.tens_cano_type_[i] = MPSTenCanoType::RIGHT;
  }
  res.center_ = 0;
#ifndef NDEBUG
  MultipleMPOResCheck_(mpo, true, *this, res, position_);
#endif
  return res;
}

template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPO1SiteVariationalCompress_(const TransferMPO &mpo,
                                                          const size_t Dmin,
                                                          const size_t Dmax,
                                                          const double trunc_err,
                                                          const double variational_converge_tol,
                                                          const size_t max_iter) const {
//  static_assert(!Tensor::IsFermionic());
  assert(!Tensor::IsFermionic());
  const size_t N = this->size();
  size_t pre_post = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
  size_t next_post = ((size_t) (position_) + 1) % 4;

  // Copy the code from VARIATIONAL2Site
  BMPS<TenElemT, QNT> res_init = InitGuessForVariationalMPOMultiplication_(mpo, Dmax, Dmax, 0.0);
  BMPS<TenElemT, QNT> res_dag(res_init);
  for (size_t i = 0; i < res_dag.size(); i++) {
    res_dag[i].Dag();
  } //initial guess for the result

  std::vector<Tensor> lenvs, renvs;  // from the view of down mps
  lenvs.reserve(N - 1);
  renvs.reserve(N - 1);
  IndexT index2 = InverseIndex((*this)[0].GetIndex(0));
  IndexT index1 = InverseIndex(mpo[0]->GetIndex(pre_post));
  IndexT index0 = InverseIndex(res_dag[0].GetIndex(0));
  auto lenv0 = Tensor({index0, index1, index2});
  lenv0({0, 0, 0}) = 1;
  lenvs.push_back(lenv0);
  index0 = InverseIndex((*this)[N - 1].GetIndex(2));
  index1 = InverseIndex(mpo[N - 1]->GetIndex(next_post));
  index2 = InverseIndex(res_dag[N - 1].GetIndex(2));
  auto renv0 = Tensor({index0, index1, index2});
  renv0({0, 0, 0}) = 1;
  renvs.push_back(renv0);

  //initially grow the renvs
  for (size_t i = N - 1; i > 1; i--) {
    Tensor renv_next, temp_ten, temp_ten2;
    Contract<TenElemT, QNT, true, true>((*this)[i], renvs.back(), 2, 0, 1, temp_ten);
    Contract<TenElemT, QNT, false, false>(temp_ten, *mpo[i], 1, position_, 2, temp_ten2);
    Contract(&temp_ten2, {2, 0}, res_dag(i), {1, 2}, &renv_next);
    renvs.emplace_back(renv_next);
  }

  // do once 2 site update to make sure the bond dimension = Dmax
  {
    //right moving
    QLTensor<QLTEN_Double, QNT> s;
    for (size_t i = 0; i < N - 2; i++) {
      Tensor tmp[6];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, pre_post, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, position_, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor *pu = new Tensor(), *pvt = new Tensor();
      s = QLTensor<QLTEN_Double, QNT>();
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          2, res_dag[i].Div(), trunc_err, Dmax, Dmax,
          pu, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i);
      res_dag(i) = pu;

      //grow lenvs
      Contract(tmp + 1, {1, 3}, res_dag(i), {0, 1}, tmp + 5);
      tmp[5].Transpose({2, 1, 0});
      lenvs.emplace_back(tmp[5]);
      renvs.pop_back();
      delete pvt;
    }
    //left moving
    for (size_t i = N - 2; i > 0; i--) {
      Tensor tmp[6];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, pre_post, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[i + 1], 1, position_, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor *pu = new Tensor(), *pvt = new Tensor();
      s = QLTensor<QLTEN_Double, QNT>();
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          2, res_dag[i].Div(), trunc_err, Dmax, Dmax,
          pu, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i + 1);
      pvt->Transpose({0, 2, 1});
      res_dag(i + 1) = pvt;
      //grow renvs
      Contract(&tmp[3], {2, 0}, res_dag(i + 1), {1, 2}, &tmp[5]);
      renvs.emplace_back(tmp[5]);
      lenvs.pop_back();
      delete pu;
    }
  }
  Tensor tmp[6];
  Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[0], 2, 0, 1, tmp[0]);
  Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[0], 1, pre_post, 2, tmp[1]);

  Contract<TenElemT, QNT, true, true>((*this)[1], renvs.back(), 2, 0, 1, tmp[2]);
  Contract<TenElemT, QNT, false, false>(tmp[2], *mpo[1], 1, position_, 2, tmp[3]);
  Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
  tmp[4].Dag();
  Tensor u, *pvt = new Tensor();
  QLTensor<QLTEN_Double, QNT> s;
  double actual_trunc_err;
  size_t D;
  SVD(tmp + 4,
      2, res_dag[0].Div(), trunc_err, Dmin, Dmax,
      &u, &s, pvt, &actual_trunc_err, &D
  );

  delete res_dag(0);
  res_dag(0) = new Tensor();
  Contract(&u, &s, {{2}, {0}}, res_dag(0));
  pvt->Transpose({0, 2, 1});
  delete (res_dag(1));
  res_dag(1) = pvt;

  // one more step in growing renvs for switching to one site update
  Contract(&tmp[3], {2, 0}, res_dag(1), {1, 2}, &tmp[5]);
  renvs.emplace_back(tmp[5]);

  // one site update begin
  double last_r_norm = 0, r_norm = 0;
  for (size_t iter = 0; iter < max_iter; iter++) {
    //right moving
    for (size_t i = 0; i < N - 1; i++) {
      Tensor tmp[4];
      Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(tmp[0], *mpo[i], 1, pre_post, 2, tmp[1]);
      Contract(tmp + 1, {0, 2}, &renvs.back(), {0, 1}, tmp + 2);
      tmp[2].Dag();
      Tensor *pq = new Tensor(), r;
      QR(tmp + 2, 2, (tmp + 2)->Div(), pq, &r);

      delete res_dag(i);
      res_dag(i) = pq;
      //grow lenvs
      Contract(tmp + 1, {1, 3}, res_dag(i), {0, 1}, tmp + 3);
      tmp[3].Transpose({2, 1, 0});
      lenvs.emplace_back(tmp[3]);
      renvs.pop_back();
    }
    //left moving
    for (size_t i = N - 1; i > 0; i--) {
      Tensor tmp[4];
      Contract<TenElemT, QNT, true, true>((*this)[i], renvs.back(), 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], *mpo[i], 1, position_, 2, tmp[1]);
      Contract(tmp + 1, {3, 1}, &lenvs.back(), {1, 2}, tmp + 2);
      tmp[2].Dag();
      Tensor *pq = new Tensor(), r;
      QR(tmp + 2, 2, (tmp + 2)->Div(), pq, &r);

      delete res_dag(i);
      pq->Transpose({2, 1, 0});
      res_dag(i) = pq;
      //grow renvs
      Contract(&tmp[1], {2, 0}, res_dag(i), {1, 2}, &tmp[3]);
      renvs.emplace_back(tmp[3]);
      lenvs.pop_back();

      r_norm = r.Get2Norm();
    }
    if (iter == 0 || std::abs(r_norm - last_r_norm) / std::abs(r_norm) > variational_converge_tol) {
      last_r_norm = r_norm;
      continue;
    } else {
      break;
    }
  }

  Tensor temp[4];
  Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[0], 2, 0, 1, temp[0]);
  Contract<TenElemT, QNT, false, true>(temp[0], *mpo[0], 1, pre_post, 2, temp[1]);
  delete res_dag(0);
  res_dag(0) = new Tensor();
  Contract(temp + 1, {0, 2}, &renvs.back(), {0, 1}, res_dag(0));
  res_dag(0)->Dag();

  BMPS<TenElemT, QNT> res(std::move(res_dag));
  for (size_t i = 0; i < res.size(); i++) {
    res[i].Dag();
    res.tens_cano_type_[i] = MPSTenCanoType::RIGHT;
  }
  res.center_ = 0;
#ifndef NDEBUG
  MultipleMPOResCheck_(mpo, true, *this, res, position_);
#endif
  return res;
}
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::InitGuessForVariationalMPOMultiplication_(const BMPS::TransferMPO &mpo,
                                                               const size_t Dmin,
                                                               const size_t Dmax,
                                                               const double trunc_err) const {
//  const size_t N = this->size();
//  BMPS<TenElemT, QNT> multip_init(position_, N);
//  for (size_t i = 0; i < N; i++) {
//    IndexT index = InverseIndex(ordered_mpo[i]->GetIndex(position_));
//    Tensor a = Tensor({index});
//    a({0}) = 1;
//    multip_init.alloc(i);
//    Contract<TenElemT, QNT, false, true>(*ordered_mpo[i], a, (size_t) position_, 0, 1, multip_init[i]);
//    multip_init[i].Transpose({2, 1, 0});
//  }
//
//#ifndef NDEBUG
//  for (size_t i = 0; i < multip_init.size(); i++) {
//    assert(multip_init[i].GetActualDataSize() > 0);
//  } //initial guess for the result
//#endif
//  multip_init.Centralize(0);
  BMPS mps_copy(*this);
  mps_copy.Centralize(this->size() - 1);
  for (size_t i = mps_copy.size() - 1; i > 0; --i) {
    mps_copy.RightCanonicalizeTruncate(i, 1, 2, 0.0);
  }
  double actual_trunc_err_max;
  size_t actual_D_max;
  auto multip_init = mps_copy.MultipleMPOSVDCompress_(mpo, Dmin, Dmax, trunc_err, actual_D_max, actual_trunc_err_max);

#ifndef NDEBUG
  for (size_t i = 0; i < multip_init.size(); i++) {
    assert(multip_init[i].GetActualDataSize() > 0);
  } //initial guess for the result
#endif
  return multip_init;
}

///< mpo is original mpo without reverse
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::InitGuessForVariationalMPOMultiplicationWithPhyIdx_(const BMPS::TransferMPO &mpo,
                                                                         const size_t Dmin,
                                                                         const size_t Dmax,
                                                                         const double trunc_err) const {
//  const size_t N = this->size();
//  BMPS<TenElemT, QNT> multip_init(position_, N);
//  for (size_t i = 0; i < N; i++) {
//    IndexT index = InverseIndex(ordered_mpo[i]->GetIndex(position_));
//    Tensor a = Tensor({index});
//    a({0}) = 1;
//    multip_init.alloc(i);
//    Contract<TenElemT, QNT, false, true>(*ordered_mpo[i], a, (size_t) position_, 0, 1, multip_init[i]);
//    multip_init[i].Transpose({2, 1, 0});
//  }
//
//#ifndef NDEBUG
//  for (size_t i = 0; i < multip_init.size(); i++) {
//    assert(multip_init[i].GetActualDataSize() > 0);
//  } //initial guess for the result
//#endif
//  multip_init.Centralize(0);

  BMPS mps_copy(*this);
  mps_copy.Centralize(this->size() - 1);
  for (size_t i = mps_copy.size() - 1; i > 0; --i) {
    mps_copy.RightCanonicalizeTruncateWithPhyIdx_(i, 1, 2, 0.0);
  }
  auto multip_init = mps_copy.MultipleMPOWithPhyIdx(mpo, Dmin, Dmax, trunc_err, 0, CompressMPSScheme::SVD_COMPRESS);

#ifndef NDEBUG
  for (size_t i = 0; i < multip_init.size(); i++) {
    assert(multip_init[i].GetActualDataSize() > 0);
  } //initial guess for the result
#endif
  return multip_init;
}

template<typename TenElemT, typename QNT>
double
BMPS<TenElemT, QNT>::RightCanonicalizeTruncateWithPhyIdx_(const size_t site, const size_t Dmin,
                                                          const size_t Dmax, const double trunc_err) {

  QLTensor<QLTEN_Double, QNT> s;
  auto pvt = new Tensor;
  Tensor u;
  double actual_trunc_err;
  size_t D;
  SVD(
      (*this)(site),
      1, qn0_, trunc_err, Dmin, Dmax,
      &u, &s, pvt, &actual_trunc_err, &D
  );
//  std::cout << "Truncate MPS bond " << std::setw(4) << site
//            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
//            << " D = " << std::setw(5) << D;
//  std::cout << std::scientific << std::endl;

  delete (*this)(site);
  (*this)(site) = pvt;

  Tensor temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  auto pnext_ten = new Tensor;
  Contract((*this)(site - 1), &temp_ten, {{3},
                                          {0}}, pnext_ten);
  delete (*this)(site - 1);
  (*this)(site - 1) = pnext_ten;

  //set ten canonical type
  return actual_trunc_err;
}

}//qlpeps

#endif //QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H
