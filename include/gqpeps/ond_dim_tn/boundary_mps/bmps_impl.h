/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-01
*
* Description: GraceQ/VMC-PEPS project. Boundary MPS
*/

#ifndef GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H
#define GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H


namespace gqpeps {
using namespace gqten;
using namespace gqmps2;

template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>::BMPS(const BMPSPOSITION position, const size_t size,
                          const BMPS::IndexT &local_hilbert_space) :
    TenVec<LocalTenT>(size),
    position_(position),
    center_(0),
    tens_cano_type_(size, MPSTenCanoType::RIGHT) {
  assert(local_hilbert_space.dim() == 1);
  LocalTenT mps_ten = LocalTenT({index0_in_, local_hilbert_space, index0_out_});
  mps_ten({0, 0, 0}) = 1.0;
  for (size_t i = 0; i < size; i++) {
    this->alloc(i);
    (*this)[i] = mps_ten;
  }
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) {
  return DuoVector<LocalTenT>::operator[](idx);
}

template<typename TenElemT, typename QNT>
const GQTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) const {
  return DuoVector<LocalTenT>::operator[](idx);
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> *&BMPS<TenElemT, QNT>::operator()(const size_t idx) {
  return DuoVector<LocalTenT>::operator()(idx);
}

template<typename TenElemT, typename QNT>
const GQTensor<TenElemT, QNT> *BMPS<TenElemT, QNT>::operator()(const size_t idx) const {
  return DuoVector<LocalTenT>::operator()(idx);
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
  size_t ldims(2);
  auto pq = new LocalTenT;
  LocalTenT r;
  QR((*this)(site_idx), ldims, Div((*this)[site_idx]), pq, &r);
  delete (*this)(site_idx);
  (*this)(site_idx) = pq;

  auto pnext_ten = new LocalTenT;
  Contract(&r, (*this)(site_idx + 1), {{1},
                                       {0}}, pnext_ten);
  delete (*this)(site_idx + 1);
  (*this)(site_idx + 1) = pnext_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::LEFT;
  tens_cano_type_[site_idx + 1] = MPSTenCanoType::NONE;
}

template<typename TenElemT, typename QNT>
GQTensor<GQTEN_Double, QNT> BMPS<TenElemT, QNT>::RightCanonicalizeTen(const size_t site_idx) {
  ///< TODO: using LU decomposition
  assert(site_idx > 0);
  size_t ldims = 1;
  LocalTenT u;
  GQTensor<GQTEN_Double, QNT> s;
  auto pvt = new LocalTenT;
  auto qndiv = Div((*this)[site_idx]);
  mock_gqten::SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  LocalTenT temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  std::vector<std::vector<size_t>> ctrct_axes = {{2},
                                                 {0}};
  auto pprev_ten = new LocalTenT;
  Contract((*this)(site_idx - 1), &temp_ten, ctrct_axes, pprev_ten);
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = pprev_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
  return s;
}

template<typename TenElemT, typename QNT>
double
BMPS<TenElemT, QNT>::RightCanonicalizeTrunctate(const size_t site, const size_t Dmin,
                                                const size_t Dmax, const GQTEN_Double trunc_err) {

  GQTensor<GQTEN_Double, QNT> s;
  auto pvt = new LocalTenT;
  LocalTenT u;
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

  LocalTenT temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  auto pnext_ten = new LocalTenT;
  Contract((*this)(site - 1), &temp_ten, {{2},
                                          {0}}, pnext_ten);
  delete (*this)(site - 1);
  (*this)(site - 1) = pnext_ten;

  return actual_trunc_err;
}

template<typename TenElemT, typename QNT>
std::vector<double> BMPS<TenElemT, QNT>::GetEntanglementEntropy(size_t n) {
  size_t N = this->size();
  std::vector<double> ee_list(N - 1);
  Centralize(N - 1);
  (*this)[N - 1].Normalize();

  for (size_t i = N - 1; i >= 1; --i) { // site
    auto s = RightCanonicalizeTen(i);
    double ee = 0;
    double sum_of_p2n = 0.0;
    for (size_t k = 0; k < s.GetShape()[0]; ++k) { // indices of singular value matrix
      double singular_value = s(k, k);
      double p = singular_value * singular_value;
      if (n == 1) {
        ee += (-p * std::log(p));
      } else {
        double p_to_n = p;
        for (size_t j = 1; j < n; j++) { // order of power
          p_to_n *= p;
        }
        sum_of_p2n += p_to_n;
      }
    }
    if (n == 1) {
      ee_list[i - 1] = ee;
    } else {
      ee_list[i - 1] = -std::log(sum_of_p2n) / (double) (n - 1);
      // note above formula must be in form of n-1 rather 1-n because of n is type of size_t
    }
  }
  return ee_list;
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
    (*this)(i)->Transpose({2, 1, 0});
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
BMPS<TenElemT, QNT>::InplaceMultipleMPO(const BMPS::TransferMPO &mpo,
                                        const size_t Dmin, const size_t Dmax,
                                        const double trunc_err,
                                        const CompressMPSScheme &scheme) {
  auto res = this->MultipleMPO(mpo, Dmin, Dmax, trunc_err, scheme);
  (*this) = res;
}

/**
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param mpo
 * @param scheme
 * @return
 */
template<typename TenElemT, typename QNT>
BMPS<TenElemT, QNT>
BMPS<TenElemT, QNT>::MultipleMPO(const BMPS::TransferMPO &mpo,
                                 const size_t Dmin, const size_t Dmax,
                                 const double trunc_err,
                                 const CompressMPSScheme &scheme) const {
  assert(mpo.size() == this->size());
  BMPS<TenElemT, QNT> res(position_, this->size());
  switch (scheme) {
    case SVD_COMPRESS: {
      size_t ctrct_mpo_ten_start_idx = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
      IndexT idx1;
      if (MPOIndex(position_) < 2) {
        idx1 = InverseIndex(mpo[0]->GetIndex(ctrct_mpo_ten_start_idx));
      } else {
        idx1 = InverseIndex(mpo.back()->GetIndex(ctrct_mpo_ten_start_idx));
      }
      IndexT idx2 = InverseIndex((*this)[0].GetIndex(0));
      LocalTenT r = IndexCombine<TenElemT, QNT>(idx1, idx2, IN);
      r.Transpose({2, 0, 1});
      for (size_t i = 0; i < this->size(); i++) {
        LocalTenT tmp1, tmp2;
        Contract<TenElemT, QNT, false, false>((*this)[i], r, 0, 2, 1, tmp1);
        size_t mpo_idx;
        if (MPOIndex(position_) < 2) {
          mpo_idx = i;
        } else {
          mpo_idx = this->size() - 1 - i;
        }
        Contract<TenElemT, QNT, true, true>(tmp1, *mpo[mpo_idx], 3, ctrct_mpo_ten_start_idx, 2, tmp2);
        res.alloc(i);
        if (i < this->size() - 1) {
          tmp2.Transpose({1, 3, 2, 0});
          QNT mps_div = (*this)[i].Div();
          r = LocalTenT();
          QR(&tmp2, 2, mps_div, res(i), &r);
        } else {
          auto trivial_idx = tmp2.GetIndex(0);
          LocalTenT tmp3({InverseIndex(trivial_idx)});
          tmp3({0}) = 1.0;
          Contract(&tmp3, {0}, &tmp2, {0}, res(i));
          res(i)->Transpose({0, 2, 1});
        }
      }
#ifndef NDEBUG
      size_t mpo_remain_idx = static_cast<size_t>(Opposite(position_));
      for (size_t i = 0; i < this->size(); i++) {
        size_t mpo_idx;
        if (MPOIndex(position_) < 2) {
          mpo_idx = i;
        } else {
          mpo_idx = this->size() - 1 - i;
        }
        assert(res[i].GetIndex(1) == mpo[mpo_idx]->GetIndex(mpo_remain_idx));
      }
      assert(res[0].GetIndex(0).dim() == 1);
      assert(res[res.size() - 1].GetIndex(2).dim() == 1);
#endif
      for (size_t i = res.size() - 1; i > 0; --i) {
        res.RightCanonicalizeTrunctate(i, Dmin, Dmax, trunc_err);
      }
      assert(res[0].GetIndex(0).dim() == 1);
      assert(res[res.size() - 1].GetIndex(2).dim() == 1);
      return res;
    }
    case VARIATION: {
      switch (position_) {
        case DOWN: {
          break;
        }
        case UP: {
          break;
        }
        case LEFT: {
          break;
        }
        case RIGHT: {

        }
        default: {

        }
      }
      return res;
    }
  }
}

}//gqpeps

#endif //GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H
