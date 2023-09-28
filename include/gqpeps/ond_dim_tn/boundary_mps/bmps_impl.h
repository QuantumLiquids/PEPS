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
    TenVec<Tensor>(size),
    position_(position),
    center_(0),
    tens_cano_type_(size, MPSTenCanoType::RIGHT) {
  assert(local_hilbert_space.dim() == 1);
  Tensor mps_ten = Tensor({index0_in_, local_hilbert_space, index0_out_});
  mps_ten({0, 0, 0}) = 1.0;
  for (size_t i = 0; i < size; i++) {
    this->alloc(i);
    (*this)[i] = mps_ten;
  }
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) {
  return DuoVector<Tensor>::operator[](idx);
}

template<typename TenElemT, typename QNT>
const GQTensor<TenElemT, QNT> &BMPS<TenElemT, QNT>::operator[](const size_t idx) const {
  return DuoVector<Tensor>::operator[](idx);
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> *&BMPS<TenElemT, QNT>::operator()(const size_t idx) {
  return DuoVector<Tensor>::operator()(idx);
}

template<typename TenElemT, typename QNT>
const GQTensor<TenElemT, QNT> *BMPS<TenElemT, QNT>::operator()(const size_t idx) const {
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
  size_t ldims(2);
  auto pq = new Tensor;
  Tensor r;
  QR((*this)(site_idx), ldims, Div((*this)[site_idx]), pq, &r);
  delete (*this)(site_idx);
  (*this)(site_idx) = pq;

  auto pnext_ten = new Tensor;
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
  Tensor u;
  GQTensor<GQTEN_Double, QNT> s;
  auto pvt = new Tensor;
  auto qndiv = Div((*this)[site_idx]);
  mock_gqten::SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  Tensor temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  std::vector<std::vector<size_t>> ctrct_axes = {{2},
                                                 {0}};
  auto pprev_ten = new Tensor;
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
  Contract((*this)(site - 1), &temp_ten, {{2},
                                          {0}}, pnext_ten);
  delete (*this)(site - 1);
  (*this)(site - 1) = pnext_ten;

  //set ten canonical type
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
                                        const double trunc_err, const size_t iter_max,
                                        const CompressMPSScheme &scheme) {
  auto res = this->MultipleMPO(mpo, Dmin, Dmax, trunc_err, iter_max, scheme);
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
                                 const double trunc_err, const size_t iter_max,
                                 const CompressMPSScheme &scheme) const {
  assert(mpo.size() == this->size());
  size_t pre_post = (MPOIndex(position_) + 3) % 4; //equivalent to -1, but work for 0
  size_t next_post = ((size_t) (position_) + 1) % 4;
  switch (scheme) {
    case SVD_COMPRESS: {
      BMPS<TenElemT, QNT> res(position_, this->size());
      IndexT idx1;
      if (MPOIndex(position_) < 2) {
        idx1 = InverseIndex(mpo[0]->GetIndex(pre_post));
      } else {
        idx1 = InverseIndex(mpo.back()->GetIndex(pre_post));
      }
      IndexT idx2 = InverseIndex((*this)[0].GetIndex(0));
      Tensor r = IndexCombine<TenElemT, QNT>(idx1, idx2, IN);
      r.Transpose({2, 0, 1});
      for (size_t i = 0; i < this->size(); i++) {
        Tensor tmp1, tmp2;
        Contract<TenElemT, QNT, false, false>((*this)[i], r, 0, 2, 1, tmp1);
        size_t mpo_idx;
        if (MPOIndex(position_) < 2) {
          mpo_idx = i;
        } else {
          mpo_idx = this->size() - 1 - i;
        }
        Contract<TenElemT, QNT, true, true>(tmp1, *mpo[mpo_idx], 3, pre_post, 2, tmp2);
        res.alloc(i);
        if (i < this->size() - 1) {
          tmp2.Transpose({1, 3, 2, 0});
          QNT mps_div = (*this)[i].Div();
          r = Tensor();
          QR(&tmp2, 2, mps_div, res(i), &r);
        } else {
          auto trivial_idx = tmp2.GetIndex(0);
          Tensor tmp3({InverseIndex(trivial_idx)});
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
    case VARIATION2Site: {
      const double converge_tol = 1e-15;
      size_t N = this->size();
      BMPS::TransferMPO ordered_mpo = mpo;
      if (position_ == RIGHT || position_ == UP) {
        std::reverse(ordered_mpo.begin(), ordered_mpo.end());
      }
      BMPS<TenElemT, QNT> res_init = InitGuessForVariationalMPOMultiplication_(mpo, Dmin, Dmax, trunc_err);
      BMPS<TenElemT, QNT> res_dag(res_init);
      for (size_t i = 0; i < res_dag.size(); i++) {
        res_dag[i].Dag();
      } //initial guess for the result

      std::vector<Tensor> lenvs, renvs;  // from the view of down mps
      lenvs.reserve(N - 1);
      renvs.reserve(N - 1);
      IndexT index2 = InverseIndex((*this)[0].GetIndex(0));
      IndexT index1 = InverseIndex(ordered_mpo[0]->GetIndex(pre_post));
      IndexT index0 = InverseIndex(res_dag[0].GetIndex(0));
      auto lenv0 = Tensor({index0, index1, index2});
      lenv0({0, 0, 0}) = 1;
      lenvs.push_back(lenv0);
      index0 = InverseIndex((*this)[N - 1].GetIndex(2));
      index1 = InverseIndex(ordered_mpo[N - 1]->GetIndex(next_post));
      index2 = InverseIndex(res_dag[N - 1].GetIndex(2));
      auto renv0 = Tensor({index0, index1, index2});
      renv0({0, 0, 0}) = 1;
      renvs.push_back(renv0);

      //initially grow the renvs
      for (size_t i = N - 1; i > 1; i--) {
        Tensor renv_next, temp_ten, temp_ten2;
        Contract<TenElemT, QNT, true, true>((*this)[i], renvs.back(), 2, 0, 1, temp_ten);
        Contract<TenElemT, QNT, false, false>(temp_ten, *ordered_mpo[i], 1, position_, 2, temp_ten2);
        Contract(&temp_ten2, {2, 0}, res_dag(i), {1, 2}, &renv_next);
        renvs.emplace_back(renv_next);
      }

      Tensor s12bond_last;
      for (size_t iter = 0; iter < iter_max; iter++) {
        //left move
        GQTensor<GQTEN_Double, QNT> s;
        for (size_t i = 0; i < N - 2; i++) {
          Tensor tmp[6];
          Contract<TenElemT, QNT, true, true>(lenvs.back(), (*this)[i], 2, 0, 1, tmp[0]);
          Contract<TenElemT, QNT, false, true>(tmp[0], *ordered_mpo[i], 1, pre_post, 2, tmp[1]);

          Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
          Contract<TenElemT, QNT, false, false>(tmp[2], *ordered_mpo[i + 1], 1, position_, 2, tmp[3]);
          Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
          tmp[4].Dag();
          Tensor *pu = new Tensor(), *pvt = new Tensor();
          s = GQTensor<GQTEN_Double, QNT>();
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
          Contract<TenElemT, QNT, false, true>(tmp[0], *ordered_mpo[i], 1, pre_post, 2, tmp[1]);

          Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
          Contract<TenElemT, QNT, false, false>(tmp[2], *ordered_mpo[i + 1], 1, position_, 2, tmp[3]);
          Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
          tmp[4].Dag();
          Tensor *pu = new Tensor(), *pvt = new Tensor();
          s = GQTensor<GQTEN_Double, QNT>();
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
      Contract<TenElemT, QNT, false, true>(tmp[0], *ordered_mpo[i], 1, pre_post, 2, tmp[1]);

      Contract<TenElemT, QNT, true, true>((*this)[i + 1], renvs.back(), 2, 0, 1, tmp[2]);
      Contract<TenElemT, QNT, false, false>(tmp[2], *ordered_mpo[i + 1], 1, position_, 2, tmp[3]);
      Contract(tmp + 1, {2, 0}, tmp + 3, {3, 1}, tmp + 4);
      tmp[4].Dag();
      Tensor u, *pvt = new Tensor();
      GQTensor<GQTEN_Double, QNT> s;
      double actual_trunc_err;
      size_t D;
      SVD(tmp + 4,
          2, res_dag[i].Div(), trunc_err, Dmin, Dmax,
          &u, &s, pvt, &actual_trunc_err, &D
      );

      delete res_dag(i);
      res_dag(i) = new Tensor();
      Contract<TenElemT, QNT, true, true>(u, s, 2, 0, 1, res_dag[i]);
      pvt->Transpose({0, 2, 1});
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
      return res;
    }
    case VARIATION1Site: {

    }
  }
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
    mps_copy.RightCanonicalizeTrunctate(i, 1, 2, 0.0);
  }
  auto multip_init = mps_copy.MultipleMPO(mpo, Dmin, Dmax, trunc_err, 0, SVD_COMPRESS);

#ifndef NDEBUG
  for (size_t i = 0; i < multip_init.size(); i++) {
    assert(multip_init[i].GetActualDataSize() > 0);
  } //initial guess for the result
#endif
  return multip_init;
}

}//gqpeps

#endif //GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_IMPL_H
