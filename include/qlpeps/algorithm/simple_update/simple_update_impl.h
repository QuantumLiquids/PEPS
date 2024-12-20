// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-25
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. Implementation for abstract class of simple update.
*/

#ifndef QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H
#define QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H

namespace qlpeps {

using namespace qlten;

//helper
std::vector<size_t> DuplicateElements(const std::vector<size_t> &input) {
  std::vector<size_t> output;
  output.reserve(input.size() * 2); // Reserve space for efficiency

  for (size_t element : input) {
    output.push_back(element);
    output.push_back(element);
  }

  return output;
}

/**
 * Calculate the exp(-tau*H) with tau a small number by Taylor expansion
 *
 * @param tau
 * @param ham
 *  N site Hamiltonian term with 2*N tensor indexes. The indexes is ordered specifically.
 *  E.g. for 2-site Hamiltonian, the hamiltonian indexes is ordered in the following figure.
 *  The 0,2 indexes will be projected to peps, and 0,1 legs for site 1.
 *
 *          1         3
 *          |         |
 *          v         v
 *          |---ham---|
 *          v         v
 *          |         |
 *          0         2
 *
 *  One more example is the 3-site hamiltonian:
 *          1         3         5
 *          |         |         |
 *          v         v         v
 *          |--------ham--------|
 *          v         v         v
 *          |         |         |
 *          0         2         4
 * @return
 *  2*N indexes tensor which has different order with that of the Hamiltonian.
 *  E.g. for 3-site hamiltonian, the indexes order of the return tensor is
 *
 *          3         4         5
 *          |         |         |
 *          v         v         v
 *          |----exp(-tau*H)----|
 *          v         v         v
 *          |         |         |
 *          0         1         2
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> TaylorExpMatrix(const double tau, const QLTensor<TenElemT, QNT> &ham) {
  using Tensor = QLTensor<TenElemT, QNT>;
  const size_t N = ham.Rank() / 2;
  Tensor ham_scale = -tau * ham;
  //transpose so that in leg first.
  std::vector<size_t> transpose_axes(2 * N, 0);
  for (size_t i = 0; i < N; i++) {
    transpose_axes[i] = 2 * i;
    transpose_axes[N + i] = 2 * i + 1;
  }
  ham_scale.Transpose(transpose_axes);
  //generate the Identity tensor
  Tensor id = Tensor(ham.GetIndexes());
  ShapeT shape = ham_scale.GetShape();
  shape.erase(shape.begin() + N, shape.end());
  std::vector<CoorsT> all_coors = GenAllCoors(shape);
  for (const auto &coor : all_coors) {
    id(DuplicateElements(coor)) = 1.0;
  }
  id.Transpose(transpose_axes);
  if (Tensor::IsFermionic() && id.GetIndex(0).GetDir() == OUT) {
    id.ActFermionPOps();
  }

  std::vector<Tensor> taylor_terms = {id, ham_scale};
  taylor_terms.reserve(kMaxTaylorExpansionOrder);
  std::vector<size_t> ctrct_axes1(N), ctrct_axes2(N);
  std::iota(ctrct_axes1.begin(), ctrct_axes1.end(), N);
  std::iota(ctrct_axes2.begin(), ctrct_axes2.end(), 0);
  for (size_t n = 2; n < kMaxTaylorExpansionOrder; n++) {
    Tensor tmp;
    Contract(&taylor_terms.back(), ctrct_axes1, &ham_scale, ctrct_axes2, &tmp);
    tmp *= 1.0 / double(n);
    taylor_terms.emplace_back(tmp);
    if (tmp.GetQuasi2Norm() < kDoubleEpsilon) {
      std::cout << "calculate the evolution gate taylor series order: " << n << std::endl;
      break;
    }
    if (n == kMaxTaylorExpansionOrder - 1) {
      std::cout << "warning: taylor expansions for evolution gate do not converge "
                << "with precision: " << tmp.GetQuasi2Norm() << "." << std::endl;
    }
  }
  Tensor expH = taylor_terms[0];
  for (size_t n = 1; n < taylor_terms.size(); n++) {
    expH += taylor_terms[n];
  }
  return expH;
}

/**
 *
 * @param update_para
 * @param ham_nn
 *          1         3
 *          |         |
 *          ^         ^
 *          |---ham---|
 *          ^         ^
 *          |         |
 *          0         2
 * @param peps_initial
 * @return
 */
template<typename TenElemT, typename QNT>
SimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                                          const PEPST &peps_initial)
    : Executor(), lx_(peps_initial.Cols()), ly_(peps_initial.Rows()),
      update_para(update_para), peps_(peps_initial) {
  std::cout << "\n";
  std::cout << "=====> SIMPLE UPDATE PROGRAM FOR Square-Lattice PEPS <=====" << "\n";
  std::cout << std::setw(40) << "System size (lx, ly) : " << "(" << lx_ << ", " << ly_ << ")\n";
  std::cout << std::setw(40) << "SquareLatticePEPS bond dimension : " << update_para.Dmin << "/" << update_para.Dmax
            << "\n";
  std::cout << std::setw(40) << "Trotter step : " << update_para.tau << "\n";

  std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
  std::cout << std::setw(40) << "The number of threads per processor : " << hp_numeric::GetTensorManipulationThreads()
            << "\n";
  SetStatus(qlten::INITED);
}

template<typename TenElemT, typename QNT>
void SimpleUpdateExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(qlten::EXEING);
  SetEvolveGate_();
  for (size_t step = 0; step < update_para.steps; step++) {
    std::cout << "step = " << step << "\t";
    estimated_energy_ = SimpleUpdateSweep_();
  }
  SetStatus(qlten::FINISH);
}

template<typename QNT>
void PrintLambda(const QLTensor<QLTEN_Double, QNT> &lambda) {
  std::cout << std::setprecision(4) << std::scientific;

  // Extract the diagonal elements of lambda into a vector
  std::vector<double> diagonal_elements(lambda.GetShape()[0]);
  for (size_t i = 0; i < lambda.GetShape()[0]; i++) {
    diagonal_elements[i] = lambda({i, i});
  }

  // Sort the diagonal elements in descending order
  std::sort(diagonal_elements.begin(), diagonal_elements.end(), std::greater<double>());

  // Print the sorted elements
  std::cout << "[";
  for (const auto &element : diagonal_elements) {
    std::cout << " " << element;
  }
  std::cout << " ]" << std::endl;
}
}//qlpeps;
#endif //QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H
