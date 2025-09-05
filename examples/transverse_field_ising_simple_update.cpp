// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Description: Minimal 4x4 transverse-field Ising ( - ZZ - h X ) Simple Update example.
*              Uses uniform nearest-neighbor ZZ bond term and uniform on-site X field.
*              Designed to be simple/fast for the getting_started.md.
*/

#include <iostream>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"

using namespace qlten;
using namespace qlpeps;

int main(int argc, char* argv[]) {

  try {
    using QNT = qlten::special_qn::TrivialRepQN;
    using TenElemT = QLTEN_Double;
    using IndexT = Index<QNT>;
    using QNSctT = QNSector<QNT>;
    using Tensor = QLTensor<TenElemT, QNT>;

    const size_t Lx = 4;
    const size_t Ly = 4;

    // Hamiltonian: H = - sum_<i,j in NN> Z_i Z_j - h * sum_i X_i
    const double h = 0.5;

    // Physical index for spin-1/2 with trivial symmetry (dimension 2)
    IndexT pb_out({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
    IndexT pb_in = InverseIndex(pb_out);

    // Build on-site Pauli operators (rank-2): X and Z
    Tensor opX({pb_in, pb_out});
    opX({0, 1}) = 1.0; // |0><1|
    opX({1, 0}) = 1.0; // |1><0|

    Tensor opZ({pb_in, pb_out});
    opZ({0, 0}) = 1.0;  // |0><0|
    opZ({1, 1}) = -1.0; // |1><1|

    // Two-site bond term -ZZ (rank-4): (in1,out1,in2,out2)
    Tensor ham_zz;
    Contract(&opZ, {}, &opZ, {}, &ham_zz); // outer product yields correct index order
    ham_zz *= -1.0;

    // Uniform on-site X field: - h * X
    Tensor ham_onsite = opX;
    ham_onsite *= -h;

    // Initial 4x4 PEPS state (product state initializer)
    using PEPST = SquareLatticePEPS<TenElemT, QNT>;
    PEPST peps0(pb_out, Ly, Lx);
    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, 0));
    peps0.Initial(activates);

    // Simple Update parameters (small and fast)
    SimpleUpdatePara su_para(
        /*steps=*/100,  // number of Trotter steps
        /*tau=*/0.05,   // step size
        /*Dmin=*/1,
        /*Dmax=*/4,
        /*Trunc_err=*/1e-14);

    // Execute Simple Update (NN model with optional on-site term)
    SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> executor(
        su_para, peps0, ham_zz, ham_onsite);

    std::cout << "[TFI-SU] Start 4x4 Simple Update: H = - ZZ - h X\n";
    std::cout << "  Lx=" << Lx << ", Ly=" << Ly
              << ", h=" << h
              << ", steps=" << su_para.steps
              << ", tau=" << su_para.tau
              << ", Dmax=" << su_para.Dmax << std::endl;

    executor.Execute();

    std::cout << "[TFI-SU] Finished. Dumping PEPS...\n";
    executor.DumpResult("peps", /*release_mem=*/true);
  } catch (const std::exception &e) {
    std::cerr << "[TFI-SU] Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}


