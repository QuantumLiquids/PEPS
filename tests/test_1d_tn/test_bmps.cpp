// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-08
*
* Description: QuantumLiquids/PEPS project. Unittests for TensorNetwork2D
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/one_dim_tn/bmps.h"
#include "qlpeps/utilities.h"     // For test helper functions

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::U1QN;
using QNSctT = qlten::special_qn::U1QNSector;
using IndexT = qlten::Index<QNT>;
using DQLTensor = qlten::QLTensor<qlten::QLTEN_Double, QNT>;

class BMPSTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create commonly used quantum numbers
    qn0 = QNT({QNSctT("Sz", 0)});
    qn1 = QNT({QNSctT("Sz", 1)});
    qn_1 = QNT({QNSctT("Sz", -1)});
    
    // Create physical index
    phys_idx = IndexT({
      QNSctT("Sz", 1), 1,
      QNSctT("Sz", -1), 1
    }, TenIndexDirType::OUT);
    
    // Create virtual index
    virtual_idx = IndexT({
      QNSctT("Sz", 0), 2,
      QNSctT("Sz", 1), 1,
      QNSctT("Sz", -1), 1
    }, TenIndexDirType::OUT);
  }

  QNT qn0, qn1, qn_1;
  IndexT phys_idx, virtual_idx;
};

// Test basic construction
TEST_F(BMPSTest, BasicConstruction) {
  const size_t length = 4;
  BMPS<DQLTensor> bmps(length);
  EXPECT_EQ(bmps.size(), length);
}

// Test simple canonical MPS
TEST_F(BMPSTest, SimpleCanonicalMPS) {
  const size_t length = 2;
  BMPS<DQLTensor> bmps(length);
  
  // Construct two simple site tensors
  auto idx_in = InverseIndex(virtual_idx);
  
  DQLTensor site1({phys_idx, virtual_idx, idx_in});
  site1({0,0,0}) = 1.0;
  site1({1,1,1}) = 1.0;
  
  DQLTensor site2({phys_idx, virtual_idx, idx_in});
  site2({0,0,0}) = 1.0;
  site2({1,1,1}) = 1.0;
  
  bmps[0] = site1;
  bmps[1] = site2;
  
  // Test normalization
  bmps.LeftCanonicalize();
  
  // Check properties after normalization
  // 1. Check left canonicality
  for (size_t i = 0; i < length; i++) {
    auto contracted = Contract(bmps[i], bmps[i], {{0,0}}); // Contract physical indices
    // Check if the result is close to the identity matrix
    // TODO: Add specific check code
  }
}

// Test expectation value of simple physical quantities
TEST_F(BMPSTest, ExpectationValue) {
  const size_t length = 2;
  BMPS<DQLTensor> bmps(length);
  
  // Construct a simple sz operator
  DQLTensor sz({phys_idx, InverseIndex(phys_idx)});
  sz({0,0}) = 0.5;   // |up><up|
  sz({1,1}) = -0.5;  // |down><down|
  
  // Construct a known state (e.g., all up)
  // TODO: Initialize bmps to an all-up state
  
  // Calculate the expectation value of sz
  // TODO: Implement expectation value calculation and verify results
}

// Test compression/truncation
TEST_F(BMPSTest, Truncation) {
  const size_t length = 4;
  const size_t D_large = 8;
  const size_t D_small = 4;
  
  // Create a large-dimensional BMPS
  BMPS<DQLTensor> bmps(length);
  // TODO: Initialize a large-dimensional BMPS
  
  // Perform truncation
  double trunc_err = 1e-6;
  // TODO: Implement truncation operation
  
  // Verify the dimensions and precision after truncation
  // TODO: Add verification code
}

// Test quantum number conservation
TEST_F(BMPSTest, QuantumNumberConservation) {
  const size_t length = 3;
  BMPS<DQLTensor> bmps(length);
  
  // Construct a state with specific quantum numbers
  // TODO: Initialize a BMPS with specific quantum numbers
  
  // Verify the total quantum number
  QNT total_qn = bmps.GetTotalQN();
  // TODO: Verify quantum number conservation
}