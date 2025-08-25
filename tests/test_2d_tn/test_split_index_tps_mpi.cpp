// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-29
*
* Description: QuantumLiquids/PEPS project. MPI functionality tests for SplitIndexTPS
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlten/framework/hp_numeric/mpi_fun.h"
#include <filesystem>
#include <mpi.h>

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DTPS = TPS<QLTEN_Double, U1QN>;
using DSITPS = SplitIndexTPS<QLTEN_Double, U1QN>;
using DTensor = QLTensor<QLTEN_Double, U1QN>;
using CTPS = TPS<QLTEN_Complex, U1QN>;
using CSITPS = SplitIndexTPS<QLTEN_Complex, U1QN>;
using CTensor = QLTensor<QLTEN_Complex, U1QN>;

class SplitIndexTPSMPITest : public testing::Test {
protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    
    qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
    pb_out = IndexT({
                      QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                      QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)
                    },
                    TenIndexDirType::OUT
    );
    pb_in = InverseIndex(pb_out);
    
    Lx = 4;
    Ly = 4;
    D = 6;
    N = Lx * Ly;
    
    tps_path = "test_mpi_s_tps_rank" + std::to_string(rank_);
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(tps_path)) {
      std::filesystem::remove_all(tps_path);
    }
  }

  DTPS CreateRandTestTPS() {
    DTPS tps(Ly, Lx);
    size_t d_sec = D / 3;
    IndexT virt_out_idx = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), D - 2 * d_sec)},
                                 TenIndexDirType::OUT);
    IndexT virt_in_idx = InverseIndex(virt_out_idx);

    IndexT left_idx = virt_in_idx;
    IndexT lower_idx = virt_out_idx;
    IndexT right_idx = virt_out_idx;
    IndexT upper_idx = virt_in_idx;

    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        tps({row, col}) = DTensor({left_idx, lower_idx, right_idx, upper_idx, pb_out});
        tps({row, col}).Random(qn0);
      }
    }
    return tps;
  }

  CSITPS CreateRandTestComplexSITPS() {
    CTPS tps(Ly, Lx);
    size_t d_sec = D / 3;
    IndexT virt_out_idx = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), D - 2 * d_sec)},
                                 TenIndexDirType::OUT);
    IndexT virt_in_idx = InverseIndex(virt_out_idx);

    IndexT left_idx = virt_in_idx;
    IndexT lower_idx = virt_out_idx;
    IndexT right_idx = virt_out_idx;
    IndexT upper_idx = virt_in_idx;

    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        tps({row, col}) = CTensor({left_idx, lower_idx, right_idx, upper_idx, pb_out});
        tps({row, col}).Random(qn0);
      }
    }
    return CSITPS(tps);
  }

  int rank_, size_;
  U1QN qn0;
  IndexT pb_out, pb_in;
  size_t Lx, Ly, D, N;
  std::string tps_path;
};

/**
 * Test BroadCast functionality for double SplitIndexTPS
 * This tests the primary Bcast implementation in split_index_tps_impl.h
 */
TEST_F(SplitIndexTPSMPITest, TestMPIBroadcastDouble) {
  DSITPS master_sitps, local_sitps;
  
  // Master rank creates the original data
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    DTPS master_tps = CreateRandTestTPS();
    master_sitps = DSITPS(master_tps);
  }
  
  // Broadcast using the SplitIndexTPS BroadCast function
  qlpeps::MPI_Bcast(master_sitps, MPI_COMM_WORLD, 0);
  
  // Verify all ranks received the same data
  local_sitps = master_sitps;
  
  // Basic validation
  EXPECT_EQ(local_sitps.rows(), Ly);
  EXPECT_EQ(local_sitps.cols(), Lx);
  EXPECT_EQ(local_sitps.PhysicalDim(), 2);
  
  // Check tensors are not default-constructed
  for (size_t row = 0; row < Ly; ++row) {
    for (size_t col = 0; col < Lx; ++col) {
      for (size_t i = 0; i < local_sitps.PhysicalDim(); ++i) {
        EXPECT_FALSE(local_sitps({row, col})[i].IsDefault());
        EXPECT_GT(local_sitps({row, col})[i].Get2Norm(), 0.0);
      }
    }
  }
  
  // Cross-rank consistency check: compute a global norm and compare
  double local_norm_sq = local_sitps.NormSquare();
  double global_norm_sq;
  MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  // All ranks should contribute the same norm
  EXPECT_NEAR(global_norm_sq, local_norm_sq * size_, 1e-12);
}

/**
 * Test BroadCast functionality for complex SplitIndexTPS
 */
TEST_F(SplitIndexTPSMPITest, TestMPIBroadcastComplex) {
  CSITPS master_sitps, local_sitps;
  
  // Master rank creates the original data
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    master_sitps = CreateRandTestComplexSITPS();
  }
  
  // Broadcast using the SplitIndexTPS BroadCast function
  qlpeps::MPI_Bcast(master_sitps, MPI_COMM_WORLD, 0);
  
  // Verify all ranks received the same data
  local_sitps = master_sitps;
  
  // Basic validation
  EXPECT_EQ(local_sitps.rows(), Ly);
  EXPECT_EQ(local_sitps.cols(), Lx);
  EXPECT_EQ(local_sitps.PhysicalDim(), 2);
  
  // Check tensors are properly initialized
  for (size_t row = 0; row < Ly; ++row) {
    for (size_t col = 0; col < Lx; ++col) {
      for (size_t i = 0; i < local_sitps.PhysicalDim(); ++i) {
        EXPECT_FALSE(local_sitps({row, col})[i].IsDefault());
        EXPECT_GT(local_sitps({row, col})[i].Get2Norm(), 0.0);
      }
    }
  }
}

/**
 * Test MPI_Send and MPI_Recv functionality for SplitIndexTPS
 */
TEST_F(SplitIndexTPSMPITest, TestMPISendRecv) {
  if (size_ < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int sender_rank = 0;
  const int receiver_rank = 1;
  
  DSITPS original_sitps, received_sitps;
  
  if (rank_ == sender_rank) {
    // Sender creates and sends data
    DTPS original_tps = CreateRandTestTPS();
    original_sitps = DSITPS(original_tps);
    
    // Send to receiver
    qlpeps::MPI_Send(original_sitps, receiver_rank, MPI_COMM_WORLD, 42);
    
  } else if (rank_ == receiver_rank) {
    // Receiver gets the data
    MPI_Status status = qlpeps::MPI_Recv(received_sitps, sender_rank, MPI_COMM_WORLD, 42);
    
    // Verify received data
    EXPECT_EQ(received_sitps.rows(), Ly);
    EXPECT_EQ(received_sitps.cols(), Lx);
    EXPECT_EQ(received_sitps.PhysicalDim(), 2);
    
    // Check tensors are properly received
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        for (size_t i = 0; i < received_sitps.PhysicalDim(); ++i) {
          EXPECT_FALSE(received_sitps({row, col})[i].IsDefault());
          EXPECT_GT(received_sitps({row, col})[i].Get2Norm(), 0.0);
        }
      }
    }
    
    EXPECT_EQ(status.MPI_SOURCE, sender_rank);
    EXPECT_EQ(status.MPI_TAG, 42);
  }
  
  // Synchronize all processes
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test behavior with different MPI communicators
 */
TEST_F(SplitIndexTPSMPITest, TestDifferentCommunicators) {
  if (size_ < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a sub-communicator with only first 2 ranks
  MPI_Comm sub_comm;
  int color = (rank_ < 2) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank_, &sub_comm);
  
  if (sub_comm != MPI_COMM_NULL) {
    int sub_rank, sub_size;
    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_size);
    
    DSITPS test_sitps;
    
    if (sub_rank == 0) {
      DTPS test_tps = CreateRandTestTPS();
      test_sitps = DSITPS(test_tps);
    }
    
    // Broadcast within sub-communicator
    qlpeps::MPI_Bcast(test_sitps, sub_comm, 0);
    
    // Verify all ranks in sub-communicator have the data
    EXPECT_EQ(test_sitps.rows(), Ly);
    EXPECT_EQ(test_sitps.cols(), Lx);
    EXPECT_EQ(test_sitps.PhysicalDim(), 2);
    
    MPI_Comm_free(&sub_comm);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test error handling and edge cases
 */
TEST_F(SplitIndexTPSMPITest, TestEdgeCases) {
  // Test with empty TPS
  DSITPS empty_sitps;
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    empty_sitps = DSITPS(2, 2, 2);  // Small but valid TPS
  }
  
  // This should work without errors
  EXPECT_NO_THROW(qlpeps::MPI_Bcast(empty_sitps, MPI_COMM_WORLD, 0));
  
  // Verify dimensions are broadcast correctly
  EXPECT_EQ(empty_sitps.rows(), 2);
  EXPECT_EQ(empty_sitps.cols(), 2);
  EXPECT_EQ(empty_sitps.PhysicalDim(), 2);
}

/**
 * Test consistency between GroupIndices and MPI operations
 */
TEST_F(SplitIndexTPSMPITest, TestGroupIndicesConsistency) {
  DSITPS sitps;
  DTPS original_tps(0, 0), reconstructed_tps(0, 0);
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    original_tps = CreateRandTestTPS();
    sitps = DSITPS(original_tps);
  }
  
  // Broadcast SplitIndexTPS
  qlpeps::MPI_Bcast(sitps, MPI_COMM_WORLD, 0);
  
  // All ranks reconstruct TPS
  reconstructed_tps = sitps.GroupIndices(pb_out);
  
  // Verify consistency on all ranks
  EXPECT_EQ(reconstructed_tps.rows(), Ly);
  EXPECT_EQ(reconstructed_tps.cols(), Lx);
  
  // Cross-rank validation
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Master can compare with original
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        auto diff_ten = original_tps({row, col}) + (-reconstructed_tps({row, col}));
        EXPECT_NEAR(diff_ten.Get2Norm(), 0.0, 1e-12);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
