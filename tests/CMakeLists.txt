# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
# Description: QuantumLiquids/PEPS project. CMake file to control unittest.
# Including the setting for compile flags and link flags for test cases

# Define source directory path for tests
add_definitions(-DTEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}/tests") # used in test code

# Test configuration options
option(BUILD_SLOW_TESTS "Build slow integration tests (may take several minutes)" OFF)

# Number of MPI processes for slow tests (can be overridden per environment)
set(SLOW_TESTS_MPI_PROCS 56 CACHE STRING "MPI processes used to run slow tests")

# Print test configuration
if(BUILD_SLOW_TESTS)
    option(ENABLE_INTEGRATION_SINGLE_RANK "Register single-rank integration tests (non-MPI)" OFF)
    message(STATUS "Slow tests: ENABLED")
else()
    message(STATUS "Slow tests: DISABLED")
endif()


include(../cmake/Modules/MathBackend.cmake)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED) # link hptt necessary

set(MATH_LIB_LINK_FLAGS "BLAS::BLAS" "LAPACK::LAPACK")

if (U1SYM)
    add_definitions(-DU1SYM)
endif ()

macro(add_unittest
        TEST_NAME TEST_SRC MATH_HEADER LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
    # NOTE: This macro links GTest::Main (gtest_main). If your TEST_SRC defines its own main(),
    # don't use this macro (or remove GTest::Main), otherwise you can get duplicate-main link errors
    # depending on how gtest_main is built/linked.
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_include_directories(${TEST_NAME}
            PRIVATE ${MATH_HEADER}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE MPI::MPI_CXX)
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            OpenMP::OpenMP_CXX
            MPI::MPI_CXX
            ${LINK_LIBS} "${LINK_LIB_FLAGS}")

    add_test(
            NAME ${TEST_NAME}
            COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    # Print the value of INPUT_ARGS
endmacro()

macro(add_two_type_unittest
        TEST_NAME TEST_SRC MATH_HEADER LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
    # Compile for QLTEN_Double
    add_executable(${TEST_NAME}_double ${TEST_SRC})
    target_include_directories(${TEST_NAME}_double
            PRIVATE ${MATH_HEADER}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${TEST_NAME}_double PRIVATE TEN_ELEM_TYPE=QLTEN_Double TEN_ELEM_TYPE_NUM=1)
    target_link_libraries(${TEST_NAME}_double
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            OpenMP::OpenMP_CXX
            MPI::MPI_CXX
            "${LINK_LIB_FLAGS}")
    add_test(NAME ${TEST_NAME}_double
            COMMAND ${TEST_NAME}_double ${INPUT_ARGS}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${TEST_NAME}_double PROPERTIES FOLDER tests)

    # Compile for QLTEN_Complex
    add_executable(${TEST_NAME}_complex ${TEST_SRC})
    target_include_directories(${TEST_NAME}_complex
            PRIVATE ${MATH_HEADER}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${TEST_NAME}_complex PRIVATE TEN_ELEM_TYPE=QLTEN_Complex TEN_ELEM_TYPE_NUM=2)
    target_link_libraries(${TEST_NAME}_complex
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            "${LINK_LIB_FLAGS}")
    add_test(NAME ${TEST_NAME}_complex
            COMMAND ${TEST_NAME}_complex ${INPUT_ARGS}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${TEST_NAME}_complex PROPERTIES FOLDER tests)
endmacro()

macro(add_mpi_unittest_run
        TEST_NAME PROCESSOR_NUM INPUT_ARGS)
    # Check if we have enough processors to run the requested number of MPI processes
    if (MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${PROCESSOR_NUM})
        add_test(NAME ${TEST_NAME}_mpi
                COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM}
                $<TARGET_FILE:${TEST_NAME}>
                ${INPUT_ARGS}
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    else()
        message(STATUS "Skipping MPI test ${TEST_NAME}_mpi: requires ${PROCESSOR_NUM} processes, but only ${MPIEXEC_MAX_NUMPROCS} available")
    endif ()
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

macro(add_mpi_unittest
        TEST_NAME TEST_SRC MATH_HEADER LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM INPUT_ARGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})
    target_include_directories(${TEST_NAME}
            PRIVATE ${MATH_HEADER}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )
    add_mpi_unittest_run(${TEST_NAME} ${PROCESSOR_NUM} "${INPUT_ARGS}")
endmacro()

## Test basis for tensor networks
# Test DuoVMatrix class
add_unittest(test_duomatrix test_2d_tn/test_duomatrix.cpp "" "" "" "")
# Test TenMatrix class
add_unittest(test_ten_mat
        "test_2d_tn/test_ten_matrix.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test Configuration class
add_unittest(test_configuration
        "test_2d_tn/test_configuration.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
## Test 2D tensor networks.
add_unittest(test_peps
        "test_2d_tn/test_peps.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_tps
        "test_2d_tn/test_tps.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_split_index_tps
        "test_2d_tn/test_split_index_tps.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# MPI functionality tests for SplitIndexTPS
add_mpi_unittest(test_split_index_tps_mpi
        "test_2d_tn/test_split_index_tps_mpi.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "3" ""
)
add_unittest(test_tn2d
        "test_2d_tn/test_tensornetwork2d.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        "/Users/wanghaoxin/GitHub/PEPS/tests/test_data/"
)
add_unittest(test_bmps_contractor
        "test_2d_tn/test_bmps_contractor.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_trg_contractor
        "test_2d_tn/test_trg_contractor.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# TRG PBC performance benchmark (for profiling VMC bottlenecks)
add_unittest(test_trg_pbc_benchmark
        "test_2d_tn/test_trg_pbc_benchmark.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_arnoldi
        "test_2d_tn/test_arnoldi.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test monte carlo tools
add_unittest(test_statistics
        "test_monte_carlo_tools/test_statistics.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_mpi_unittest(test_statistics_mpi
        "test_monte_carlo_tools/test_statistics_mpi.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "3" ""
)
add_unittest(test_suwa_todo_update
        "test_monte_carlo_tools/test_suwa_todo_update.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test model solvers
add_unittest(test_tJ_model_solver
        "test_model_solvers/test_tJ_model_solver.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_unittest(test_square_xxz_measurer
        "test_model_solvers/test_square_xxz_measurer.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_unittest(test_hubbard_u1u1_updater
        "test_model_solvers/test_hubbard_u1u1_updater.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# PBC/TRG exact-sum smoke tests (single-rank, no MPI required)
add_unittest(test_tfim_pbc_trg_exact_sum
        "test_optimizer/test_tfim_pbc_trg_exact_sum.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

add_two_type_unittest(test_optimizer_adagrad_exact_sum_tfim_pbc_trg
        "test_optimizer/test_optimizer_adagrad_exact_sum_tfim_pbc_trg.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test algorithms
# NOTE: Test files have been reorganized:
# - All active algorithm tests are now in test_algorithm/
# - Deprecated tests are in test_deprecated/
# - Old test_algorithm_boson/ and test_algorithm_fermion/ directories removed
#
# Test boson simple update
add_two_type_unittest(test_boson_simple_update
        "test_algorithm/test_boson_simple_update.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# Test fermion simple update
add_two_type_unittest(test_fermion_simple_update
        "test_algorithm/test_fermion_simple_update.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)
# DEPRECATED: Loop update test moved to test_deprecated/
# add_two_type_unittest(test_loop_update
#        "test_deprecated/test_loop_update_deprecated.cpp"
#        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
#        ""
# )

# Legacy test_vmc_peps removed - use test_vmc_peps_optimizer instead

# Test optimizer
add_two_type_unittest(test_optimizer
        "test_optimizer/test_optimizer.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# Test VMC PEPS optimizer executor
add_two_type_unittest(test_vmc_peps_optimizer
        "test_algorithm/test_vmc_peps_optimizer.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

add_mpi_unittest_run(test_vmc_peps_optimizer_double "4" " ")
add_mpi_unittest_run(test_vmc_peps_optimizer_complex "4" " ")


## Optimizer exact-sum AdaGrad test (pure optimizer, deterministic gradients)
add_two_type_unittest(test_optimizer_adagrad_exact_sum
        "test_optimizer/test_optimizer_adagrad_exact_sum.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)

# Add MPI test runs for optimizer exact-sum AdaGrad to verify MPI-aware ExactSumEnergyEvaluator
add_mpi_unittest_run(test_optimizer_adagrad_exact_sum_double "4" " ")
add_mpi_unittest_run(test_optimizer_adagrad_exact_sum_complex "4" " ")

# Test pure exact summation evaluator (separated from optimizer complexity)
add_two_type_unittest(test_exact_summation_evaluator
        "test_algorithm/test_exact_summation_evaluator.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)

# Add MPI test runs for exact summation evaluator to verify MPI-aware computation
add_mpi_unittest_run(test_exact_summation_evaluator_double "4" " ")
add_mpi_unittest_run(test_exact_summation_evaluator_complex "4" " ")


## Optimizer exact-sum SGD test (pure optimizer, deterministic gradients)
add_two_type_unittest(test_optimizer_sgd_exact_sum
        "test_optimizer/test_optimizer_sgd_exact_sum.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)
add_mpi_unittest_run(test_optimizer_sgd_exact_sum_double "4" " ")
add_mpi_unittest_run(test_optimizer_sgd_exact_sum_complex "4" " ")

## Optimizer exact-sum Adam test (pure optimizer, deterministic gradients)
add_two_type_unittest(test_optimizer_adam_exact_sum
        "test_optimizer/test_optimizer_adam_exact_sum.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)
add_mpi_unittest_run(test_optimizer_adam_exact_sum_double "4" " ")
add_mpi_unittest_run(test_optimizer_adam_exact_sum_complex "4" " ")

## Optimizer exact-sum L-BFGS strong-Wolfe test (pure optimizer, deterministic gradients)
add_two_type_unittest(test_optimizer_lbfgs_exact_sum
        "test_optimizer/test_optimizer_lbfgs_exact_sum.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)
add_mpi_unittest_run(test_optimizer_lbfgs_exact_sum_double "4" " ")
add_mpi_unittest_run(test_optimizer_lbfgs_exact_sum_complex "4" " ")

# Golden regression (integration): fermionic exact-summation + optimizer signatures
add_two_type_unittest(test_fermion_exact_optimizer_integration_golden
        "test_optimizer/test_fermion_exact_optimizer_integration_golden.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)


# Learning rate scheduler unit tests (header-only, no extra libs)
add_unittest(test_lr_schedulers
        "test_optimizer/test_lr_schedulers.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)


# Optimizer params utility tests
add_unittest(test_optimizer_params
        "test_optimizer/test_optimizer_params.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# Auto step selector policy unit tests (single-rank)
add_unittest(test_auto_step_selector_policy
        "test_optimizer/test_auto_step_selector_policy.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# Spike detection / EMA tracker unit tests (header-only, no extra libs)
add_unittest(test_spike_detection
        "test_optimizer/test_spike_detection.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# Optimizer gradient preprocessing integration (single-rank)
add_unittest(test_optimizer_gradient_preprocessing
        "test_optimizer/test_optimizer_gradient_preprocessing.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# L-BFGS recovery policy unit tests (single-rank)
add_unittest(test_optimizer_lbfgs_recovery_policy
        "test_optimizer/test_optimizer_lbfgs_recovery_policy.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

# L-BFGS fixed-step MC regression tests (boson + fermion, 2x2 systems)
add_two_type_unittest(test_optimizer_lbfgs_mc_regression
        "test_optimizer/test_optimizer_lbfgs_mc_regression.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)

# Fast tests (2x2 systems, quick execution)
add_two_type_unittest(test_mc_peps_measure
        "test_algorithm/test_mc_peps_measure.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        "test_data/tps_square_heisenberg2x2D4Double"
)


# Independent MC energy+gradient evaluator smoke test
add_two_type_unittest(test_mc_energy_grad_evaluator_smoke
        "test_algorithm/test_mc_energy_grad_evaluator_smoke.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# Smoke: PBC TRG contractor + MC updater (trial/accept/reject)
add_two_type_unittest(test_mc_updater_trg_smoke
        "test_algorithm/test_mc_updater_trg_smoke.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# Deterministic MonteCarlo regression (classical Ising via TRG amplitude)
add_two_type_unittest(test_mc_ising_trg_pbc
        "test_algorithm/test_mc_ising_trg_pbc.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# 2x2 classical Ising (h=0) smoke test using MonteCarloMeasure
add_two_type_unittest(test_api_smoke
        "test_algorithm/test_api_smoke.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# Add MPI runs for the API smoke test (lightweight)
add_mpi_unittest_run(test_api_smoke_double "4" " ")
add_mpi_unittest_run(test_api_smoke_complex "4" " ")

# 2x2 spinless fermion exact vs MC evaluator test
add_two_type_unittest(test_mc_energy_grad_evaluator
        "test_algorithm/test_mc_energy_grad_evaluator.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

add_mpi_unittest_run(test_mc_energy_grad_evaluator_double "4" " ")
add_mpi_unittest_run(test_mc_energy_grad_evaluator_complex "4" " ")

# Golden regression: fermionic MC gradient + SR natural-gradient signatures
add_two_type_unittest(test_fermion_mc_sr_golden
        "test_algorithm/test_fermion_mc_sr_golden.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

# MonteCarloEngine tests (configuration rescue, etc.)
# Tests require 4 MPI ranks to test all configuration rescue scenarios
add_mpi_unittest(test_monte_carlo_engine
        "test_algorithm/test_monte_carlo_engine.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
        ""
)

# MPI “death test” (ctest-level): verify MCPEPSMeasurer forwards config_rescue into MonteCarloEngine.
# Expected behavior: MPI_Abort (non-zero exit) AND prints the failure header.
add_executable(test_mcpeps_measurer_config_rescue_wiring_death
        "test_algorithm/test_mcpeps_measurer_config_rescue_wiring_death_mpi.cpp")
target_include_directories(test_mcpeps_measurer_config_rescue_wiring_death
        PRIVATE ${BLAS_INCLUDE_DIRS}
        PRIVATE ${QLPEPS_HEADER_PATH}
        PRIVATE ${hptt_INCLUDE_DIR}
        PRIVATE ${QLTENSOR_HEADER_PATH}
        PRIVATE ${QLMPS_HEADER_PATH}
        PRIVATE ${MPI_CXX_HEADER_DIR})
target_link_libraries(test_mcpeps_measurer_config_rescue_wiring_death
        # This test defines its own main() (MPI/GTest init), so don't link GTest::Main (gtest_main).
        GTest::GTest
        ${hptt_LIBRARY}
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        ${MATH_LIB_LINK_FLAGS}
)
if (MPIEXEC_MAX_NUMPROCS GREATER_EQUAL 2)
    add_test(NAME test_mcpeps_measurer_config_rescue_wiring_death_mpi
            COMMAND ${CMAKE_COMMAND}
            -DMPIEXEC=${MPIEXEC_EXECUTABLE}
            -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
            -DNPROCS=2
            -DEXECUTABLE=$<TARGET_FILE:test_mcpeps_measurer_config_rescue_wiring_death>
            -DEXPECT_REGEX_1=CONFIGURATION.*FAILURE
            -DEXPECT_REGEX_2=Configuration.*rescue.*DISABLED
            -P ${CMAKE_CURRENT_LIST_DIR}/tools/ctest_expect_mpi_abort.cmake
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    set_tests_properties(test_mcpeps_measurer_config_rescue_wiring_death_mpi
            PROPERTIES TIMEOUT 20)
else()
    message(STATUS "Skipping MPI death test test_mcpeps_measurer_config_rescue_wiring_death_mpi: requires 2 processes, but only ${MPIEXEC_MAX_NUMPROCS} available")
endif()


## Test utility
add_unittest(test_conjugate_gradient_solver
        "test_utility/test_conjugate_gradient_solver.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_unittest(test_observable_matrix
        "test_utility/test_observable_matrix.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_mpi_unittest(test_conjugate_gradient_mpi_solver
        "test_utility/test_conjugate_gradient_mpi_solver.cpp"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "3"
        ""
)



if(BUILD_SLOW_TESTS)
    # Slow tests (4x4 systems, long execution)
    add_two_type_unittest(test_boson_measure
            "slow_tests/test_boson_mc_peps_measure.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            "slow_tests/test_data/tps_square_heisenberg4x4D8Double"
    )
    
    add_two_type_unittest(test_fermion_measure
            "slow_tests/test_fermion_mc_peps_measure.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )

    add_unittest(test_fermion_mc_updater
            "slow_tests/test_fermion_mc_updater_large_size.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            "${CMAKE_CURRENT_LIST_DIR}/test_data/"
    )

    # SU + VMC + Measure integration tests
    # Integration tests: only MPI variants enabled by default (single-rank runs omitted)
    # To run all MPI integration tests: ctest -V -L mpi-integration --test-dir build
    add_two_type_unittest(test_square_heisenberg_obc
            "integration_tests/test_square_heisenberg_obc.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_square_heisenberg_obc_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_square_heisenberg_obc_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_square_heisenberg_obc_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_square_heisenberg_obc_complex PROPERTIES LABELS "integration;single;slow")
    add_mpi_unittest_run(test_square_heisenberg_obc_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_heisenberg_obc_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    set_tests_properties(test_square_heisenberg_obc_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_heisenberg_obc_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")

    add_two_type_unittest(test_square_nn_spinless_free_fermion
            "integration_tests/test_square_nn_spinless_free_fermion.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_square_nn_spinless_free_fermion_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_square_nn_spinless_free_fermion_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_square_nn_spinless_free_fermion_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_square_nn_spinless_free_fermion_complex PROPERTIES LABELS "integration;single;slow")
    add_mpi_unittest_run(test_square_nn_spinless_free_fermion_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_nn_spinless_free_fermion_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    set_tests_properties(test_square_nn_spinless_free_fermion_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_nn_spinless_free_fermion_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")

    add_two_type_unittest(test_square_j1j2_xxz_obc
            "integration_tests/test_square_j1j2_xxz_obc.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_square_j1j2_xxz_obc_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_square_j1j2_xxz_obc_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_square_j1j2_xxz_obc_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_square_j1j2_xxz_obc_complex PROPERTIES LABELS "integration;single;slow")


    add_two_type_unittest(test_triangle_heisenberg
            "integration_tests/test_triangle_heisenberg.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_triangle_heisenberg_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_triangle_heisenberg_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_triangle_heisenberg_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_triangle_heisenberg_complex PROPERTIES LABELS "integration;single;slow")

    add_two_type_unittest(test_triangle_j1j2_heisenberg
            "integration_tests/test_triangle_j1j2_heisenberg.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_triangle_j1j2_heisenberg_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_triangle_j1j2_heisenberg_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_triangle_j1j2_heisenberg_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_triangle_j1j2_heisenberg_complex PROPERTIES LABELS "integration;single;slow")

    add_mpi_unittest_run(test_square_j1j2_xxz_obc_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_j1j2_xxz_obc_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    # Legacy test runs removed - functionality merged into test_square_j1j2_xxz_obc
    add_mpi_unittest_run(test_triangle_heisenberg_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_triangle_heisenberg_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_triangle_j1j2_heisenberg_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_triangle_j1j2_heisenberg_complex "${SLOW_TESTS_MPI_PROCS}" " ")

    set_tests_properties(test_square_j1j2_xxz_obc_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_j1j2_xxz_obc_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_triangle_heisenberg_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_triangle_heisenberg_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_triangle_j1j2_heisenberg_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_triangle_j1j2_heisenberg_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")

    add_two_type_unittest(test_square_tj_model
            "integration_tests/test_square_tj_model.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )

    add_two_type_unittest(test_square_tfim_pbc_trg
            "integration_tests/test_square_tfim_pbc_trg.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    add_mpi_unittest_run(test_square_tfim_pbc_trg_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_tfim_pbc_trg_complex "${SLOW_TESTS_MPI_PROCS}" " ")

    add_two_type_unittest(test_square_heisenberg_pbc
            "integration_tests/test_square_heisenberg_pbc.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    add_mpi_unittest_run(test_square_heisenberg_pbc_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_heisenberg_pbc_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    set_tests_properties(test_square_heisenberg_pbc_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_heisenberg_pbc_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")

    add_two_type_unittest(test_square_j1j2_xxz_pbc
            "integration_tests/test_square_j1j2_xxz_pbc.cpp"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
            " "
    )
    add_mpi_unittest_run(test_square_j1j2_xxz_pbc_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_j1j2_xxz_pbc_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    set_tests_properties(test_square_j1j2_xxz_pbc_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_j1j2_xxz_pbc_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    if (NOT ENABLE_INTEGRATION_SINGLE_RANK)
        set_tests_properties(test_square_tj_model_double PROPERTIES DISABLED TRUE)
        set_tests_properties(test_square_tj_model_complex PROPERTIES DISABLED TRUE)
    endif()
    set_tests_properties(test_square_tj_model_double PROPERTIES LABELS "integration;single;slow")
    set_tests_properties(test_square_tj_model_complex PROPERTIES LABELS "integration;single;slow")
    add_mpi_unittest_run(test_square_tj_model_double "${SLOW_TESTS_MPI_PROCS}" " ")
    add_mpi_unittest_run(test_square_tj_model_complex "${SLOW_TESTS_MPI_PROCS}" " ")
    set_tests_properties(test_square_tj_model_double_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
    set_tests_properties(test_square_tj_model_complex_mpi PROPERTIES LABELS "integration;mpi;mpi-integration;slow")
endif()

# =============================================================================
# Developer tools (not registered as ctests)
# =============================================================================
add_executable(gen_tfim_2x2_pbc_simple_update_data
        "tools/gen_tfim_2x2_pbc_simple_update_data.cpp"
)
target_include_directories(gen_tfim_2x2_pbc_simple_update_data
        PRIVATE ${BLAS_INCLUDE_DIRS}
        PRIVATE ${QLPEPS_HEADER_PATH}
        PRIVATE ${hptt_INCLUDE_DIR}
        PRIVATE ${QLTENSOR_HEADER_PATH}
        PRIVATE ${QLMPS_HEADER_PATH}
        PRIVATE MPI::MPI_CXX
)
target_link_libraries(gen_tfim_2x2_pbc_simple_update_data
        ${hptt_LIBRARY}
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        "${MATH_LIB_LINK_FLAGS}"
)
set_target_properties(gen_tfim_2x2_pbc_simple_update_data PROPERTIES FOLDER tools)
