# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
# Description: QuantumLiquids/PEPS project. CMake file to control unittest.
# Including the setting for compile flags and link flags for test cases

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(BLA_VENDOR Intel10_64lp_seq)   #FOR VENDOR CHOICE: https://cmake.org/cmake/help/latest/module/FindBLAS.html
    set(BLAS_INCLUDE_DIR "$ENV{MKLROOT}/include")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64") # (e.g., Apple Silicon)
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/") #work for homebrew installed openblas in MacOSX
    set(Lapack_ROOT "/opt/homebrew/opt/lapack")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OpenBLAS_ROOT} ${Lapack_ROOT}) # For find
    set(BLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
endif ()

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
    target_compile_definitions(${TEST_NAME}_double PRIVATE TEN_ELEM_TYPE=QLTEN_Double)
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
    target_compile_definitions(${TEST_NAME}_complex PRIVATE TEN_ELEM_TYPE=QLTEN_Complex)
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
    if (NOT (MPIEXEC_MAX_NUMPROCS MATCHES "1"))
        add_test(NAME ${TEST_NAME}_mpi
                COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM}
                $<TARGET_FILE:${TEST_NAME}>
                ${INPUT_ARGS}
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
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
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test Configuration class
add_unittest(test_configuration
        "test_2d_tn/test_configuration.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
## Test 2D tensor networks.
add_unittest(test_peps
        "test_2d_tn/test_peps.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_split_index_tps
        "test_2d_tn/test_split_index_tps.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_tn2d
        "test_2d_tn/test_tensornetwork2d.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "/Users/wanghaoxin/GitHub/PEPS/tests/test_data/"
)
add_unittest(test_arnoldi
        "test_2d_tn/test_arnoldi.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test monte carlo tools
add_unittest(test_statistics
        "test_monte_carlo_tools/test_statistics.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_mpi_unittest(test_statistics_mpi
        "test_monte_carlo_tools/test_statistics_mpi.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "3" ""
)
add_unittest(test_non_detailed_balance_mcmc
        "test_monte_carlo_tools/test_non_detailed_balance_mcmc.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test model solvers
add_unittest(test_tJ_model_solver
        "test_model_solvers/test_tJ_model_solver.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

## Test algorithms
# Test simple update
add_two_type_unittest(test_simple_update
        "test_algorithm_boson/test_simple_update.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_boson/test_params.json"
)

#add_unittest(test_fermion_simple_update
#        "test_algorithm_fermion/test_fermion_simple_and_loop_update.cpp"
#        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
#        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
#)
add_two_type_unittest(test_loop_update
        "test_algorithm_boson/test_loop_update.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_boson/test_params.json"
)

# Test vmc-peps
add_two_type_unittest(test_vmc_peps
        "test_algorithm_boson/test_vmc_peps.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_boson/test_params.json"
)

add_mpi_unittest_run(test_vmc_peps_double "4" "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_boson/test_params.json")
add_mpi_unittest_run(test_vmc_peps_complex "4" "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_boson/test_params.json")


#add_unittest(test_fermion_vmc_peps
#        "test_algorithm_fermion/test_fermion_vmc_peps.cpp"
#        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
#        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
#)
add_unittest(test_fermion_peps_exact_summation
        "test_algorithm_fermion/test_fermion_exact_optimization.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" " "
)

add_two_type_unittest(test_square_heisenberg
        "test_algorithm_boson/test_square_heisenberg.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

add_two_type_unittest(test_square_j1j2_xxz
        "test_algorithm_boson/test_square_j1_j2_xxz.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

add_mpi_unittest_run(test_square_heisenberg_double "56" " ")
add_mpi_unittest_run(test_square_heisenberg_complex "56" " ")

add_two_type_unittest(test_measure
        "test_algorithm_boson/test_mc_peps_measure.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_data/tps_square_heisenberg4x4D8Double"
)


add_two_type_unittest(test_fermion_measure
        "test_algorithm_fermion/test_fermion_mc_measure.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
)

add_unittest(test_fermion_mc_update
        "test_algorithm_fermion/test_fermion_mc_updater.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_data/"
)

add_two_type_unittest(test_square_nn_spinless_free_fermion
        "test_algorithm_fermion/test_square_nn_spinless_free_fermion.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        " "
)

add_mpi_unittest_run(test_square_nn_spinless_free_fermion_double "56" " ")
add_mpi_unittest_run(test_square_nn_spinless_free_fermion_complex "56" " ")

## Test utility
add_unittest(test_conjugate_gradient_solver
        "test_utility/test_conjugate_gradient_solver.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_mpi_unittest(test_conjugate_gradient_mpi_solver
        "test_utility/test_conjugate_gradient_mpi_solver.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "3"
        ""
)
