#  SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
#  Description: QuantumLiquids/PEPS project. CMake file to control unittest.
# Including the setting for MKL compile flags and link flags.

option(QLTEN_USE_OPENBLAS "Use openblas rather mkl" OFF)

if (NOT QLTEN_USE_OPENBLAS)
    if (APPLE)
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
            #Need test
            set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
            # May not work
            set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -liomp5 -lpthread -lm -ldl)
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
            # Note as of Intel oneAPI 2021.2, "source /opt/intel/oneapi/compiler/latest/env/vars.sh"
            set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -Wl, -rpath $ENV{CMPLR_ROOT}/mac/compiler/lib/libiomp5.dylib -liomp5 -lpthread -lm -ldl)
        endif ()
    elseif (UNIX)
        #UNIX include APPLE, but we except it here
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
            set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
            # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
            set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
            set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
        endif ()
    endif ()
else ()
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    #        FIND_PACKAGE(BLAS REQUIRED)
    #        FIND_PACKAGE(LAPACK REQUIRED)
    set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/")
    set(Lapack_ROOT "/opt/homebrew/opt/lapack")
    message(${OpenBLAS_ROOT})
    set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_ROOT}/include")
    set(OpenBLAS_LIBRARIES "${OpenBLAS_ROOT}/lib/libblas.dylib")
    message(${OpenBLAS_LIBRARIES})
    set(MATH_LIB_COMPILE_FLAGS -I${OpenBLAS_INCLUDE_DIRS} -pthread)
    set(MATH_LIB_LINK_FLAGS ${OpenBLAS_LIBRARIES} ${OpenBLAS_ROOT}/lib/liblapack.dylib -lm -lpthread -ldl -fopenmp -lclapack)
endif ()


find_package(MPI REQUIRED)

#set omp flag
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(OMP_FLAGS -qopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(OMP_FLAGS -fopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(OMP_FLAGS -fopenmp)
endif ()

if (U1SYM)
    add_definitions(-DU1SYM)
endif ()

macro(add_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )

    target_include_directories(${TEST_NAME}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
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
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
    # Compile for QLTEN_Double
    add_executable(${TEST_NAME}_double ${TEST_SRC})
    target_compile_options(${TEST_NAME}_double
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS})
    target_include_directories(${TEST_NAME}_double
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_compile_definitions(${TEST_NAME}_double PRIVATE TEN_ELEM_TYPE=QLTEN_Double)
    target_link_libraries(${TEST_NAME}_double
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIB_FLAGS}")
    add_test(NAME ${TEST_NAME}_double
            COMMAND ${TEST_NAME}_double ${INPUT_ARGS}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${TEST_NAME}_double PROPERTIES FOLDER tests)

    # Compile for QLTEN_Complex
    add_executable(${TEST_NAME}_complex ${TEST_SRC})
    target_compile_options(${TEST_NAME}_complex
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS})
    target_include_directories(${TEST_NAME}_complex
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_compile_definitions(${TEST_NAME}_complex PRIVATE TEN_ELEM_TYPE=QLTEN_Complex)
    target_link_libraries(${TEST_NAME}_complex
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
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
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM INPUT_ARGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${QLPEPS_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${QLTENSOR_HEADER_PATH}
            PRIVATE ${QLMPS_HEADER_PATH}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
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
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test Configuration class
add_unittest(test_configuration
        "test_2d_tn/test_configuration.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
## Test 2D tensor networks.
add_unittest(test_peps
        "test_2d_tn/test_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_split_index_tps
        "test_2d_tn/test_split_index_tps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_tn2d
        "test_2d_tn/test_tensornetwork2d.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "/Users/wanghaoxin/GitHub/PEPS/tests/test_data/"
)
add_unittest(test_arnoldi
        "test_2d_tn/test_arnoldi.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test monte carlo tools
add_unittest(test_statistics
        "test_monte_carlo_tools/test_statistics.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_mpi_unittest(test_statistics_mpi
        "test_monte_carlo_tools/test_statistics_mpi.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "3" ""
)
add_unittest(test_non_detailed_balance_mcmc
        "test_monte_carlo_tools/test_non_detailed_balance_mcmc.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
## Test algorithms
# Test simple update
add_two_type_unittest(test_simple_update
        "test_algorithm/test_simple_update.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)

add_unittest(test_fermion_simple_update
        "test_algorithm_fermion/test_fermion_simple_and_loop_update.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
)
add_two_type_unittest(test_loop_update
        "test_algorithm/test_loop_update.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)

# Test vmc-peps
add_two_type_unittest(test_vmc_peps
        "test_algorithm/test_vmc_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)

add_mpi_unittest_run(test_vmc_peps_double "4" "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json")
add_mpi_unittest_run(test_vmc_peps_complex "4" "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json")


add_unittest(test_fermion_vmc_peps
        "test_algorithm_fermion/test_fermion_vmc_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
)
add_unittest(test_fermion_peps_exact_summation
        "test_algorithm_fermion/test_fermion_exact_optimization.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" " "
)
add_unittest(test_square_heisenberg
        "test_algorithm/test_square_heisenberg.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "4"
)

add_two_type_unittest(test_measure
        "test_algorithm/test_mc_peps_measure.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)


add_two_type_unittest(test_fermion_measure
        "test_algorithm_fermion/test_fermion_mc_measure.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm_fermion/test_params.json"
)

add_unittest(test_fermion_mc_update
        "test_algorithm_fermion/test_fermion_mc_update.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_data/"
)


## Test utility
add_unittest(test_conjugate_gradient_solver
        "test_utility/test_conjugate_gradient_solver.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        ""
)

add_mpi_unittest(test_conjugate_gradient_mpi_solver
        "test_utility/test_conjugate_gradient_mpi_solver.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "3"
        ""
)
