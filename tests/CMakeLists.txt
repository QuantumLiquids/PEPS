#  SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
#  Description: GraceQ/VMC-PEPS project. CMake file to control unittest.
# Including the setting for MKL compile flags and link flags.

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


find_package(Boost
        1.76.0
        REQUIRED
        COMPONENTS serialization mpi)

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
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${GQTENSOR_HEADER_PATH}
            PRIVATE ${GQMPS2_HEADER_PATH}
            PRIVATE ${GQPEPS_HEADER_PATH}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
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


macro(add_mpi_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM INPUT_ARGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${GQTENSOR_HEADER_PATH}
            PRIVATE ${GQMPS2_HEADER_PATH}
            PRIVATE ${GQPEPS_HEADER_PATH}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )

    if (NOT (MPIEXEC_MAX_NUMPROCS MATCHES "1"))
        add_test(
                NAME ${TEST_NAME}
                COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM}
                $<TARGET_FILE:${TEST_NAME}>
                ${INPUT_ARGS}
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif ()
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


macro(compile_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${GQTENSOR_HEADER_PATH}
            PRIVATE ${GQMPS2_HEADER_PATH}
            PRIVATE ${GQPEPS_HEADER_PATH}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

### Test sites vector.
#add_unittest(test_site_vec test_site_vec.cc "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "")
#

## Test basis for tensr networks
# Test DuoVMatrix class
add_unittest(test_duomatrix test_2d_tn/test_duomatrix.cpp "" "" "" "")
# Test TenMatrix class
add_unittest(test_ten_mat
        "test_2d_tn/test_ten_matrix.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test 2D tensor networks.
add_unittest(test_peps
        "test_2d_tn/test_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_tn2d
        "test_2d_tn/test_tensornetwork2d.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test algorithms
# Test simple update
add_unittest(test_simple_update
        "test_algorithm/test_simple_update.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)
# Test vmc-peps
add_unittest(test_vmc_peps
        "test_algorithm/test_vmc_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)
add_mpi_unittest(test_vmc_peps_mpi
        "test_algorithm/test_vmc_peps.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "2"
        "${CMAKE_CURRENT_LIST_DIR}/test_algorithm/test_params.json"
)


#
## Test mpi two site update
#add_mpi_unittest(test_mpi_two_site_update_finite_vmps
#        "test_mpi_algorithm/test_mpi_two_site_update_finite_vmps.cc"
#        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
#        )
