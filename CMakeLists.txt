#  SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
#  Description: GraceQ/VMC-PEPS project. Main CMake file.
#
cmake_minimum_required(VERSION 3.12)


project("GraceQ/VMC-PEPS"
        DESCRIPTION "A high-performance VMC-PEPS algorithms library based on GraceQ/tensor"
        LANGUAGES CXX)


# Build type control.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Build output control.
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

option(GQPEPS_BUILD_UNITTEST "Build unittests for GraceQ/vmc-peps." OFF)
option(GQPEPS_BUILD_PROFILER "Build profiler test cases for GraceQ/vmc-peps." OFF)

# Advanced options.
option(GQPEPS_TIMING_MODE "Timing mode." OFF)
if (GQPEPS_TIMING_MODE)
    add_definitions(-DGQPEPS_TIMING_MODE)
endif ()

option(TEST_BUILD_USE_EXTERNAL_HPTT_LIB "Use external hptt library when building unittests for GraceQ/vmc-peps." ON)
option(TEST_BUILD_USE_EXTERNAL_GQTEN "Use external gqten library when building unittests for GraceQ/vmc-peps." ON)
option(TEST_BUILD_USE_EXTERNAL_GQMPS2 "Use external gqmps2 library when building unittests for GraceQ/vmc-peps." ON)


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 17)
add_definitions(-g)


# Installation path
set(GQPEPS_TOOL_SCRIPT_DEST "${CMAKE_INSTALL_PREFIX}/bin")
set(GQPEPS_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")


# Build(Install) library.
set(GQPEPS_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
message("${GQPEPS_HEADER_PATH}")
set(GQPEPS_HEADER_DIR "${GQPEPS_HEADER_PATH}/gqpeps")
install(
        DIRECTORY ${GQPEPS_HEADER_DIR}
        DESTINATION ${GQPEPS_INCLUDE_DEST})


# Build tool scripts
#add_subdirectory(script)


# Build unittests.
if (GQPEPS_BUILD_UNITTEST)
    if (NOT TEST_BUILD_USE_EXTERNAL_HPTT_LIB)
        set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/gqten/external/hptt/include")
        set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/gqten/external/hptt/libhptt.a")
    else ()
        find_path(hptt_INCLUDE_DIR "hptt.h")
        find_library(hptt_LIBRARY "libhptt.a")
    endif ()


    if (NOT TEST_BUILD_USE_EXTERNAL_GQTEN)
        add_subdirectory(external)
        set(GQTENSOR_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/gqten/include")
    else ()
        find_path(GQTENSOR_HEADER_PATH "gqten")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_GQMPS2)
        add_subdirectory(external)
        set(GQMPS2_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/gqten/include")
    else ()
        find_path(GQMPS2_HEADER_PATH "gqmps2")
    endif ()


    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests tests)
endif ()

if (GQPEPS_BUILD_PROFILER)
    if (NOT TEST_BUILD_USE_EXTERNAL_HPTT_LIB)
        set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/gqten/external/hptt/include")
        set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/gqten/external/hptt/libhptt.a")
    else ()
        find_path(hptt_INCLUDE_DIR "hptt.h")
        find_library(hptt_LIBRARY "libhptt.a")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_GQTEN)
        add_subdirectory(external)
        set(GQPEPS_TENSOR_LIB_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/gqten/include")
    else ()
        find_path(GQPEPS_TENSOR_LIB_HEADER_PATH "gqten")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_GQMPS2)
        add_subdirectory(external)
        set(GQPEPS_MPS2_LIB_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/gqten/include")
    else ()
        find_path(GQPEPS_MPS_LIB_HEADER_PATH "gqmps2")
    endif ()


    enable_testing()
    find_path(PROFILER_INCLUDE_DIR "gperftools")
    find_library(LIBPROFILER_LIBRARY "libprofiler.a")
    add_subdirectory(profiler profiler)
endif ()
