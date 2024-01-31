#  SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
#  Description: QuantumLiquids/PEPS project. Main CMake file.
#
cmake_minimum_required(VERSION 3.12)


project("QuantumLiquids/PEPS"
        DESCRIPTION "A finite-size PEPS algorithms library"
        LANGUAGES CXX)


# Build type control.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Build output control.
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

option(QLPEPS_BUILD_UNITTEST "Build unittests for QuantumLiquids/PEPS." OFF)
option(QLPEPS_BUILD_PROFILER "Build profiler test cases for QuantumLiquids/PEPS." OFF)

# Advanced options.
option(QLPEPS_TIMING_MODE "Timing mode." OFF)
if (QLPEPS_TIMING_MODE)
    add_definitions(-DQLPEPS_TIMING_MODE)
endif ()

option(TEST_BUILD_USE_EXTERNAL_HPTT_LIB "Use external hptt library when building unittests for QuantumLiquids/PEPS." ON)
option(TEST_BUILD_USE_EXTERNAL_QLTEN "Use external qlten library when building unittests for QuantumLiquids/PEPS." ON)
option(TEST_BUILD_USE_EXTERNAL_QLMPS "Use external qlmps library when building unittests for QuantumLiquids/PEPS." ON)


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 17)
add_definitions(-g)


# Installation path
set(QLPEPS_TOOL_SCRIPT_DEST "${CMAKE_INSTALL_PREFIX}/bin")
set(QLPEPS_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")


# Build(Install) library.
set(QLPEPS_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
message("${QLPEPS_HEADER_PATH}")
set(QLPEPS_HEADER_DIR "${QLPEPS_HEADER_PATH}/qlpeps")
install(
        DIRECTORY ${QLPEPS_HEADER_DIR}
        DESTINATION ${QLPEPS_INCLUDE_DEST})


# Build tool scripts
#add_subdirectory(script)


# Build unittests.
if (QLPEPS_BUILD_UNITTEST)
    if (NOT TEST_BUILD_USE_EXTERNAL_HPTT_LIB)
        set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/qlten/external/hptt/include")
        set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/qlten/external/hptt/libhptt.a")
    else ()
        find_path(hptt_INCLUDE_DIR "hptt.h")
        find_library(hptt_LIBRARY "libhptt.a")
    endif ()


    if (NOT TEST_BUILD_USE_EXTERNAL_QLTEN)
        add_subdirectory(external)
        set(QLTENSOR_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/qlten/include")
    else ()
        find_path(QLTENSOR_HEADER_PATH "qlten")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_QLMPS)
        add_subdirectory(external)
        set(QLMPS_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/qlten/include")
    else ()
        find_path(QLMPS_HEADER_PATH "qlmps")
    endif ()


    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests tests)
endif ()

if (QLPEPS_BUILD_PROFILER)
    if (NOT TEST_BUILD_USE_EXTERNAL_HPTT_LIB)
        set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/qlten/external/hptt/include")
        set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/qlten/external/hptt/libhptt.a")
    else ()
        find_path(hptt_INCLUDE_DIR "hptt.h")
        find_library(hptt_LIBRARY "libhptt.a")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_QLTEN)
        add_subdirectory(external)
        set(QLPEPS_TENSOR_LIB_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/qlten/include")
    else ()
        find_path(QLPEPS_TENSOR_LIB_HEADER_PATH "qlten")
    endif ()

    if (NOT TEST_BUILD_USE_EXTERNAL_QLMPS)
        add_subdirectory(external)
        set(QLPEPS_MPS_LIB_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/qlten/include")
    else ()
        find_path(QLPEPS_MPS_LIB_HEADER_PATH "qlmps")
    endif ()


    enable_testing()
    find_path(PROFILER_INCLUDE_DIR "gperftools")
    find_library(LIBPROFILER_LIBRARY "libprofiler.a")
    add_subdirectory(profiler profiler)
endif ()
