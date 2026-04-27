#  SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
# Creation Date: 2023-07-19
#
#  Description: QuantumLiquids/PEPS project. Main CMake file.
#
cmake_minimum_required(VERSION 3.27)

project("QuantumLiquids/PEPS"
        VERSION 0.1.0
        DESCRIPTION "A finite-size PEPS algorithms library"
        LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Build type control.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Build output control.
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

option(QLPEPS_BUILD_UNITTEST "Build unittests for QuantumLiquids/PEPS." OFF)
option(QLPEPS_BUILD_EXAMPLES "Build examples for QuantumLiquids/PEPS." OFF)

# Advanced options.
option(QLPEPS_TIMING_MODE "Timing mode." OFF)
option(QLPEPS_USE_SCALAPACK "Enable ScaLAPACK for distributed MinSR eigensolve" OFF)

set(CMAKE_CXX_STANDARD 20) # keyword "requires"
add_definitions(-g)

set(QLPEPS_MIN_QLMPS_VERSION "0.2.0")
set(QLPEPS_PACKAGE_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/PEPS")

find_package(UltraDMRG ${QLPEPS_MIN_QLMPS_VERSION} CONFIG REQUIRED)

# Build(Install) library.
set(QLPEPS_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(QLPEPS_HEADER_DIR "${QLPEPS_HEADER_PATH}/qlpeps")

add_library(PEPS INTERFACE)
add_library(PEPS::PEPS ALIAS PEPS)
target_compile_features(PEPS INTERFACE cxx_std_20)
target_compile_definitions(PEPS
        INTERFACE
        $<$<BOOL:${QLPEPS_TIMING_MODE}>:QLPEPS_TIMING_MODE>)
target_include_directories(PEPS
        INTERFACE
        $<BUILD_INTERFACE:${QLPEPS_HEADER_PATH}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(PEPS
        INTERFACE
        UltraDMRG::UltraDMRG)

set(QLPEPS_SCALAPACK_BACKEND "")
set(QLPEPS_SCALAPACK_BLA_VENDOR "")
set(QLPEPS_SCALAPACK_LIBRARY "")
set(QLPEPS_BLACS_LIBRARY "")

if (QLPEPS_USE_SCALAPACK)
    block(SCOPE_FOR VARIABLES
            PROPAGATE
            QLPEPS_SCALAPACK_BACKEND
            QLPEPS_SCALAPACK_BLA_VENDOR
            QLPEPS_SCALAPACK_LIBRARY
            QLPEPS_BLACS_LIBRARY)
        list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
        if (NOT TARGET ScaLAPACK::ScaLAPACK)
            find_package(ScaLAPACK REQUIRED)
        endif ()

        if (DEFINED _scalapack_backend)
            set(QLPEPS_SCALAPACK_BACKEND "${_scalapack_backend}")
        elseif (HP_NUMERIC_USE_MKL)
            set(QLPEPS_SCALAPACK_BACKEND "mkl")
        elseif (HP_NUMERIC_USE_AOCL)
            set(QLPEPS_SCALAPACK_BACKEND "aocl")
        elseif (HP_NUMERIC_USE_OPENBLAS)
            set(QLPEPS_SCALAPACK_BACKEND "openblas")
        endif ()

        if (DEFINED BLA_VENDOR)
            set(QLPEPS_SCALAPACK_BLA_VENDOR "${BLA_VENDOR}")
        endif ()
        if (DEFINED SCALAPACK_LIBRARY)
            set(QLPEPS_SCALAPACK_LIBRARY "${SCALAPACK_LIBRARY}")
        endif ()
        if (DEFINED BLACS_LIBRARY)
            set(QLPEPS_BLACS_LIBRARY "${BLACS_LIBRARY}")
        endif ()

        if (QLPEPS_SCALAPACK_BACKEND STREQUAL "mkl" AND NOT QLPEPS_BLACS_LIBRARY)
            message(FATAL_ERROR "QLPEPS_USE_SCALAPACK=ON with MKL requires BLACS_LIBRARY.")
        endif ()
    endblock ()

    target_compile_definitions(PEPS INTERFACE QLPEPS_HAS_SCALAPACK)
    target_link_libraries(PEPS INTERFACE ScaLAPACK::ScaLAPACK)
endif ()

install(DIRECTORY ${QLPEPS_HEADER_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
        cmake/PEPSConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSConfig.cmake"
        INSTALL_DESTINATION "${QLPEPS_PACKAGE_CONFIG_DIR}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

configure_file(
        cmake/PEPSDependencies.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSDependencies.cmake"
        @ONLY)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)

install(TARGETS PEPS
        EXPORT PEPSTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT PEPSTargets
        FILE PEPSTargets.cmake
        NAMESPACE PEPS::
        DESTINATION "${QLPEPS_PACKAGE_CONFIG_DIR}")

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/PEPSDependencies.cmake"
        DESTINATION "${QLPEPS_PACKAGE_CONFIG_DIR}")

install(FILES cmake/Modules/FindScaLAPACK.cmake
        DESTINATION "${QLPEPS_PACKAGE_CONFIG_DIR}/modules")

set(QLPEPS_VERIFY_EXPECT_OPENMP OFF)
if (TARGET OpenMP::OpenMP_CXX)
    set(QLPEPS_VERIFY_EXPECT_OPENMP ON)
endif ()

set(QLPEPS_VERIFY_EXPECT_SCALAPACK OFF)
if (QLPEPS_USE_SCALAPACK)
    set(QLPEPS_VERIFY_EXPECT_SCALAPACK ON)
endif ()

add_custom_target(verify-package
        COMMAND ${CMAKE_COMMAND}
        -DQLPEPS_VERIFY_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DQLPEPS_VERIFY_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DQLPEPS_VERIFY_UltraDMRG_DIR:PATH=${UltraDMRG_DIR}
        -DQLPEPS_VERIFY_TensorToolkit_DIR:PATH=${TensorToolkit_DIR}
        -DQLPEPS_VERIFY_INSTALL_LIBDIR:STRING=${CMAKE_INSTALL_LIBDIR}
        -DQLPEPS_VERIFY_USE_SCALAPACK:BOOL=${QLPEPS_USE_SCALAPACK}
        -DQLPEPS_VERIFY_TIMING_MODE:BOOL=${QLPEPS_TIMING_MODE}
        -DQLPEPS_VERIFY_EXPECT_OPENMP:BOOL=${QLPEPS_VERIFY_EXPECT_OPENMP}
        -DQLPEPS_VERIFY_EXPECT_SCALAPACK:BOOL=${QLPEPS_VERIFY_EXPECT_SCALAPACK}
        -P ${PROJECT_SOURCE_DIR}/tests/package_consumer/verify_install.cmake
        USES_TERMINAL)

# Doxygen documentation (optional)
option(QLPEPS_BUILD_DOCS "Build Doxygen documentation." OFF)
if (QLPEPS_BUILD_DOCS)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        add_subdirectory(docs)
    else ()
        message(WARNING "Doxygen not found. Skipping documentation build.")
    endif ()
endif ()

# Build tool scripts
#add_subdirectory(script)

# Build unittests.
if (QLPEPS_BUILD_UNITTEST)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests tests)
endif ()

# Build examples.
if (QLPEPS_BUILD_EXAMPLES)
    add_subdirectory(examples examples)
endif ()
