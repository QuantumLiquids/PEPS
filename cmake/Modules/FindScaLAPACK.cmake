#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2026-03-01
#
#  Description: QuantumLiquids/PEPS project. CMake module to find ScaLAPACK library.
#
#  Detection strategy by BLAS backend:
#    - MKL (HP_NUMERIC_USE_MKL):      mkl_scalapack_lp64 + mkl_blacs_intelmpi_lp64 in $ENV{MKLROOT}/lib
#    - AOCL (HP_NUMERIC_USE_AOCL):    scalapack in $ENV{AOCL_ROOT}/lib
#    - OpenBLAS (HP_NUMERIC_USE_OPENBLAS): libscalapack in Homebrew + system paths
#
#  Creates imported target: ScaLAPACK::ScaLAPACK

include(CheckFunctionExists)

# ===========================================================================
# Determine backend: use HP_NUMERIC_USE_* if set, otherwise auto-detect
# from environment variables (MKLROOT, AOCL_ROOT).
# ===========================================================================

set(_scalapack_backend "generic")

if (HP_NUMERIC_USE_MKL)
    set(_scalapack_backend "mkl")
elseif (HP_NUMERIC_USE_AOCL)
    set(_scalapack_backend "aocl")
elseif (HP_NUMERIC_USE_OPENBLAS)
    set(_scalapack_backend "openblas")
elseif (NOT "$ENV{MKLROOT}" STREQUAL "")
    # Auto-detect: MKLROOT is set → MKL backend
    set(_scalapack_backend "mkl")
elseif (NOT "$ENV{AOCL_ROOT}" STREQUAL "")
    # Auto-detect: AOCL_ROOT is set → AOCL backend
    set(_scalapack_backend "aocl")
endif ()

message(STATUS "FindScaLAPACK: detected backend = ${_scalapack_backend}")

# ===========================================================================
# Backend-specific library search
# ===========================================================================

if (_scalapack_backend STREQUAL "mkl")
    # Intel MKL ScaLAPACK
    if (NOT DEFINED ENV{MKLROOT} OR "$ENV{MKLROOT}" STREQUAL "")
        message(WARNING "FindScaLAPACK: MKL backend selected but MKLROOT is not defined.")
    else ()
        # ScaLAPACK core — NO_DEFAULT_PATH prevents picking up an incompatible
        # system ScaLAPACK when MKL is the intended backend.
        find_library(SCALAPACK_LIBRARY
                NAMES mkl_scalapack_lp64
                PATHS "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64"
                NO_DEFAULT_PATH
                DOC "Path to MKL ScaLAPACK library"
        )
        # BLACS communication layer (try Intel MPI first, then OpenMPI)
        find_library(BLACS_LIBRARY
                NAMES mkl_blacs_intelmpi_lp64 mkl_blacs_openmpi_lp64
                PATHS "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64"
                NO_DEFAULT_PATH
                DOC "Path to MKL BLACS library"
        )
    endif ()

elseif (_scalapack_backend STREQUAL "aocl")
    # AMD AOCL ScaLAPACK
    if (NOT DEFINED ENV{AOCL_ROOT} OR "$ENV{AOCL_ROOT}" STREQUAL "")
        message(WARNING "FindScaLAPACK: AOCL backend selected but AOCL_ROOT is not defined.")
    else ()
        find_library(SCALAPACK_LIBRARY
                NAMES scalapack
                PATHS "$ENV{AOCL_ROOT}/lib"
                NO_DEFAULT_PATH
                DOC "Path to AOCL ScaLAPACK library"
        )
    endif ()

elseif (_scalapack_backend STREQUAL "openblas")
    # OpenBLAS / generic ScaLAPACK
    find_library(SCALAPACK_LIBRARY
            NAMES scalapack scalapack-openmpi
            PATHS /opt/homebrew/lib
                  /usr/local/lib
                  /usr/lib
                  /usr/lib/x86_64-linux-gnu
            DOC "Path to ScaLAPACK library"
    )

else ()
    # Fallback: try common names in default paths
    find_library(SCALAPACK_LIBRARY
            NAMES scalapack mkl_scalapack_lp64
            DOC "Path to ScaLAPACK library"
    )
endif ()

# ===========================================================================
# Best-effort verification: check for pdpotrf_ symbol.
# This requires BLAS/LAPACK + MPI to link successfully.  When these are not
# in CMAKE_REQUIRED_LIBRARIES the check will report "not found" — that is
# expected and non-fatal.  SCALAPACK_WORKS is informational only and is NOT
# included in find_package_handle_standard_args REQUIRED_VARS.
# ===========================================================================

set(_scalapack_libs "")
if (SCALAPACK_LIBRARY)
    list(APPEND _scalapack_libs ${SCALAPACK_LIBRARY})
endif ()
if (BLACS_LIBRARY)
    list(APPEND _scalapack_libs ${BLACS_LIBRARY})
endif ()

if (_scalapack_libs)
    set(CMAKE_REQUIRED_LIBRARIES ${_scalapack_libs})
    # Append MPI/BLAS/LAPACK libraries IF they have already been found by the
    # parent project.  We intentionally do NOT call find_package() here because
    # this module runs early (before MathBackend.cmake sets BLA_VENDOR), and an
    # unguarded find_package(BLAS) would pick up Apple Accelerate — poisoning
    # the BLAS::BLAS imported target for all downstream consumers.
    #
    # Use plain library paths, not imported targets: check_function_exists →
    # try_compile creates a scratch project that cannot resolve imported targets.
    if (MPI_C_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${MPI_C_LIBRARIES})
    endif ()
    if (LAPACK_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    endif ()
    if (BLAS_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    endif ()
    check_function_exists(pdpotrf_ SCALAPACK_WORKS)
    unset(CMAKE_REQUIRED_LIBRARIES)
endif ()

# ===========================================================================
# Standard find_package handling
# ===========================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ScaLAPACK
        REQUIRED_VARS SCALAPACK_LIBRARY
)

# ===========================================================================
# Create imported target
# ===========================================================================

if (SCALAPACK_FOUND)
    set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
    if (BLACS_LIBRARY)
        list(APPEND SCALAPACK_LIBRARIES ${BLACS_LIBRARY})
    endif ()

    add_library(ScaLAPACK::ScaLAPACK INTERFACE IMPORTED)
    set_target_properties(ScaLAPACK::ScaLAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${SCALAPACK_LIBRARIES}"
    )

    message(STATUS "Found ScaLAPACK: ${SCALAPACK_LIBRARIES}")
    if (SCALAPACK_WORKS)
        message(STATUS "ScaLAPACK function check (pdpotrf_): passed")
    else ()
        message(STATUS "ScaLAPACK function check (pdpotrf_): skipped or failed (may still work at link time)")
    endif ()
else ()
    message(WARNING "ScaLAPACK not found. Set SCALAPACK_LIBRARY manually or ensure your BLAS backend provides it.")
endif ()

mark_as_advanced(SCALAPACK_LIBRARY BLACS_LIBRARY)
