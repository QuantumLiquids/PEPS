#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2026-03-01
#
#  Description: QuantumLiquids/PEPS project. CMake module to find ScaLAPACK.
#
#  Creates imported target: ScaLAPACK::ScaLAPACK

include(CheckCXXSourceCompiles)
include(FindPackageHandleStandardArgs)

if (TARGET ScaLAPACK::ScaLAPACK)
    set(ScaLAPACK_FOUND TRUE)
    set(SCALAPACK_FOUND TRUE)
    if (NOT DEFINED SCALAPACK_LIBRARIES)
        get_target_property(_scalapack_existing_links
                ScaLAPACK::ScaLAPACK INTERFACE_LINK_LIBRARIES)
        if (_scalapack_existing_links)
            set(SCALAPACK_LIBRARIES "${_scalapack_existing_links}")
        endif ()
    endif ()
    return()
endif ()

function(_qlpeps_scalapack_has_value result variable_name)
    if (DEFINED ${variable_name})
        set(_qlpeps_scalapack_value "${${variable_name}}")
        if (NOT _qlpeps_scalapack_value STREQUAL ""
                AND NOT _qlpeps_scalapack_value MATCHES "-NOTFOUND$")
            set(${result} TRUE PARENT_SCOPE)
            return()
        endif ()
    endif ()

    set(${result} FALSE PARENT_SCOPE)
endfunction()

function(_qlpeps_scalapack_use_first_hint output_variable)
    foreach (_qlpeps_scalapack_hint_variable IN LISTS ARGN)
        _qlpeps_scalapack_has_value(
                _qlpeps_scalapack_hint_has_value
                ${_qlpeps_scalapack_hint_variable})
        if (_qlpeps_scalapack_hint_has_value)
            set(${output_variable}
                    "${${_qlpeps_scalapack_hint_variable}}"
                    PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
endfunction()

set(_scalapack_backend "generic")
if (DEFINED QLPEPS_SCALAPACK_BACKEND
        AND NOT "${QLPEPS_SCALAPACK_BACKEND}" STREQUAL "")
    set(_scalapack_backend "${QLPEPS_SCALAPACK_BACKEND}")
elseif (HP_NUMERIC_USE_MKL)
    set(_scalapack_backend "mkl")
elseif (HP_NUMERIC_USE_AOCL)
    set(_scalapack_backend "aocl")
elseif (HP_NUMERIC_USE_OPENBLAS)
    set(_scalapack_backend "openblas")
elseif (NOT "$ENV{MKLROOT}" STREQUAL "")
    set(_scalapack_backend "mkl")
elseif (NOT "$ENV{AOCL_ROOT}" STREQUAL "")
    set(_scalapack_backend "aocl")
endif ()

string(TOLOWER "${_scalapack_backend}" _scalapack_backend)
if (_scalapack_backend MATCHES "mkl|intel")
    set(_scalapack_backend "mkl")
elseif (_scalapack_backend MATCHES "aocl|amd")
    set(_scalapack_backend "aocl")
elseif (_scalapack_backend MATCHES "openblas")
    set(_scalapack_backend "openblas")
else ()
    set(_scalapack_backend "generic")
endif ()

message(STATUS "FindScaLAPACK: backend = ${_scalapack_backend}")

_qlpeps_scalapack_use_first_hint(
        SCALAPACK_LIBRARY
        SCALAPACK_LIBRARY
        QLPEPS_RECORDED_SCALAPACK_LIBRARY
        QLPEPS_SCALAPACK_LIBRARY)
_qlpeps_scalapack_has_value(_scalapack_library_is_hinted SCALAPACK_LIBRARY)

_qlpeps_scalapack_use_first_hint(
        BLACS_LIBRARY
        BLACS_LIBRARY
        QLPEPS_RECORDED_BLACS_LIBRARY
        QLPEPS_BLACS_LIBRARY)
_qlpeps_scalapack_has_value(_blacs_library_is_hinted BLACS_LIBRARY)

set(_scalapack_mkl_paths "")
if (NOT "$ENV{MKLROOT}" STREQUAL "")
    list(APPEND _scalapack_mkl_paths
            "$ENV{MKLROOT}/lib"
            "$ENV{MKLROOT}/lib/intel64")
endif ()

if (NOT _scalapack_library_is_hinted)
    if (_scalapack_backend STREQUAL "mkl")
        find_library(SCALAPACK_LIBRARY
                NAMES mkl_scalapack_lp64
                PATHS ${_scalapack_mkl_paths}
                NO_DEFAULT_PATH
                DOC "Path to MKL ScaLAPACK library"
        )
    elseif (_scalapack_backend STREQUAL "aocl")
        find_library(SCALAPACK_LIBRARY
                NAMES scalapack
                PATHS "$ENV{AOCL_ROOT}/lib"
                NO_DEFAULT_PATH
                DOC "Path to AOCL ScaLAPACK library"
        )
    elseif (_scalapack_backend STREQUAL "openblas")
        find_library(SCALAPACK_LIBRARY
                NAMES scalapack scalapack-openmpi
                PATHS /opt/homebrew/lib
                      /usr/local/lib
                      /usr/lib
                      /usr/lib/x86_64-linux-gnu
                DOC "Path to ScaLAPACK library"
        )
    else ()
        find_library(SCALAPACK_LIBRARY
                NAMES scalapack scalapack-openmpi mkl_scalapack_lp64
                DOC "Path to ScaLAPACK library"
        )
    endif ()
endif ()

set(_scalapack_uses_mkl_library FALSE)
if (SCALAPACK_LIBRARY)
    foreach (_scalapack_library_item IN LISTS SCALAPACK_LIBRARY)
        if ("${_scalapack_library_item}" MATCHES "mkl_scalapack")
            set(_scalapack_uses_mkl_library TRUE)
        endif ()
    endforeach ()
endif ()

set(_scalapack_requires_blacs FALSE)
if (_scalapack_backend STREQUAL "mkl" OR _scalapack_uses_mkl_library)
    set(_scalapack_requires_blacs TRUE)
endif ()

set(_scalapack_blacs_search_paths ${_scalapack_mkl_paths})
if (SCALAPACK_LIBRARY)
    foreach (_scalapack_library_item IN LISTS SCALAPACK_LIBRARY)
        if (IS_ABSOLUTE "${_scalapack_library_item}")
            get_filename_component(_scalapack_library_dir
                    "${_scalapack_library_item}" DIRECTORY)
            list(APPEND _scalapack_blacs_search_paths
                    "${_scalapack_library_dir}")
        endif ()
    endforeach ()
endif ()
if (_scalapack_blacs_search_paths)
    list(REMOVE_DUPLICATES _scalapack_blacs_search_paths)
endif ()

if (_scalapack_requires_blacs AND NOT _blacs_library_is_hinted)
    set(_scalapack_mkl_blacs_names
            mkl_blacs_intelmpi_lp64
            mkl_blacs_openmpi_lp64)
    set(_scalapack_mpi_flavor "")
    foreach (_scalapack_mpi_hint IN ITEMS
            MPI_CXX_LIBRARY_VERSION_STRING
            MPI_CXX_COMPILER
            MPI_CXX_LIBRARIES)
        if (DEFINED ${_scalapack_mpi_hint})
            string(APPEND _scalapack_mpi_flavor
                    " ${${_scalapack_mpi_hint}}")
        endif ()
    endforeach ()
    string(TOLOWER "${_scalapack_mpi_flavor}"
            _scalapack_mpi_flavor)
    if (_scalapack_mpi_flavor MATCHES "open[ -]?mpi|openmpi")
        set(_scalapack_mkl_blacs_names
                mkl_blacs_openmpi_lp64
                mkl_blacs_intelmpi_lp64)
    endif ()

    find_library(BLACS_LIBRARY
            NAMES ${_scalapack_mkl_blacs_names}
            PATHS ${_scalapack_blacs_search_paths}
            NO_DEFAULT_PATH
            DOC "Path to BLACS library"
    )
endif ()

set(_scalapack_link_libraries "")
set(SCALAPACK_WORKS FALSE)
if (SCALAPACK_LIBRARY)
    list(APPEND _scalapack_link_libraries ${SCALAPACK_LIBRARY})
endif ()
if (BLACS_LIBRARY)
    list(APPEND _scalapack_link_libraries ${BLACS_LIBRARY})
endif ()
foreach (_scalapack_dependency_target IN ITEMS
        MPI::MPI_CXX
        LAPACK::LAPACK
        BLAS::BLAS)
    if (TARGET ${_scalapack_dependency_target})
        list(APPEND _scalapack_link_libraries
                ${_scalapack_dependency_target})
    endif ()
endforeach ()

get_property(_scalapack_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
list(FIND _scalapack_enabled_languages CXX _scalapack_cxx_language_index)
set(_scalapack_can_verify_link_closure FALSE)
if (SCALAPACK_LIBRARY
        AND _scalapack_link_libraries
        AND NOT _scalapack_cxx_language_index EQUAL -1)
    set(_scalapack_can_verify_link_closure TRUE)
endif ()

if (_scalapack_can_verify_link_closure)
    set(_scalapack_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
    set(_scalapack_saved_required_quiet "${CMAKE_REQUIRED_QUIET}")
    set(CMAKE_REQUIRED_LIBRARIES "${_scalapack_link_libraries}")
    set(CMAKE_REQUIRED_QUIET "${ScaLAPACK_FIND_QUIETLY}")

    unset(ScaLAPACK_SCALAPACK_SYMBOL_LINKS CACHE)
    unset(ScaLAPACK_BLACS_SYMBOL_LINKS CACHE)
    check_cxx_source_compiles([=[
extern "C" void pdsyev_();
int main() {
    pdsyev_();
    return 0;
}
]=] ScaLAPACK_SCALAPACK_SYMBOL_LINKS)
    check_cxx_source_compiles([=[
extern "C" void Cblacs_pinfo(int*, int*);
int main() {
    int mypnum = 0;
    int nprocs = 0;
    Cblacs_pinfo(&mypnum, &nprocs);
    return 0;
}
]=] ScaLAPACK_BLACS_SYMBOL_LINKS)

    set(CMAKE_REQUIRED_LIBRARIES "${_scalapack_saved_required_libraries}")
    set(CMAKE_REQUIRED_QUIET "${_scalapack_saved_required_quiet}")
endif ()

set(_scalapack_required_vars SCALAPACK_LIBRARY)
if (_scalapack_requires_blacs)
    list(APPEND _scalapack_required_vars BLACS_LIBRARY)
endif ()
if (_scalapack_can_verify_link_closure)
    list(APPEND _scalapack_required_vars
            ScaLAPACK_SCALAPACK_SYMBOL_LINKS
            ScaLAPACK_BLACS_SYMBOL_LINKS)
endif ()

find_package_handle_standard_args(ScaLAPACK
        REQUIRED_VARS ${_scalapack_required_vars}
        REASON_FAILURE_MESSAGE
        "Set SCALAPACK_LIBRARY and, when BLACS is separate, BLACS_LIBRARY to a link-complete ScaLAPACK stack."
)
set(SCALAPACK_FOUND "${ScaLAPACK_FOUND}")

if (ScaLAPACK_FOUND)
    set(SCALAPACK_LIBRARIES ${_scalapack_link_libraries})
    if (_scalapack_can_verify_link_closure
            AND ScaLAPACK_SCALAPACK_SYMBOL_LINKS
            AND ScaLAPACK_BLACS_SYMBOL_LINKS)
        set(SCALAPACK_WORKS TRUE)
    endif ()

    if (NOT TARGET ScaLAPACK::ScaLAPACK)
        add_library(ScaLAPACK::ScaLAPACK INTERFACE IMPORTED)
        set_target_properties(ScaLAPACK::ScaLAPACK PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_scalapack_link_libraries}"
        )
    endif ()
endif ()

mark_as_advanced(SCALAPACK_LIBRARY BLACS_LIBRARY)
