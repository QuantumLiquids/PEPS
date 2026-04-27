if (NOT DEFINED QLPEPS_VERIFY_BUILD_TYPE OR QLPEPS_VERIFY_BUILD_TYPE STREQUAL "")
  set(QLPEPS_VERIFY_BUILD_TYPE Debug)
endif ()
if (NOT DEFINED QLPEPS_VERIFY_INSTALL_LIBDIR OR QLPEPS_VERIFY_INSTALL_LIBDIR STREQUAL "")
  set(QLPEPS_VERIFY_INSTALL_LIBDIR lib)
endif ()
if (NOT DEFINED QLPEPS_VERIFY_USE_SCALAPACK)
  set(QLPEPS_VERIFY_USE_SCALAPACK OFF)
endif ()
if (NOT DEFINED QLPEPS_VERIFY_TIMING_MODE)
  set(QLPEPS_VERIFY_TIMING_MODE OFF)
endif ()
if (NOT DEFINED QLPEPS_VERIFY_EXPECT_OPENMP)
  set(QLPEPS_VERIFY_EXPECT_OPENMP OFF)
endif ()
if (NOT DEFINED QLPEPS_VERIFY_EXPECT_SCALAPACK)
  set(QLPEPS_VERIFY_EXPECT_SCALAPACK OFF)
endif ()

string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef _qlpeps_verify_suffix)
set(_qlpeps_verify_root "/tmp/qlpeps-verify-${_qlpeps_verify_suffix}")
set(package_build_dir "${_qlpeps_verify_root}/package-build")
set(package_prefix "${_qlpeps_verify_root}/prefix")
set(relocated_prefix "${_qlpeps_verify_root}/relocated-prefix")
set(consumer_build_dir "${_qlpeps_verify_root}/consumer-build")
get_filename_component(_qlpeps_source_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

message(STATUS "PEPS package verification root: ${_qlpeps_verify_root}")

set(_qlpeps_configure_args
  -S "${_qlpeps_source_root}"
  -B "${package_build_dir}"
  -DCMAKE_BUILD_TYPE=${QLPEPS_VERIFY_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${package_prefix}
  -DCMAKE_INSTALL_LIBDIR=${QLPEPS_VERIFY_INSTALL_LIBDIR}
  -DQLPEPS_BUILD_UNITTEST=OFF
  -DQLPEPS_BUILD_EXAMPLES=OFF
  -DQLPEPS_BUILD_DOCS=OFF
  -DQLPEPS_USE_SCALAPACK=${QLPEPS_VERIFY_USE_SCALAPACK}
  -DQLPEPS_TIMING_MODE=${QLPEPS_VERIFY_TIMING_MODE})

if (DEFINED QLPEPS_VERIFY_CXX_COMPILER AND NOT QLPEPS_VERIFY_CXX_COMPILER STREQUAL "")
  list(APPEND _qlpeps_configure_args -DCMAKE_CXX_COMPILER=${QLPEPS_VERIFY_CXX_COMPILER})
endif ()
if (DEFINED QLPEPS_VERIFY_UltraDMRG_DIR AND NOT QLPEPS_VERIFY_UltraDMRG_DIR STREQUAL "")
  list(APPEND _qlpeps_configure_args -DUltraDMRG_DIR=${QLPEPS_VERIFY_UltraDMRG_DIR})
endif ()
if (DEFINED QLPEPS_VERIFY_TensorToolkit_DIR AND NOT QLPEPS_VERIFY_TensorToolkit_DIR STREQUAL "")
  list(APPEND _qlpeps_configure_args -DTensorToolkit_DIR=${QLPEPS_VERIFY_TensorToolkit_DIR})
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_qlpeps_verify_root}"
  RESULT_VARIABLE cleanup_root_result
)
if (NOT cleanup_root_result EQUAL 0)
  message(FATAL_ERROR "PEPS package verification cleanup failed: ${cleanup_root_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} ${_qlpeps_configure_args}
  RESULT_VARIABLE configure_result
)
if (NOT configure_result EQUAL 0)
  message(FATAL_ERROR "PEPS package configure failed: ${configure_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${package_build_dir}" -j4
  RESULT_VARIABLE build_result
)
if (NOT build_result EQUAL 0)
  message(FATAL_ERROR "PEPS package build failed: ${build_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --install "${package_build_dir}"
  RESULT_VARIABLE install_result
)
if (NOT install_result EQUAL 0)
  message(FATAL_ERROR "PEPS package install failed: ${install_result}")
endif ()

set(_qlpeps_package_dir "${package_prefix}/${QLPEPS_VERIFY_INSTALL_LIBDIR}/cmake/PEPS")
set(_qlpeps_installed_package_files
  "${_qlpeps_package_dir}/PEPSConfig.cmake"
  "${_qlpeps_package_dir}/PEPSConfigVersion.cmake"
  "${_qlpeps_package_dir}/PEPSDependencies.cmake"
  "${_qlpeps_package_dir}/PEPSTargets.cmake")

foreach(_qlpeps_package_file IN LISTS _qlpeps_installed_package_files)
  if (NOT EXISTS "${_qlpeps_package_file}")
    message(FATAL_ERROR "Missing installed PEPS package file: ${_qlpeps_package_file}")
  endif ()

  file(READ "${_qlpeps_package_file}" _qlpeps_package_file_contents)
  foreach(_qlpeps_forbidden_path IN ITEMS "${_qlpeps_source_root}" "${package_build_dir}")
    if (_qlpeps_forbidden_path STREQUAL "")
      continue ()
    endif ()
    string(FIND "${_qlpeps_package_file_contents}" "${_qlpeps_forbidden_path}" _qlpeps_leak_pos)
    if (NOT _qlpeps_leak_pos EQUAL -1)
      message(FATAL_ERROR
        "PEPS package file leaked a source/build-tree path: ${_qlpeps_package_file}")
    endif ()
  endforeach ()
endforeach ()

if (QLPEPS_VERIFY_EXPECT_SCALAPACK
    AND NOT EXISTS "${_qlpeps_package_dir}/modules/FindScaLAPACK.cmake")
  message(FATAL_ERROR "Missing installed PEPS ScaLAPACK module.")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rename "${package_prefix}" "${relocated_prefix}"
  RESULT_VARIABLE relocate_result
)
if (NOT relocate_result EQUAL 0)
  message(FATAL_ERROR "PEPS package relocation failed: ${relocate_result}")
endif ()

set(_qlpeps_consumer_configure_args
  -S "${CMAKE_CURRENT_LIST_DIR}"
  -B "${consumer_build_dir}"
  -DCMAKE_BUILD_TYPE=${QLPEPS_VERIFY_BUILD_TYPE}
  -DCMAKE_PREFIX_PATH=${relocated_prefix}
  -DQLPEPS_VERIFY_EXPECT_OPENMP=${QLPEPS_VERIFY_EXPECT_OPENMP}
  -DQLPEPS_VERIFY_EXPECT_SCALAPACK=${QLPEPS_VERIFY_EXPECT_SCALAPACK})

if (DEFINED QLPEPS_VERIFY_CXX_COMPILER AND NOT QLPEPS_VERIFY_CXX_COMPILER STREQUAL "")
  list(APPEND _qlpeps_consumer_configure_args
    -DCMAKE_CXX_COMPILER=${QLPEPS_VERIFY_CXX_COMPILER})
endif ()
if (DEFINED QLPEPS_VERIFY_UltraDMRG_DIR AND NOT QLPEPS_VERIFY_UltraDMRG_DIR STREQUAL "")
  list(APPEND _qlpeps_consumer_configure_args
    -DUltraDMRG_DIR=${QLPEPS_VERIFY_UltraDMRG_DIR})
endif ()
if (DEFINED QLPEPS_VERIFY_TensorToolkit_DIR AND NOT QLPEPS_VERIFY_TensorToolkit_DIR STREQUAL "")
  list(APPEND _qlpeps_consumer_configure_args
    -DTensorToolkit_DIR=${QLPEPS_VERIFY_TensorToolkit_DIR})
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} ${_qlpeps_consumer_configure_args}
  RESULT_VARIABLE consumer_configure_result
)
if (NOT consumer_configure_result EQUAL 0)
  message(FATAL_ERROR "PEPS consumer configure failed: ${consumer_configure_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${consumer_build_dir}" -j4
  RESULT_VARIABLE consumer_build_result
)
if (NOT consumer_build_result EQUAL 0)
  message(FATAL_ERROR "PEPS consumer build failed: ${consumer_build_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${consumer_build_dir}" --output-on-failure
  RESULT_VARIABLE consumer_test_result
)
if (NOT consumer_test_result EQUAL 0)
  message(FATAL_ERROR "PEPS consumer test failed: ${consumer_test_result}")
endif ()
