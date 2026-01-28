# CTest helper: run an MPI program expecting non-zero exit and specific output.
#
# Usage (from add_test):
#   ${CMAKE_COMMAND}
#     -DMPIEXEC=<path>
#     -DMPIEXEC_NUMPROC_FLAG=<flag>   (e.g. -n)
#     -DNPROCS=<int>
#     -DEXECUTABLE=<path-to-test-exe>
#     -DEXPECT_REGEX_1=<regex>
#     -DEXPECT_REGEX_2=<regex>
#     -P <this-script>
#
# The script returns success (exit 0) iff:
# - MPI program exits with non-zero code
# - output (stdout+stderr) matches EXPECT_REGEX_1 and EXPECT_REGEX_2 (if provided)

if(NOT DEFINED MPIEXEC)
  message(FATAL_ERROR "MPIEXEC is required")
endif()
if(NOT DEFINED MPIEXEC_NUMPROC_FLAG)
  message(FATAL_ERROR "MPIEXEC_NUMPROC_FLAG is required")
endif()
if(NOT DEFINED NPROCS)
  message(FATAL_ERROR "NPROCS is required")
endif()
if(NOT DEFINED EXECUTABLE)
  message(FATAL_ERROR "EXECUTABLE is required")
endif()

execute_process(
  COMMAND "${MPIEXEC}" "${MPIEXEC_NUMPROC_FLAG}" "${NPROCS}" "${EXECUTABLE}"
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout_out
  ERROR_VARIABLE stderr_out
)

set(all_out "${stdout_out}\n${stderr_out}")

if(exit_code EQUAL 0)
  message(FATAL_ERROR "Expected non-zero exit code (MPI_Abort), but got 0.\nOutput:\n${all_out}")
endif()

if(DEFINED EXPECT_REGEX_1)
  if(NOT all_out MATCHES "${EXPECT_REGEX_1}")
    message(FATAL_ERROR "Expected regex not found: ${EXPECT_REGEX_1}\nOutput:\n${all_out}")
  endif()
endif()

if(DEFINED EXPECT_REGEX_2)
  if(NOT all_out MATCHES "${EXPECT_REGEX_2}")
    message(FATAL_ERROR "Expected regex not found: ${EXPECT_REGEX_2}\nOutput:\n${all_out}")
  endif()
endif()

