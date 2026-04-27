cmake_minimum_required(VERSION 3.27)

project(PEPSPackageConsumer LANGUAGES CXX)

find_package(PEPS CONFIG REQUIRED)

foreach(_qlpeps_required_target IN ITEMS MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK)
    if (NOT TARGET ${_qlpeps_required_target})
        message(FATAL_ERROR "Missing transitive dependency target: ${_qlpeps_required_target}")
    endif ()
endforeach ()

if (QLPEPS_VERIFY_EXPECT_OPENMP AND NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "Missing transitive dependency target: OpenMP::OpenMP_CXX")
endif ()

if (QLPEPS_VERIFY_EXPECT_SCALAPACK AND NOT TARGET ScaLAPACK::ScaLAPACK)
    message(FATAL_ERROR "Missing PEPS ScaLAPACK dependency target: ScaLAPACK::ScaLAPACK")
endif ()

add_executable(qlpeps_package_smoke smoke.cc)
target_compile_definitions(qlpeps_package_smoke
        PRIVATE
        $<$<BOOL:${QLPEPS_VERIFY_EXPECT_SCALAPACK}>:QLPEPS_VERIFY_EXPECT_SCALAPACK>)
target_link_libraries(qlpeps_package_smoke PRIVATE PEPS::PEPS)

enable_testing()
add_test(
        NAME qlpeps_package_smoke
        COMMAND "$<TARGET_FILE:qlpeps_package_smoke>")
