# CMakeLists.txt for VMC PEPS Migration Tutorial Examples

cmake_minimum_required(VERSION 3.16)
project(vmc_peps_examples)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)

include(../cmake/Modules/MathBackend.cmake)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

find_path(hptt_INCLUDE_DIR "hptt.h")
find_library(hptt_LIBRARY "libhptt.a")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Add the migration example executable
# add_executable(migration_example migration_example.cpp)

# Link libraries
# target_include_directories(migration_example PRIVATE ${BLAS_INCLUDE_DIR} ${hptt_INCLUDE_DIR})
# target_link_libraries(migration_example
#     ${hptt_LIBRARY}
#     BLAS::BLAS
#     LAPACK::LAPACK
#     OpenMP::OpenMP_CXX
#     MPI::MPI_CXX
# )

# Set MPI compiler flags
# target_compile_options(migration_example PRIVATE )

# Add transverse field Ising Simple Update example
add_executable(transverse_field_ising_simple_update transverse_field_ising_simple_update.cpp)
target_include_directories(transverse_field_ising_simple_update PRIVATE ${BLAS_INCLUDE_DIRS} ${hptt_INCLUDE_DIR})
target_link_libraries(transverse_field_ising_simple_update
    ${hptt_LIBRARY}
    BLAS::BLAS
    LAPACK::LAPACK
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)
target_compile_options(transverse_field_ising_simple_update PRIVATE )

# Add transverse field Ising VMC optimize example
add_executable(transverse_field_ising_vmc_optimize transverse_field_ising_vmc_optimize.cpp)
target_include_directories(transverse_field_ising_vmc_optimize PRIVATE ${BLAS_INCLUDE_DIRS} ${hptt_INCLUDE_DIR})
target_link_libraries(transverse_field_ising_vmc_optimize
    ${hptt_LIBRARY}
    BLAS::BLAS
    LAPACK::LAPACK
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)
target_compile_options(transverse_field_ising_vmc_optimize PRIVATE )

# Install targets
# install(TARGETS migration_example transverse_field_ising_simple_update
#     RUNTIME DESTINATION bin)
