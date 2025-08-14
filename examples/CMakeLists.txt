# CMakeLists.txt for VMC PEPS Migration Tutorial Examples

cmake_minimum_required(VERSION 3.16)
project(vmc_peps_migration_tutorial)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Add the migration example executable
add_executable(migration_example migration_example.cpp)

# Link libraries
target_link_libraries(migration_example
    qlpeps
    qlten
    ${MPI_CXX_LIBRARIES}
)

# Set MPI compiler flags
target_compile_options(migration_example PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Add test target
add_executable(test_migration test_migration.cpp)
target_link_libraries(test_migration
    qlpeps
    qlten
    ${MPI_CXX_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)
target_compile_options(test_migration PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Install targets
install(TARGETS migration_example test_migration
    RUNTIME DESTINATION bin
)
