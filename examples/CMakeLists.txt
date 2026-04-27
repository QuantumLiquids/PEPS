# CMakeLists.txt for VMC PEPS Migration Tutorial Examples

cmake_minimum_required(VERSION 3.16)
project(vmc_peps_examples)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the migration example executable
# add_executable(migration_example migration_example.cpp)
# target_link_libraries(migration_example PRIVATE PEPS::PEPS)

# Add transverse field Ising Simple Update example
add_executable(transverse_field_ising_simple_update transverse_field_ising_simple_update.cpp)
target_link_libraries(transverse_field_ising_simple_update
    PRIVATE
    PEPS::PEPS
)

# Add transverse field Ising VMC optimize example
add_executable(transverse_field_ising_vmc_optimize transverse_field_ising_vmc_optimize.cpp)
target_link_libraries(transverse_field_ising_vmc_optimize
    PRIVATE
    PEPS::PEPS
)

# Add transverse field Ising Monte Carlo measurement example
add_executable(transverse_field_ising_mc_measure transverse_field_ising_mc_measure.cpp)
target_link_libraries(transverse_field_ising_mc_measure
    PRIVATE
    PEPS::PEPS
)

# Install targets
# install(TARGETS migration_example transverse_field_ising_simple_update
#     RUNTIME DESTINATION bin)
