# CMakeLists.txt for PEPS Documentation
# This file handles the documentation build process using Doxygen

cmake_minimum_required(VERSION 3.16)
project(PEPS_Documentation)

# Find Doxygen
find_package(Doxygen REQUIRED)

# Set documentation output directory to be within docs/build
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

# Configure the Doxyfile with CMake variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
               ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile @ONLY)

# Create a custom target for building documentation
add_custom_target(docs ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)

# Create a clean target for documentation
add_custom_target(docs-clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/build
    COMMENT "Cleaning documentation build directory"
)

# Install documentation if requested
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build/install)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/html
        DESTINATION share/doc/peps
        OPTIONAL)

# Print configuration information
message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
message(STATUS "Documentation output directory: ${DOXYGEN_OUTPUT_DIRECTORY}")
message(STATUS "Documentation install directory: ${CMAKE_INSTALL_PREFIX}/share/doc/peps")
