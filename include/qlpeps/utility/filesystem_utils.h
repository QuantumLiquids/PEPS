/**
 * @file filesystem_utils.h
 * @brief Modern filesystem utilities using C++17 std::filesystem
 * 
 * This file provides filesystem operations with consistent style across the codebase.
 * Designed to replace qlmps::IsPathExist and qlmps::CreatPath functions.
 */

#ifndef QLPEPS_UTILITY_FILESYSTEM_UTILS_H
#define QLPEPS_UTILITY_FILESYSTEM_UTILS_H

#include <string>
#include <filesystem>
#include <iostream>

namespace qlpeps {

/**
 * @brief Check if a path exists
 * @param path The path to check
 * @return true if path exists, false otherwise
 */
inline bool IsPathExist(const std::string& path) {
  if (path.empty()) return false;
  
  try {
    return std::filesystem::exists(path);
  } catch (const std::exception& e) {
    std::cerr << "Error checking path existence " << path << ": " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Create directory recursively  
 * @param path The directory path to create
 * @return true if successful or directory already exists, false on error
 */
inline bool CreatePath(const std::string& path) {
  if (path.empty()) return false;
  
  try {
    if (std::filesystem::exists(path)) return true;
    
    std::cout << "Creating directory: " << path << std::endl;
    return std::filesystem::create_directories(path);
  } catch (const std::exception& e) {
    std::cerr << "Error creating directory " << path << ": " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Ensure directory exists, create if not
 * @param path The directory path
 * @return true if directory exists or was created successfully
 * 
 * This function combines IsPathExist and CreatePath into one operation,
 * eliminating the common pattern: if(!IsPathExist(path)) CreatePath(path);
 */
inline bool EnsureDirectoryExists(const std::string& path) {
  return IsPathExist(path) || CreatePath(path);
}

} // namespace qlpeps

#endif // QLPEPS_UTILITY_FILESYSTEM_UTILS_H
