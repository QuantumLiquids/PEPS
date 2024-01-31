/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-14
*
* Description: QuantumLiquids/PEPS project. Helper function for test
*/


#ifndef VMC_PEPS_TEST_PARSE_PARAMETERS_H
#define VMC_PEPS_TEST_PARSE_PARAMETERS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>


template<typename T>
std::map<std::string, T> ParseParametersFromFile(const std::string &filename) {
  std::map<std::string, T> parameters;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return parameters;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream lineStream(line);
    std::string key;
    T value;

    if (lineStream >> key >> value) {
      parameters[key] = value;
    }
  }

  file.close();
  return parameters;
}

#endif //VMC_PEPS_TEST_PARSE_PARAMETERS_H
