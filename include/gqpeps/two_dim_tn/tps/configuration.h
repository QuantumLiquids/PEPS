/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: GraceQ/VMC-PEPS project. Configurations.
*/
#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H

#include <random>
#include "gqpeps/two_dim_tn/framework/duomatrix.h"

namespace gqpeps {

class Configuration : public DuoMatrix<size_t> {
 public:

  using DuoMatrix<size_t>::DuoMatrix;
  using DuoMatrix<size_t>::operator();

  /**
   * Random generate a configuration
   *
   * @param occupancy_num  a vector with length dim, where dim is the dimension of loccal hilbert space
   *                  occupancy_num[i] indicates how many sites occupy the i-th state.
   * @param seed seed for random number generator
   */
  void Random(const std::vector<size_t> &occupancy_num, size_t seed) {
    size_t dim = occupancy_num.size();
    size_t rows = this->rows();
    size_t cols = this->cols();
    std::vector<size_t> data(rows * cols);
    size_t off_set = 0;
    for (size_t i = 0; i < dim; i++) {
      for (size_t j = off_set; j < off_set + occupancy_num[i]; j++) {
        data[j] = i;
      }
      off_set += occupancy_num[i];
    }
    assert(off_set == data.size());

    std::srand(seed);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);

    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)({row, col}) = data[row * cols + col];
      }
    }
  }

  /**
   *
   * @param path
   * @param label e.g. the MPI rank
   */
  void Dump(const std::string &path, const size_t label) {
    if (!gqmps2::IsPathExist(path)) { gqmps2::CreatPath(path); }
    std::string file = path + "/configuration" + std::to_string(label);
    std::ofstream ofs(file, std::ofstream::binary);
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        ofs << (*this)({row, col}) << std::endl;
      }
    }
    ofs << std::endl;
    ofs.close();
  }

  bool Load(const std::string &path, const size_t label) {
    std::string file = path + "/configuration" + std::to_string(label);
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) {
      return false; // Failed to open the file
    }
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        ifs >> (*this)({row, col});
      }
    }
    ifs.close();
  }

 private:

};

}//gqpeps
#endif //GQPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
