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

  void Random(const std::vector<size_t> &occupancy_num) {
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

    std::srand(std::time(nullptr));
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);

    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; row < cols; col++) {
        (*this)({row, col}) = data[row * cols + col];
      }
    }
  }

 private:

};

}//gqpeps
#endif //GQPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
