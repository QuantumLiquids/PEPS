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
#include "mpi.h"        //MPI BroadCast

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

  size_t Sum(void) const {
    size_t summation = 0;
    size_t rows = this->rows();
    size_t cols = this->cols();
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        summation += (*this)({row, col});
      }
    }
    return summation;
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
    return true;
  }

 private:

};

inline void MPI_Send(
    Configuration &config,
    size_t dest,
    int tag,
    MPI_Comm comm
) {
  const size_t rows = config.rows(), cols = config.cols(), N = config.size();
  size_t *config_raw_data = new size_t[N];
  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      config_raw_data[row * cols + col] = config({row, col});
    }
  }
  ::MPI_Send(config_raw_data, N, MPI_UNSIGNED_LONG_LONG, dest, tag, comm);
  delete[]config_raw_data;
}

///< config must reserve the memory space
inline int MPI_Recv(
    Configuration &config,
    size_t source,
    int tag,
    MPI_Comm comm,
    MPI_Status *status
) {
  const size_t rows = config.rows(), cols = config.cols(), N = config.size();
  size_t *config_raw_data = new size_t[N];
  int err_message = ::MPI_Recv(config_raw_data, N, MPI_UNSIGNED_LONG_LONG, source, tag, comm, status);
  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      config({row, col}) = config_raw_data[row * cols + col];
    }
  }
  delete[]config_raw_data;
  return err_message;
}

inline int MPI_Sendrecv(
    const Configuration &config_send,
    size_t dest, int sendtag,
    Configuration &config_recv,
    size_t source, int recvtag,
    MPI_Comm comm,
    MPI_Status *status
) {
  const size_t rows = config_send.rows(), cols = config_send.cols(), N = config_send.size();
  size_t *config_raw_data_send = new size_t[N];
  size_t *config_raw_data_recv = new size_t[N];

  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      config_raw_data_send[row * cols + col] = config_send({row, col});
    }
  }
  int err_message = ::MPI_Sendrecv(config_raw_data_send, N, MPI_UNSIGNED_LONG_LONG, dest, sendtag,
                                   config_raw_data_recv, N, MPI_UNSIGNED_LONG_LONG, source, recvtag,
                                   comm, status);

  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      config_recv({row, col}) = config_raw_data_recv[row * cols + col];
    }
  }
  delete[]config_raw_data_send;
  delete[]config_raw_data_recv;
  return err_message;
}


inline void MPI_BCast(
    Configuration &config,
    const size_t root,
    MPI_Comm comm
) {
  const size_t rows = config.rows(), cols = config.cols(), N = config.size();
  size_t *config_raw_data = new size_t[N];
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  if (my_rank == root) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        config_raw_data[row * cols + col] = config({row, col});
      }
    }
  }

  ::MPI_Bcast(config_raw_data, N, MPI_UNSIGNED_LONG_LONG, root, comm);

  if (my_rank != root) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        config({row, col}) = config_raw_data[row * cols + col];
      }
    }
  }
  delete[]config_raw_data;
}

}//gqpeps
#endif //GQPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
