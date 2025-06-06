/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: QuantumLiquids/PEPS project. Configuration Class.
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H

#include <random>
#include "mpi.h"                // MPI BroadCast
#include "qlten/qlten.h"        // Showable, Streamable
#include "qlmps/qlmps.h"        // IsPathExist, CreatPath
#include "qlpeps/two_dim_tn/framework/duomatrix.h"

namespace qlpeps {
using qlten::Showable;
using qlten::Streamable;

/**
 * Configuration in Models.
 *
 * It can represent the spin/charge configurations in spin/fermion models,
 * and can represent a Monte-Carlo sampling configuration in VMC.
 *
 * The configurations are numbered from 0
 */
class Configuration : public DuoMatrix<size_t>, public Showable, public Streamable {
 public:

  using DuoMatrix<size_t>::DuoMatrix;
  using DuoMatrix<size_t>::operator();
  using DuoMatrix<size_t>::operator==;

  Configuration(std::vector<std::vector<size_t>> config_data) : DuoMatrix<size_t>(config_data.size(),
                                                                                  config_data[0].size()) {
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        (*this)({row, col}) = config_data.at(row).at(col);
      }
    }
  }

  /**
   * Random generate a configuration
   *
   * @param config_map_to_occupancy_num
   *    e.g {{0, 10}, {2,4}, {5,8}}
   *    0, 2, 5 represent some spin configurations, the hilbert space dimension should has at least 6 in this example
   *    10, 4, 8 indicate how many sites are occupied by the corresponding spin configuration.
   */
  void Random(const std::map<size_t, size_t> config_map_to_occupancy_num) {
    size_t rows = this->rows();
    size_t cols = this->cols();
    std::vector<size_t> configuration_list(rows * cols);
    size_t off_set = 0;
    for (auto [config, occupancy_num] : config_map_to_occupancy_num) {
      std::fill(configuration_list.begin() + off_set, configuration_list.begin() + off_set + occupancy_num, config);
      off_set += occupancy_num;
    }
    assert(off_set == configuration_list.size());
    std::random_device rd;
    std::mt19937 rand_num_gen(rd());
    std::shuffle(configuration_list.begin(), configuration_list.end(), rand_num_gen);

    *this = Configuration(rows, cols, configuration_list);
  }

  /**
   * Random generate a configuration
   *
   * @param occupancy_num  a vector with length dim, where dim is the dimension of local hilbert space
   *                  occupancy_num[i] indicates how many sites occupy the i-th state.
   * @param seed seed for random number generator
   */
  void Random(const std::vector<size_t> &occupancy_num) {
    size_t dim = occupancy_num.size();
    size_t rows = this->rows();
    size_t cols = this->cols();
    std::vector<size_t> configuration_list(rows * cols);
    size_t off_set = 0;
    for (size_t i = 0; i < dim; i++) {
      std::fill(configuration_list.begin() + off_set, configuration_list.begin() + off_set + occupancy_num[i], i);
      off_set += occupancy_num[i];
    }
    assert(off_set == configuration_list.size());

    // random_device can generate different random number during the running
    // and do not need to feed the seed;
    // But if the mt19937 is not feed the seed, the rand number it generate in
    // each time is the same.
    std::random_device rd;
    std::mt19937 rand_num_gen(rd());
    std::shuffle(configuration_list.begin(), configuration_list.end(), rand_num_gen);

    *this = Configuration(rows, cols, configuration_list);
  }

  size_t Sum(void) const {
    size_t sum = 0;
    for (auto elem : *this) {
      sum += elem;
    }
    return sum;
  }

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  /**
   *
   * @param path
   * @param label e.g. the MPI rank
   */
  void Dump(const std::string &path, const size_t label) {
    if (!qlmps::IsPathExist(path)) { qlmps::CreatPath(path); }
    std::string file = path + "/configuration" + std::to_string(label);
    std::ofstream ofs(file, std::ofstream::binary);
    ofs << (*this);
    ofs.close();
  }

  bool Load(const std::string &path, const size_t label) {
    std::string file = path + "/configuration" + std::to_string(label);
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) {
      return false; // Failed to open the file
    }
    ifs >> (*this);
    ifs.close();
    return true;
  }

  void Show(const size_t indent_level = 0) const override;
 private:
  Configuration(size_t rows, size_t cols, const std::vector<size_t> &configuration_list) : Configuration(rows, cols) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)({row, col}) = configuration_list.at(row * cols + col);
      }
    }
  }

};

void Configuration::StreamRead(std::istream &is) {
  for (size_t row = 0; row < this->rows(); row++) {
    for (size_t col = 0; col < this->cols(); col++) {
      is >> (*this)({row, col});
    }
  }
}

void Configuration::StreamWrite(std::ostream &os) const {
  for (size_t row = 0; row < this->rows(); row++) {
    for (size_t col = 0; col < this->cols() - 1; col++) {
      os << (*this)({row, col}) << " ";
    }
    os << (*this)({row, this->cols() - 1}) << std::endl;
  }
}

void Configuration::Show(const size_t indent_level) const {
  using qlten::IndentPrinter;
  std::cout << IndentPrinter(indent_level) << "Configurations:" << std::endl;
  for (size_t row = 0; row < this->rows(); row++) {
    std::cout << IndentPrinter(indent_level + 1);
    for (size_t col = 0; col < this->cols(); col++) {
      std::cout << (*this)({row, col}) << " ";
    }
    std::cout << std::endl;
  }
}

inline void MPI_Send(
    Configuration &config,
    size_t dest,
    int tag,
    const MPI_Comm &comm
) {
  const size_t rows = config.rows(), cols = config.cols(), N = config.size();
  auto *config_raw_data = new size_t[N];
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
    const MPI_Comm &comm,
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
    const MPI_Comm &comm,
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
    const MPI_Comm &comm
) {
  using namespace qlten;
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

  HANDLE_MPI_ERROR(::MPI_Bcast(config_raw_data, N, MPI_UNSIGNED_LONG_LONG, root, comm));

  if (my_rank != root) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        config({row, col}) = config_raw_data[row * cols + col];
      }
    }
  }
  delete[]config_raw_data;
}

}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_H
