/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: GraceQ/VMC-PEPS project. Vector data operations.
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VEC_DATA_OPERATIONS_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VEC_DATA_OPERATIONS_H

#include <vector>
#include <fstream>
#include <algorithm>

#include "boost/mpi.hpp"

#include "gqpeps/consts.h"      //kMasterProc

namespace gqpeps {
using namespace boost::mpi;

template<typename DataType>
void DumpVecData(
    const std::string &filename,
    const std::vector<DataType> &data
) {
  std::ofstream ofs(filename, std::ofstream::binary);
  for (auto datum : data) {
    ofs << datum << '\n';
  }
  ofs << std::endl;
  ofs.close();
}

template<typename T>
T Mean(const std::vector<T> data) {
  if (data.empty()) {
    return T(0);
  }
  auto const count = static_cast<T>(data.size());
//  return std::reduce(data.begin(), data.end()) / count;
  return std::accumulate(data.begin(), data.end(), T(0)) / count;
}

template<typename T>
double Variance(const std::vector<T> data,
                const T &mean) {
  size_t data_size = data.size();
  std::vector<T> diff(data_size);
  std::transform(data.begin(), data.end(), diff.begin(), [mean](T x) { return x - mean; });
  double sq_sum = std::transform_reduce(diff.begin(), diff.end(), 0.0, std::plus{},
                                        [](const T &num) {
                                          return std::norm(num);
                                        });
  auto const count = static_cast<double>(data_size);
  return sq_sum / count;
}

template<typename T>
double Variance(const std::vector<T> data) {
  return Variance(data, Mean(data));
}

template<typename T>
double StandardError(const std::vector<T> data,
                     const T &mean) {
  if (data.size() == 1) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(Variance(data, mean) / ((double) data.size() - 1.0));
}

template<typename T>
std::vector<T> AveListOfData(
    const std::vector<std::vector<T> > &data //outside idx: sample index; inside idx: something like site/bond
) {
  if (data.size() == 0) {
    return std::vector<T>();
  }
  const size_t N = data[0].size();
  if (N == 0) {
    return std::vector<T>();
  }
  const size_t sample_size = data.size();
  std::vector<T> sum(N, T(0)), ave(N);
  for (size_t sample_idx = 0; sample_idx < sample_size; sample_idx++) {
    for (size_t i = 0; i < N; i++) {
      sum[i] += data[sample_idx][i];
    }
  }
  for (size_t i = 0; i < N; i++) {
    ave[i] = sum[i] / sample_size;
  }
  return ave;
}

///< only rank 0 obatained the result.
template<typename ElemT>
std::pair<ElemT, double> GatherStatisticSingleData(
    ElemT data,
    MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  if (comm_size == 1) {
    return std::make_pair(data, std::numeric_limits<double>::infinity());
  }
  ElemT mean(0);
  double standard_err(0);

  ElemT *gather_data;
  if (comm_rank == kMasterProc) {
    gather_data = new ElemT[comm_size];
  }
  if (sizeof(ElemT) == sizeof(double)) {
    int err_msg = ::MPI_Gather((void *) &data, 1, MPI_DOUBLE, (void *) gather_data, 1, MPI_DOUBLE, kMasterProc, comm);
  } else if (sizeof(ElemT) == sizeof(std::complex<double>)) {
    int err_msg = ::MPI_Gather((void *) &data,
                               1,
                               MPI_DOUBLE_COMPLEX,
                               (void *) gather_data,
                               1,
                               MPI_DOUBLE_COMPLEX,
                               kMasterProc,
                               comm);
  }

  if (comm_rank == kMasterProc) {
    ElemT sum = 0.0;
    for (size_t i = 0; i < comm_size; i++) {
      sum += *(gather_data + i);
    }
    mean = sum / comm_size;
    if (comm_size > 1) {
      double sum_square = 0.0;
      for (size_t i = 0; i < comm_size; i++) {
        sum_square += std::norm(gather_data[i]);
      }
      double variance = sum_square / comm_size - std::norm(mean);
      standard_err = std::sqrt(variance / (comm_size - 1));
    }
    delete gather_data;
  }
  return std::make_pair(mean, standard_err);
}

template<typename ElemT>
void GatherStatisticListOfData(
    std::vector<ElemT> data,
    const boost::mpi::communicator &world,
    std::vector<ElemT> &avg, //output
    std::vector<double> &std_err//output
) {
  const size_t data_size = data.size(); // number of data
  if (data_size == 0) {
    avg = std::vector<ElemT>();
    std_err = std::vector<double>();
    return;
  }
  const size_t world_size = world.size();
  if (world_size == 1) {
    avg = data;
    std_err = std::vector<double>();
    return;
  }
  std::vector<ElemT> all_data(world_size * data_size);
  boost::mpi::gather(world, data.data(), data_size, all_data.data(), kMasterProc);

  if (world.rank() == kMasterProc) {
    std::vector<std::vector<ElemT>> data_gather_transposed(data_size, std::vector<ElemT>(world_size));
    for (size_t i = 0; i < world_size; i++) {
      for (size_t j = 0; j < data_size; j++) {
        data_gather_transposed[j][i] = all_data[i * data_size + j];
      }
    }
    avg.resize(data_size);
    std_err.resize(data_size);
    for (size_t i = 0; i < data_size; i++) {
      avg[i] = Mean(data_gather_transposed[i]);
      std_err[i] = StandardError(data_gather_transposed[i], avg[i]);
    }
  }
  return;
}

}//gqpeps
#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VEC_DATA_OPERATIONS_H
