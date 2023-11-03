/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-06
*
* Description: GraceQ/VMC-PEPS project. The split index TPS class, where the tensor are stored in tensors splited in physical index
*/

#ifndef GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H
#define GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqpeps/two_dim_tn/tps/tps.h"                  // TPS

namespace gqpeps {
using namespace gqten;

// Helpers
inline std::string GenSplitIndexTPSTenName(const std::string &tps_path,
                                           const size_t row, const size_t col,
                                           const size_t compt) {
  return tps_path + "/" +
         kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "_" + std::to_string(compt) + "." +
         kGQTenFileSuffix;
}


template<typename TenElemT, typename QNT>
class SplitIndexTPS : public TenMatrix<std::vector<GQTensor<TenElemT, QNT>>> {
  using Tensor = GQTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using TN2D = TensorNetwork2D<TenElemT, QNT>;
 public:

  //constructor
  SplitIndexTPS(void) = default;    // default constructor for MPI

  SplitIndexTPS(const size_t rows, const size_t cols) : TenMatrix<std::vector<Tensor>>(rows, cols) {}

  SplitIndexTPS(const size_t rows, const size_t cols, const size_t phy_dim) : SplitIndexTPS(rows, cols) {
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      }
    }
  }

  SplitIndexTPS(const SplitIndexTPS &brotps) : TenMatrix<std::vector<GQTensor<TenElemT, QNT>>>(brotps) {}

  SplitIndexTPS(const TPST &tps) : TenMatrix<std::vector<Tensor>>(tps.rows(), tps.cols()) {
    const size_t phy_idx = 4;
    for (size_t row = 0; row < tps.rows(); row++) {
      for (size_t col = 0; col < tps.cols(); col++) {
        const Index<QNT> local_hilbert = tps({row, col}).GetIndex(phy_idx);
        const auto local_hilbert_inv = InverseIndex(local_hilbert);
        const size_t dim = local_hilbert.dim();
        (*this)({row, col}) = std::vector<Tensor>(dim);
        for (size_t i = 0; i < dim; i++) {
          Tensor project_ten = Tensor({local_hilbert_inv});
          project_ten({i}) = TenElemT(1.0);
          Contract(tps(row, col), {phy_idx}, &project_ten, {0}, &(*this)({row, col})[i]);
        }
      }
    }
  }

//  using TenMatrix<std::vector<Tensor>>::operator=;
  ///< using below explicitly definition to be compatible with lower version of g++
  SplitIndexTPS &operator=(const SplitIndexTPS &rhs) {
    TenMatrix<std::vector<Tensor>>::operator=(rhs);
    return *this;
  }

  // TODO
  operator TPST() {
    TPST tps(this->rows(), this->cols());

  }

  size_t PhysicalDim(void) const {
    return (*this)({0, 0}).size();
  }

  TN2D Project(const Configuration &config) const {
    assert(config.rows() == this->rows());
    assert(config.cols() == this->cols());
    return TN2D((*this), config);
  }

  size_t GetMaxBondDimension(void) const {
    size_t dmax = 0;
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        const Tensor &tensor = (*this)({row, col})[0];
        dmax = std::max(dmax, tensor.GetShape()[0]);
        dmax = std::max(dmax, tensor.GetShape()[1]);
      }
    }
    return dmax;
  };

  bool IsBondDimensionEven(void) const;

  SplitIndexTPS operator*(const TenElemT scalar) const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i] * scalar;
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator*=(const TenElemT scalar) {
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            (*this)({row, col})[i] *= scalar;
        }
      }
    }
    return *this;
  }

  SplitIndexTPS operator+(const SplitIndexTPS &right) const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault() && !right({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i] + right({row, col})[i];
          else if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = (*this)({row, col})[i];
          else if (!right({row, col})[i].IsDefault())
            res({row, col})[i] = right({row, col})[i];
        }
      }
    }
    return res;
  }

  SplitIndexTPS &operator+=(const SplitIndexTPS &right) {
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if ((*this)({row, col})[i].IsDefault())
            (*this)({row, col})[i] = right({row, col})[i];
          else if (!right({row, col})[i].IsDefault())
            (*this)({row, col})[i] += right({row, col})[i];
        }
      }
    }
    return *this;
  }

  ///< Inner product
  TenElemT operator*(const SplitIndexTPS &right) const {
    TenElemT res(0);
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if ((*this)({row, col})[i].IsDefault() || right({row, col})[i].IsDefault()) {
            continue;
          }
          Tensor ten_dag = Dag((*this)({row, col})[i]);
          Tensor scalar;
          Contract(&ten_dag, {0, 1, 2, 3}, &right({row, col})[i], {0, 1, 2, 3}, &scalar);
          res += scalar();
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator-() const {
    SplitIndexTPS res(this->rows(), this->cols());
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        res({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault())
            res({row, col})[i] = -(*this)({row, col})[i];
        }
      }
    }
    return res;
  }

  SplitIndexTPS operator-(const SplitIndexTPS &right) const {
    return (*this) + (-right);
  }

  ///< NB! not the wave function norm, it's the summation of tensor norm.
  ///< NB! the norm is defined as summation of element square, no square root.
  double Norm() const {
    double norm = 0;
    size_t phy_dim = PhysicalDim();
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        for (size_t i = 0; i < phy_dim; i++) {
          if (!(*this)({row, col})[i].IsDefault()) {
            double norm_local = (*this)({row, col})[i].Get2Norm();
            norm += norm_local * norm_local;
          }
        }
      }
    }
    return norm;
  }

  bool DumpTen(const size_t row, const size_t col, const size_t compt, const std::string &file) const {
    std::ofstream ofs(file, std::ofstream::binary);
    if (!ofs) {
      return false; // Failed to open the file
    }
    ofs << (*this)({row, col})[compt];
    if (!ofs) {
      return false; // Failed to write the tensor to the file
    }
    ofs.close();
    return true; // Successfully dumped the tensor
  }


  ///< assume has alloc memory
  bool LoadTen(const size_t row, const size_t col, const size_t compt, const std::string &file) {
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) {
      return false; // Failed to open the file
    }
    ifs >> (*this)({row, col})[compt];
    if (!ifs) {
      return false; // Failed to read the tensor from the file
    }
    ifs.close();
    return true; // Successfully loaded the tensor
  }

  // note don't confilt the name of tensors with original tps
  void Dump(const std::string &tps_path = kTpsPath, const bool release_mem = false);

  bool Load(const std::string &tps_path = kTpsPath);
};

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::IsBondDimensionEven(void) const {
  size_t d = (*this)({0, 0})[0].GetShape()[1];
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      const Tensor &tensor = (*this)({row, col})[0];
      if (row < this->rows() - 1 && d != tensor.GetShape()[1]) {
        return false;
      }
      if (col < this->cols() - 1 && d != tensor.GetShape()[2]) {
        return false;
      }
    }
  }
  return true;
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::Dump(const std::string &tps_path, const bool release_mem) {
  if (!gqmps2::IsPathExist(tps_path)) { gqmps2::CreatPath(tps_path); }
  std::string file;
  const size_t phy_dim = (*this)({0, 0}).size();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      for (size_t compt = 0; compt < phy_dim; compt++) {
        file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        this->DumpTen(row, col, compt, file);
      }
      if (release_mem) {
        this->dealloc(row, col);
      }
    }
  }
  file = tps_path + "/" + "phys_dim";
  std::ofstream ofs(file, std::ofstream::binary);
  ofs << phy_dim;
  ofs.close();
}

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::Load(const std::string &tps_path) {

  if (!gqmps2::IsPathExist(tps_path)) {
    std::cout << "No path " << tps_path << std::endl;
    return false;
  }
  std::string file = tps_path + "/" + "phys_dim";
  std::ifstream ifs(file, std::ofstream::binary);
  size_t phy_dim(0);
  ifs >> phy_dim;
  ifs.close();
  if (phy_dim == 0) {
    return false;
  }
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t compt = 0; compt < phy_dim; compt++) {
        file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        if (!(this->LoadTen(row, col, compt, file))) {
          return false;
        }
      }
    }
  }
  return true;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> operator*(const TenElemT scalar, const SplitIndexTPS<TenElemT, QNT> &split_idx_tps) {
  return split_idx_tps * scalar;
}

template<typename QNT>
SplitIndexTPS<GQTEN_Complex, QNT>
operator*(const GQTEN_Double scalar, const SplitIndexTPS<GQTEN_Complex, QNT> &split_idx_tps) {
  return split_idx_tps * GQTEN_Complex(scalar, 0.0);
}


template<typename TenElemT, typename QNT>
void CGSolverBroadCastVector(
    SplitIndexTPS<TenElemT, QNT> &v,
    boost::mpi::communicator &world
) {
  using Tensor = GQTensor<TenElemT, QNT>;
  size_t rows = v.rows(), cols = v.cols(), phy_dim = 0;
  broadcast(world, rows, kMasterProc);
  broadcast(world, cols, kMasterProc);
  if (world.rank() != kMasterProc) {
    v = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  } else {
    phy_dim = v({0, 0}).size();
  }
  broadcast(world, phy_dim, kMasterProc);
  if (world.rank() == kMasterProc) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        for (size_t compt = 0; compt < phy_dim; compt++) {
          SendBroadCastGQTensor(world, v({row, col})[compt], kMasterProc);
        }
      }
    }
  } else {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        v({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t compt = 0; compt < phy_dim; compt++) {
          RecvBroadCastGQTensor(world, v({row, col})[compt], kMasterProc);
        }
      }
    }
  }
}

template<typename TenElemT, typename QNT>
void CGSolverSendVector(
    boost::mpi::communicator &world,
    const SplitIndexTPS<TenElemT, QNT> &v,
    const size_t dest,
    const int tag
) {
  using Tensor = GQTensor<TenElemT, QNT>;
  size_t rows = v.rows(), cols = v.cols(), phy_dim = v({0, 0}).size();
  world.send(dest, tag, rows);
  world.send(dest, tag, cols);
  world.send(dest, tag, phy_dim);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      for (size_t compt = 0; compt < phy_dim; compt++) {
        const Tensor &ten = v({row, col})[compt];
        send_gqten(world, dest, tag, ten);
      }
    }
  }
}


template<typename TenElemT, typename QNT>
size_t CGSolverRecvVector(
    boost::mpi::communicator &world,
    SplitIndexTPS<TenElemT, QNT> &v,
    const size_t src,
    const int tag
) {
  using Tensor = GQTensor<TenElemT, QNT>;
  size_t rows, cols, phy_dim;
  boost::mpi::status status = world.recv(src, tag, rows);
  size_t actual_src = status.source();
  size_t actual_tag = status.tag();
  world.recv(actual_src, actual_tag, cols);
  world.recv(actual_src, actual_tag, phy_dim);
  v = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      v({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t compt = 0; compt < phy_dim; compt++) {
        Tensor &ten = v({row, col})[compt];
        recv_gqten(world, actual_src, actual_tag, ten);
      }
    }
  }
  return actual_src;
}

}//gqpeps


#endif //GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_H
