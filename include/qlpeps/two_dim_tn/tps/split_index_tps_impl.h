/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-12
*
* Description: QuantumLiquids/PEPS project. The split index TPS class, implementation for functions
*/

#ifndef GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
#define GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
size_t SplitIndexTPS<TenElemT, QNT>::GetMaxBondDimension() const {
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

template<typename TenElemT, typename QNT>
size_t SplitIndexTPS<TenElemT, QNT>::GetMinBondDimension() const {
  auto bond_dims = GetAllInnerBondDimensions();
  return *std::min_element(bond_dims.cbegin(), bond_dims.cend());
}

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> SplitIndexTPS<TenElemT, QNT>::GetMinMaxBondDimension() const {
  auto bond_dims = GetAllInnerBondDimensions();
  return std::make_pair(*std::min_element(bond_dims.cbegin(), bond_dims.cend()),
                        *std::max_element(bond_dims.cbegin(), bond_dims.cend()));
}

template<typename TenElemT, typename QNT>
std::vector<size_t> SplitIndexTPS<TenElemT, QNT>::GetAllInnerBondDimensions() const {
  std::vector<size_t> bond_dims;
  bond_dims.reserve(this->rows() * this->cols() - this->rows() - this->cols());
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      const Tensor &tensor = (*this)({row, col})[0];
      if (row < this->rows() - 1) {
        bond_dims.push_back(tensor.GetShape()[1]);
      }
      if (col < this->cols() - 1) {
        bond_dims.push_back(tensor.GetShape()[2]);
      }
    }
  }
  return bond_dims;
}

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
double SplitIndexTPS<TenElemT, QNT>::NormSquare() const {
  double norm_square = 0;
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      for (size_t i = 0; i < phy_dim; i++) {
        if (!(*this)({row, col})[i].IsDefault()) {
          double norm_local = (*this)({row, col})[i].Get2Norm();
          norm_square += norm_local * norm_local;
        }
      }
    }
  }
  return norm_square;
}

template<typename TenElemT, typename QNT>
double SplitIndexTPS<TenElemT, QNT>::NormalizeSite(const SiteIdx &site) {
  const size_t phy_dim = PhysicalDim(site);
  double norm_square(0);
  for (size_t dim = 0; dim < phy_dim; dim++) {
    if (!(*this)(site)[dim].IsDefault()) {
      double norm_local = (*this)(site)[dim].Get2Norm();
      norm_square += norm_local * norm_local;
    }
  }
  double norm = std::sqrt(norm_square);
  double inv = 1.0 / norm;
  for (size_t dim = 0; dim < phy_dim; dim++) {
    if (!(*this)(site)[dim].IsDefault()) {
      (*this)(site)[dim] *= inv;
    }
  }
  return norm;
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::NormalizeAllSite() {
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      NormalizeSite({row, col});
    }
  }
}

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::DumpTen(const size_t row,
                                           const size_t col,
                                           const size_t compt,
                                           const std::string &file) const {
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

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::LoadTen(const size_t row,
                                           const size_t col,
                                           const size_t compt,
                                           const std::string &file) {
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

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::Dump(const std::string &tps_path, const bool release_mem) {
  if (!qlmps::IsPathExist(tps_path)) { qlmps::CreatPath(tps_path); }
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

  if (!qlmps::IsPathExist(tps_path)) {
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
SplitIndexTPS<QLTEN_Complex, QNT>
operator*(const QLTEN_Double scalar, const SplitIndexTPS<QLTEN_Complex, QNT> &split_idx_tps) {
  return split_idx_tps * QLTEN_Complex(scalar, 0.0);
}

template<typename TenElemT, typename QNT>
void BroadCast(
    SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const boost::mpi::communicator &world
) {
  CGSolverBroadCastVector(split_index_tps, world);
}

template<typename TenElemT, typename QNT>
void CGSolverBroadCastVector(
    SplitIndexTPS<TenElemT, QNT> &v,
    const boost::mpi::communicator &world
) {
  using Tensor = QLTensor<TenElemT, QNT>;
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
          SendBroadCastQLTensor(world, v({row, col})[compt], kMasterProc);
        }
      }
    }
  } else {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        v({row, col}) = std::vector<Tensor>(phy_dim);
        for (size_t compt = 0; compt < phy_dim; compt++) {
          RecvBroadCastQLTensor(world, v({row, col})[compt], kMasterProc);
        }
      }
    }
  }
}

template<typename TenElemT, typename QNT>
void CGSolverSendVector(
    const boost::mpi::communicator &world,
    const SplitIndexTPS<TenElemT, QNT> &v,
    const size_t dest,
    const int tag
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  size_t rows = v.rows(), cols = v.cols(), phy_dim = v({0, 0}).size();
  world.send(dest, tag, rows);
  world.send(dest, tag, cols);
  world.send(dest, tag, phy_dim);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      for (size_t compt = 0; compt < phy_dim; compt++) {
        const Tensor &ten = v({row, col})[compt];
        send_qlten(world, dest, tag, ten);
      }
    }
  }
}

template<typename TenElemT, typename QNT>
size_t CGSolverRecvVector(
    const boost::mpi::communicator &world,
    SplitIndexTPS<TenElemT, QNT> &v,
    const size_t src,
    const int tag
) {
  using Tensor = QLTensor<TenElemT, QNT>;
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
        recv_qlten(world, actual_src, actual_tag, ten);
      }
    }
  }
  return actual_src;
}

}//qlpeps

#endif //GRACEQ_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
