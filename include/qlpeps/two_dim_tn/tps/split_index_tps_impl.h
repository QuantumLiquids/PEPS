/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-12
*
* Description: QuantumLiquids/PEPS project. The split index TPS class, implementation for functions
*/

#ifndef QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
#define QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H

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
          double norm_local = (*this)({row, col})[i].GetQuasi2Norm();
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
      double norm_local = (*this)(site)[dim].GetQuasi2Norm();
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

/// scale the single site tensor on TPS to make the max abs of element equal to `aiming_max_abs`
template<typename TenElemT, typename QNT>
double SplitIndexTPS<TenElemT, QNT>::ScaleMaxAbsForSite(const qlpeps::SiteIdx &site, double aiming_max_abs) {
  const size_t phy_dim = PhysicalDim(site);
  double max_abs = 0;
  for (size_t dim = 0; dim < phy_dim; dim++) {
    const Tensor &ten = (*this)(site)[dim];
    if (!ten.IsDefault()) {
      max_abs = std::max(max_abs, ten.GetMaxAbs());
    }
  }
  double scale_factor = aiming_max_abs / max_abs;
  for (size_t dim = 0; dim < phy_dim; dim++) {
    if (!(*this)(site)[dim].IsDefault()) {
      (*this)(site)[dim] *= scale_factor;
    }
  }
  return 1.0 / scale_factor;
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::NormalizeAllSite() {
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      NormalizeSite({row, col});
    }
  }
}

///< Normalize split index tps according to the max abs of tensors in each site
template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ScaleMaxAbsForAllSite(double aiming_max_abs) {
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      ScaleMaxAbsForSite({row, col}, aiming_max_abs);
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

  file = tps_path + "/" + "tps_meta.txt";
  std::ofstream ofs(file, std::ofstream::binary);
  ofs << this->rows() << " " << this->cols() << " " << phy_dim;
  ofs.close();
}

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::Load(const std::string &tps_path) {
  if (!qlmps::IsPathExist(tps_path)) {
    std::cout << "No path " << tps_path << std::endl;
    return false;
  }

  size_t phy_dim = 0;
  std::string meta_file = tps_path + "/" + "tps_meta.txt";
  std::ifstream ifs(meta_file, std::ifstream::binary);

  if (ifs.good()) {
    // New format
    size_t rows = 0, cols = 0;
    ifs >> rows >> cols >> phy_dim;
    if (ifs.fail()) { return false; } // Handle corrupted meta file
    *this = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  } else {
    // Old format
    std::string old_meta_file = tps_path + "/" + "phys_dim";
    std::ifstream old_ifs(old_meta_file, std::ifstream::binary);
    if (!old_ifs.good()) { return false; } // No meta file found
    old_ifs >> phy_dim;
    if (old_ifs.fail()) { return false; } // Handle corrupted phys_dim file
    // For old format, assume `this` is already sized correctly.
  }

  if (phy_dim == 0) {
    return false;
  }

  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t compt = 0; compt < phy_dim; compt++) {
        std::string ten_file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        if (!this->LoadTen(row, col, compt, ten_file)) {
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

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> &SplitIndexTPS<TenElemT, QNT>::operator-=(const SplitIndexTPS &right) {
  (*this) = (*this) + (-right);
  return *this;
}

template<typename QNT>
SplitIndexTPS<QLTEN_Complex, QNT>
operator*(const QLTEN_Double scalar, const SplitIndexTPS<QLTEN_Complex, QNT> &split_idx_tps) {
  return split_idx_tps * QLTEN_Complex(scalar, 0.0);
}

template<typename TenElemT, typename QNT>
void BroadCast(
  SplitIndexTPS<TenElemT, QNT> &split_index_tps,
  const MPI_Comm &comm
) {
  CGSolverBroadCastVector(split_index_tps, comm);
}

template<typename TenElemT, typename QNT>
void CGSolverBroadCastVector(
  SplitIndexTPS<TenElemT, QNT> &v,
  const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  using Tensor = QLTensor<TenElemT, QNT>;
  size_t rows = v.rows(), cols = v.cols(), phy_dim = 0;
  HANDLE_MPI_ERROR(::MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG_LONG, kMPIMasterRank, comm));
  HANDLE_MPI_ERROR(::MPI_Bcast(&cols, 1, MPI_UNSIGNED_LONG_LONG, kMPIMasterRank, comm));
  if (rank != kMPIMasterRank) {
    v = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  } else {
    phy_dim = v.PhysicalDim();
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(&phy_dim, 1, MPI_UNSIGNED_LONG_LONG, kMPIMasterRank, comm));

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      if (rank != kMPIMasterRank) { v({row, col}) = std::vector<Tensor>(phy_dim); }
      for (size_t compt = 0; compt < phy_dim; compt++) {
        qlten::MPI_Bcast(v({row, col})[compt], kMPIMasterRank, comm);
      }
    }
  }
}

template<typename TenElemT, typename QNT>
void CGSolverSendVector(
  const MPI_Comm &comm,
  const SplitIndexTPS<TenElemT, QNT> &v,
  const size_t dest,
  const int tag
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  size_t peps_size[3] = {v.rows(), v.cols(), v.PhysicalDim()};
  ::MPI_Send(peps_size, 3, MPI_UNSIGNED_LONG_LONG, dest, tag, comm);
  for (auto &tens : v) {
    for (const Tensor &ten : tens) {
      ten.MPI_Send(dest, tag, comm);
    }
  }
}

template<typename TenElemT, typename QNT>
MPI_Status CGSolverRecvVector(
  const MPI_Comm &comm,
  SplitIndexTPS<TenElemT, QNT> &v,
  int src,
  int tag
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  size_t peps_size[3];
  MPI_Status status;
  HANDLE_MPI_ERROR(::MPI_Recv(peps_size, 3, MPI_UNSIGNED_LONG_LONG, src, tag, comm, &status));
  src = status.MPI_SOURCE;
  tag = status.MPI_TAG;
  auto [rows, cols, phy_dim] = peps_size;
  v = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  for (auto &tens : v) {
    tens = std::vector<Tensor>(phy_dim);
    for (Tensor &ten : tens) {
      status = ten.MPI_Recv(src, tag, comm);
    }
  }
  return status;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>::SplitIndexTPS(SplitIndexTPS &&other) noexcept
    : TenMatrix<std::vector<QLTensor<TenElemT, QNT>>>(std::move(other)) {
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> &SplitIndexTPS<TenElemT, QNT>::operator=(SplitIndexTPS &&other) noexcept {
  TenMatrix<std::vector<QLTensor<TenElemT, QNT>>>::operator=(std::move(other));
  return *this;
}
} //qlpeps

#endif //QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
