/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-12
*
* Description: QuantumLiquids/PEPS project. The split index TPS class, implementation for functions
*/

#ifndef QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
#define QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H

#include "qlpeps/utility/filesystem_utils.h"

namespace qlpeps {

// Helper function implementations
template<typename TenElemT, typename QNT>
template<typename Func>
void SplitIndexTPS<TenElemT, QNT>::ForEachValidTensor_(Func&& func) const {
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      for (size_t i = 0; i < phy_dim; ++i) {
        if (!(*this)({row, col})[i].IsDefault()) {
          func(row, col, i, (*this)({row, col})[i]);
        }
      }
    }
  }
}

template<typename TenElemT, typename QNT>
template<typename Func>
void SplitIndexTPS<TenElemT, QNT>::ForEachValidTensor_(Func&& func) {
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      for (size_t i = 0; i < phy_dim; ++i) {
        if (!(*this)({row, col})[i].IsDefault()) {
          func(row, col, i, (*this)({row, col})[i]);
        }
      }
    }
  }
}

template<typename TenElemT, typename QNT>
template<typename Func>
SplitIndexTPS<TenElemT, QNT> SplitIndexTPS<TenElemT, QNT>::ApplyBinaryOp_(const SplitIndexTPS &right, Func&& func) const {
  SplitIndexTPS res(this->rows(), this->cols());
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      res({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t i = 0; i < phy_dim; ++i) {
        func((*this)({row, col})[i], right({row, col})[i], res({row, col})[i]);
      }
    }
  }
  return res;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> SplitIndexTPS<TenElemT, QNT>::CreateInitializedResult_() const {
  SplitIndexTPS res(this->rows(), this->cols());
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      res({row, col}) = std::vector<Tensor>(phy_dim);
    }
  }
  return res;
}

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

/**
 * @brief Calculate the sum of squared quasi-2-norms of all tensor components
 * 
 * This function computes the sum of squared quasi-2-norms across all tensor
 * components and lattice sites. This is NOT the physical wave function norm.
 * 
 * The quasi-2-norm is defined as:
 * \f[
 *   \|A\|_{2,\mathrm{quasi}} = \sqrt{\sum_i |a_i|^2}
 * \f]
 * 
 * For fermionic tensors, this differs from the graded 2-norm:
 * \f[
 *   \|A\|_{2,\mathrm{graded}} = \sqrt{\sum_{i\in E} |a_i|^2\; - \; \sum_{i\in O} |a_i|^2}
 * \f]
 * where \f$E\f$ denotes even blocks and \f$O\f$ denotes odd blocks.
 * 
 * The quasi-2-norm is always well-defined and non-negative for both bosonic
 * and fermionic tensors, while the graded 2-norm can be ill-defined when
 * the odd contribution exceeds the even one.
 * 
 * @return Sum of squared quasi-2-norms: \f$\sum_{r,c,i} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2\f$
 * 
 * @note Uses `GetQuasi2Norm()` which is always well-defined for fermionic tensors
 * @see TensorToolkit documentation on fermionic tensor norms
 */
template<typename TenElemT, typename QNT>
double SplitIndexTPS<TenElemT, QNT>::NormSquare() const {
  double norm_square = 0;
  ForEachValidTensor_([&norm_square](size_t, size_t, size_t, const Tensor& ten) {
    double norm_local = ten.GetQuasi2Norm();
    norm_square += norm_local * norm_local;
  });
  return norm_square;
}

/**
 * @brief Normalize all tensor components at a specific site using quasi-2-norm
 * 
 * This function normalizes all tensor components at the given site such that
 * the sum of their squared quasi-2-norms equals 1:
 * \f[
 *   \sum_{i=0}^{d-1} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2 = 1
 * \f]
 * 
 * The normalization factor is computed as:
 * \f[
 *   \mathcal{N} = \sqrt{\sum_{i=0}^{d-1} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2}
 * \f]
 * 
 * Each tensor component is then scaled by \f$1/\mathcal{N}\f$:
 * \f[
 *   T_{r,c}^{(i)} \leftarrow \frac{T_{r,c}^{(i)}}{\mathcal{N}}
 * \f]
 * 
 * For fermionic tensors, this uses the quasi-2-norm which is always well-defined,
 * rather than the graded 2-norm which can be problematic when odd blocks dominate.
 * 
 * @param site The lattice site \f$(r,c)\f$ to normalize
 * @return The normalization factor \f$\mathcal{N}\f$ that was applied
 * 
 * @note Uses `GetQuasi2Norm()` for robust fermionic tensor handling
 * @note Default tensors are not affected by normalization
 * @see QuasiNormalize() in TensorToolkit for single tensor normalization
 */
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

// Scale the single site tensor on TPS to make the max abs of element equal to `aiming_max_abs`
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

// Normalize split index tps according to the max abs of tensors in each site
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
  EnsureDirectoryExists(tps_path);
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
  if (!ofs.is_open()) {
    throw std::ios_base::failure("Failed to open metadata file: " + file);
  }
  ofs << this->rows() << " " << this->cols() << " " << phy_dim;
  if (ofs.fail()) {
    throw std::ios_base::failure("Failed to write metadata to file: " + file);
  }
  ofs.close();
  if (ofs.fail()) {
    throw std::ios_base::failure("Failed to close metadata file: " + file);
  }
}

template<typename TenElemT, typename QNT>
bool SplitIndexTPS<TenElemT, QNT>::Load(const std::string &tps_path) {
  if (!IsPathExist(tps_path)) {
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
    if (ifs.fail()) {
#ifndef NDEBUG
      std::cerr << "[SplitIndexTPS::Load][DEBUG] Failed to parse metadata file: "
                << meta_file << std::endl;
#endif
      return false; // Handle corrupted meta file
    }
#ifndef NDEBUG
    std::cerr << "[SplitIndexTPS::Load][DEBUG] Using new format metadata. path="
              << tps_path << ", rows=" << rows << ", cols=" << cols
              << ", phy_dim=" << phy_dim << std::endl;
#endif
    *this = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  } else {
    // Old format
    std::string old_meta_file = tps_path + "/" + "phys_dim";
    std::ifstream old_ifs(old_meta_file, std::ifstream::binary);
    if (!old_ifs.good()) {
#ifndef NDEBUG
      std::cerr << "[SplitIndexTPS::Load][DEBUG] Neither new nor old metadata file exists. path="
                << tps_path << std::endl;
#endif
      return false; // No meta file found
    }
    old_ifs >> phy_dim;
    if (old_ifs.fail()) {
#ifndef NDEBUG
      std::cerr << "[SplitIndexTPS::Load][DEBUG] Failed to parse old metadata file: "
                << old_meta_file << std::endl;
#endif
      return false; // Handle corrupted phys_dim file
    }
#ifndef NDEBUG
    std::cerr << "[SplitIndexTPS::Load][DEBUG] Using old format metadata. path="
              << tps_path << ", phy_dim=" << phy_dim
              << " (rows/cols must be pre-sized: rows=" << this->rows()
              << ", cols=" << this->cols() << ")" << std::endl;
#endif
    // For old format, assume `this` is already sized correctly.
  }

  if (phy_dim == 0) {
#ifndef NDEBUG
    std::cerr << "[SplitIndexTPS::Load][DEBUG] Invalid phy_dim=0, aborting load. path="
              << tps_path << std::endl;
#endif
    return false;
  }

#ifndef NDEBUG
  std::cerr << "[SplitIndexTPS::Load][DEBUG] Begin loading tensors. total_sites="
            << (this->rows() * this->cols()) << ", phy_dim per site=" << phy_dim
            << std::endl;
#endif
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      (*this)({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t compt = 0; compt < phy_dim; compt++) {
        std::string ten_file = GenSplitIndexTPSTenName(tps_path, row, col, compt);
        if (!this->LoadTen(row, col, compt, ten_file)) {
#ifndef NDEBUG
          std::cerr << "[SplitIndexTPS::Load][DEBUG] Failed to load tensor file: row="
                    << row << ", col=" << col << ", compt=" << compt
                    << ", file=" << ten_file << std::endl;
#endif
          return false;
        }
#ifndef NDEBUG
        else {
          std::cerr << "[SplitIndexTPS::Load][DEBUG] Loaded tensor: row=" << row
                    << ", col=" << col << ", compt=" << compt
                    << ", file=" << ten_file << std::endl;
        }
#endif
      }
    }
  }
#ifndef NDEBUG
  std::cerr << "[SplitIndexTPS::Load][DEBUG] All tensors loaded successfully. path="
            << tps_path << std::endl;
#endif
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

// =============================
// Free out-of-place operations
// =============================
template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseSquare(const SplitIndexTPS<TenElemT, QNT> &tps) {
  return tps.ElementWiseSquare();
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseSqrt(const SplitIndexTPS<TenElemT, QNT> &tps) {
  return tps.ElementWiseSqrt();
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseInverse(const SplitIndexTPS<TenElemT, QNT> &tps, double epsilon) {
  return tps.ElementWiseInverse(epsilon);
}


template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
SplitIndexTPS<TenElemT, QNT>::ElementWiseSquare() const {
  SplitIndexTPS result = CreateInitializedResult_();
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      result({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t i = 0; i < phy_dim; ++i) {
        const Tensor &src = (*this)({row, col})[i];
        if (!src.IsDefault()) {
          result({row, col})[i] = src;
          result({row, col})[i].ElementWiseSquare();
        }
      }
    }
  }
  return result;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
SplitIndexTPS<TenElemT, QNT>::ElementWiseSqrt() const {
  SplitIndexTPS result = CreateInitializedResult_();
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      result({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t i = 0; i < phy_dim; ++i) {
        const Tensor &src = (*this)({row, col})[i];
        if (!src.IsDefault()) {
          result({row, col})[i] = src;
          result({row, col})[i].ElementWiseSqrt();
        }
      }
    }
  }
  return result;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
SplitIndexTPS<TenElemT, QNT>::ElementWiseInverse(double epsilon) const {
  SplitIndexTPS result = CreateInitializedResult_();
  const size_t phy_dim = PhysicalDim();
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      result({row, col}) = std::vector<Tensor>(phy_dim);
      for (size_t i = 0; i < phy_dim; ++i) {
        const Tensor &src = (*this)({row, col})[i];
        if (!src.IsDefault()) {
          result({row, col})[i] = src;
          result({row, col})[i].ElementWiseInv(epsilon);
        }
      }
    }
  }
  return result;
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ElementWiseClipTo(double clip_value) {
  ForEachValidTensor_([clip_value](size_t, size_t, size_t, Tensor &ten) {
    ten.ElementWiseClipTo(clip_value);
  });
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ClipByGlobalNorm(double clip_norm) {
  double norm_square = this->NormSquare();
  double r = std::sqrt(norm_square);
  if (r > 0.0 && r > clip_norm) {
    double scale = clip_norm / r;
    (*this) *= TenElemT(scale);
  }
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ElementWiseSquareInPlace() {
  ForEachValidTensor_([](size_t, size_t, size_t, Tensor &ten) {
    ten.ElementWiseSquare();
  });
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ElementWiseSqrtInPlace() {
  ForEachValidTensor_([](size_t, size_t, size_t, Tensor &ten) {
    ten.ElementWiseSqrt();
  });
}

template<typename TenElemT, typename QNT>
void SplitIndexTPS<TenElemT, QNT>::ElementWiseInverseInPlace(double epsilon) {
  ForEachValidTensor_([epsilon](size_t, size_t, size_t, Tensor &ten) {
    ten.ElementWiseInv(epsilon);
  });
}

// removed old InPlace aliases (now the primary member names)

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseClipTo(const SplitIndexTPS<TenElemT, QNT> &tps, double clip_value) {
  SplitIndexTPS<TenElemT, QNT> result = tps;
  result.ElementWiseClipTo(clip_value);
  return result;
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ClipByGlobalNorm(const SplitIndexTPS<TenElemT, QNT> &tps, double clip_norm) {
  SplitIndexTPS<TenElemT, QNT> result = tps;
  result.ClipByGlobalNorm(clip_norm);
  return result;
}









/**
 * @brief Send SplitIndexTPS to a destination MPI rank
 * 
 * This function provides a proper MPI send interface for SplitIndexTPS objects.
 * It first sends the dimensions, then sends all tensors sequentially.
 * 
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @param split_index_tps The SplitIndexTPS to send
 * @param dest Destination MPI rank
 * @param comm MPI communicator
 * @param tag MPI message tag
 */
template<typename TenElemT, typename QNT>
void MPI_Send(
  const SplitIndexTPS<TenElemT, QNT> &split_index_tps,
  const int dest,
  const MPI_Comm &comm,
  const int tag = 0
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  
  // Send dimensions first
  size_t peps_size[3] = {split_index_tps.rows(), split_index_tps.cols(), split_index_tps.PhysicalDim()};
  HANDLE_MPI_ERROR(::MPI_Send(peps_size, 3, MPI_UNSIGNED_LONG_LONG, dest, tag, comm));
  
  // Send all tensors
  for (const auto &tens : split_index_tps) {
    for (const Tensor &ten : tens) {
      ten.MPI_Send(dest, tag, comm);
    }
  }
}

/**
 * @brief Receive SplitIndexTPS from a source MPI rank
 * 
 * This function provides a proper MPI receive interface for SplitIndexTPS objects.
 * It first receives the dimensions, creates the appropriate structure, then receives all tensors.
 * 
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @param split_index_tps The SplitIndexTPS to receive into
 * @param src Source MPI rank, can be MPI_ANY_SOURCE
 * @param comm MPI communicator
 * @param tag MPI message tag
 * @return MPI status of the last receive operation
 */
template<typename TenElemT, typename QNT>
MPI_Status MPI_Recv(
  SplitIndexTPS<TenElemT, QNT> &split_index_tps,
  const int src,
  const MPI_Comm &comm,
  const int tag = 0
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  
  // Receive dimensions first
  size_t peps_size[3];
  MPI_Status status;
  HANDLE_MPI_ERROR(::MPI_Recv(peps_size, 3, MPI_UNSIGNED_LONG_LONG, src, tag, comm, &status));
  int actual_src = src;
  if (src == MPI_ANY_SOURCE) {
    actual_src = status.MPI_SOURCE;
  }
  #ifdef QLPEPS_MPI_DEBUG
  int dbg_rank = -1;
  MPI_Comm_rank(comm, &dbg_rank);
  std::cerr << "[MPI DEBUG] rank " << dbg_rank
            << " SplitIndexTPS::MPI_Recv header from "
            << (src == MPI_ANY_SOURCE ? std::string("ANY(actual ") + std::to_string(actual_src) + ")" : std::to_string(actual_src))
            << ", tag=" << tag << std::endl;
  #endif
  
  auto [rows, cols, phy_dim] = peps_size;
  split_index_tps = SplitIndexTPS<TenElemT, QNT>(rows, cols);
  
  // Receive all tensors
  for (auto &tens : split_index_tps) {
    tens = std::vector<Tensor>(phy_dim);
    for (Tensor &ten : tens) {
      status = ten.MPI_Recv(actual_src, tag, comm);
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

/**
 * @brief Broadcast SplitIndexTPS tensor to all MPI ranks
 * 
 * This function provides unified broadcasting for SplitIndexTPS objects.
 * All ranks must call this function with the same root rank.
 * 
 * @param v Tensor to broadcast (input on root, output on others)
 * @param comm MPI communicator
 * @param root Root rank that provides the data (default: 0)
 */
template<typename TenElemT, typename QNT>
void MPI_Bcast(
    SplitIndexTPS<TenElemT, QNT> &v,
    const MPI_Comm &comm,
    int root = 0
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  
  int rank;
  MPI_Comm_rank(comm, &rank);
  
  // Broadcast dimensions first
  size_t peps_size[3];
  if (rank == root) {
    peps_size[0] = v.rows();
    peps_size[1] = v.cols();
    peps_size[2] = v.PhysicalDim();
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(peps_size, 3, MPI_UNSIGNED_LONG_LONG, root, comm));
  
  auto [rows, cols, phy_dim] = peps_size;
  
  // Initialize tensor structure on non-root ranks
  if (rank != root) {
    v = SplitIndexTPS<TenElemT, QNT>(rows, cols);
    for (auto &tens : v) {
      tens = std::vector<Tensor>(phy_dim);
    }
  }
  
  // Broadcast all tensors
  for (auto &tens : v) {
    for (Tensor &ten : tens) {
      ten.MPI_Bcast(root, comm);
    }
  }
}

} //qlpeps

#endif //QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_IMPL_H
