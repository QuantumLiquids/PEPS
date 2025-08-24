/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-06
*
* Description: QuantumLiquids/PEPS project. The split index TPS class, where the tensor are stored in tensors splited in physical index
*/

/**
@file split_index_tps.h
@brief Split-index TPS class API.
*/
#ifndef QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_H
#define QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"
#include "qlpeps/two_dim_tn/tps/tps.h"                  // TPS

namespace qlpeps {

// Helpers
inline std::string GenSplitIndexTPSTenName(const std::string &tps_path,
                                           const size_t row, const size_t col,
                                           const size_t compt) {
  return tps_path + "/" +
      kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "_" + std::to_string(compt) + "." +
      kQLTenFileSuffix;
}

/**
 * @brief Split-index Tensor Product State (TPS) class
 * 
 * This class represents a TPS where tensors are split along the physical index.
 * Each site contains a vector of tensors, one for each physical state component.
 * This representation is particularly useful for variational Monte Carlo calculations
 * and tensor network contractions where physical indices need to be projected.
 * 
 * The class supports standard tensor network operations including:
 * - Arithmetic operations (addition, subtraction, scalar multiplication)
 * - Inner products and norm calculations
 * - Serialization and deserialization (Dump/Load)
 * - MPI communication (Send/Recv/Broadcast)
 * - Site-wise normalization and scaling
 * - Element-wise transforms (square, sqrt, safe-inverse with epsilon)
 *   - In-place member operations: mutate tensors in this object
 *   - Out-of-place free functions: return a new `SplitIndexTPS` with transform
 * 
 * Bosonic vs. fermionic conventions:
 * - Bosonic tensors have 4 virtual indices; contractions typically use index set {0,1,2,3}.
 * - Fermionic tensors carry an extra 1-dim parity index (as the last index). When contracting
 *   or forming inner products, use index set {0,1,2,3,4} and apply `ActFermionPOps()`
 *   to ensure correct graded algebra.
 * - When converting from a regular TPS, fermionic components are projected by matching
 *   quantum number sectors, while bosonic components use a simple Kronecker projection.
 *
 * @note The operator() method from the base class TenMatrix will automatically
 * allocate memory for elements when accessed if they haven't been initialized.
 * For std::vector<Tensor> elements, this means creating an empty vector.
 * Users should ensure proper initialization before accessing elements.
 * 
 * @note This class uses helper functions with trailing underscores for private
 * methods following the coding convention.
 * 
 * @tparam TenElemT Tensor element type (typically QLTEN_Double or QLTEN_Complex)
 * @tparam QNT Quantum number type (e.g., U1QN for U(1) symmetry)
 * 
 * @see TPS for the non-split index version
 * @see TensorNetwork2D for tensor network contraction functionality
 * 
 * Example usage:
 * @code
 * using SplitTPS = SplitIndexTPS<QLTEN_Double, U1QN>;
 * SplitTPS split_tps(4, 4, 2);  // 4x4 lattice with physical dimension 2
 * split_tps.NormalizeAllSite();  // Normalize all sites
 * auto norm_sq = split_tps.NormSquare();  // Calculate sum of squared quasi-2-norms
 * @endcode
 */
template<typename TenElemT, typename QNT>
class SplitIndexTPS : public TenMatrix<std::vector<QLTensor<TenElemT, QNT>>> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using TN2D = TensorNetwork2D<TenElemT, QNT>;
 public:

  // Constructors
  
  /**
   * @brief Default constructor
   * 
   * Creates an empty SplitIndexTPS. This constructor is primarily used for MPI
   * operations where the object will be populated via receive operations.
   */
  SplitIndexTPS(void) = default;
  
  /**
   * @brief Constructor with dimensions
   * 
   * Creates a SplitIndexTPS with specified lattice dimensions.
   * Tensors are not initialized and must be set manually.
   * 
   * @param rows Number of rows in the lattice
   * @param cols Number of columns in the lattice
   */
  SplitIndexTPS(const size_t rows, const size_t cols) : TenMatrix<std::vector<Tensor>>(rows, cols) {}
  
  /**
   * @brief Constructor with dimensions and physical dimension
   * 
   * Creates a SplitIndexTPS with specified lattice and physical dimensions.
   * Each site is initialized with a vector of default tensors.
   * 
   * @param rows Number of rows in the lattice
   * @param cols Number of columns in the lattice  
   * @param phy_dim Physical dimension at each site
   */
  SplitIndexTPS(const size_t rows, const size_t cols, const size_t phy_dim) : SplitIndexTPS(rows, cols) {
    for (auto &split_ten : *this) {
      split_ten = std::vector<Tensor>(phy_dim);
    }
  }
  
  /**
   * @brief Copy constructor
   * 
   * @param rhs The SplitIndexTPS to copy from
   */
  SplitIndexTPS(const SplitIndexTPS &rhs) : TenMatrix<std::vector<QLTensor<TenElemT, QNT>>>(rhs) {}
  
  /**
   * @brief Move constructor
   * 
   * @param other The SplitIndexTPS to move from
   */
  SplitIndexTPS(SplitIndexTPS &&other) noexcept;
  
  /**
   * @brief Move assignment operator
   * 
   * @param other The SplitIndexTPS to move from
   * @return Reference to this object
   */
  SplitIndexTPS &operator=(SplitIndexTPS &&other) noexcept;


  /**
   * @brief Constructor from regular TPS
   * 
   * Converts a regular TPS to split-index format by projecting each tensor
   * onto its physical index components. For fermionic tensors, proper quantum
   * number handling is maintained.
   * 
   * @param tps The source TPS to convert from
   * 
   * @note The physical index is assumed to be at position 4 in the tensor.
   * @note For fermionic tensors, appropriate quantum number sectors are created.
   */
  SplitIndexTPS(const TPST &tps) : TenMatrix<std::vector<Tensor>>(tps.rows(), tps.cols()) {
    const size_t phy_idx = 4;
    for (size_t row = 0; row < tps.rows(); row++) {
      for (size_t col = 0; col < tps.cols(); col++) {
        const Index<QNT> local_hilbert = tps({row, col}).GetIndex(phy_idx);
        const auto local_hilbert_inv = InverseIndex(local_hilbert);
        const size_t dim = local_hilbert.dim();
        (*this)({row, col}) = std::vector<Tensor>(dim);
        for (size_t i = 0; i < dim; i++) {
          if constexpr (Tensor::IsFermionic()) {
            QNT qn = local_hilbert.GetQNSctFromActualCoor(i).GetQn();
            Index<QNT> match_idx = Index<QNT>({QNSector<QNT>(qn, 1)}, IN);
            Tensor project_ten = Tensor({local_hilbert_inv, match_idx});
            project_ten({i, 0}) = TenElemT(1.0);
            Contract(tps(row, col), {phy_idx}, &project_ten, {0}, &(*this)({row, col})[i]);
          } else {
            Tensor project_ten = Tensor({local_hilbert_inv});
            project_ten({i}) = TenElemT(1.0);
            Contract(tps(row, col), {phy_idx}, &project_ten, {0}, &(*this)({row, col})[i]);
          }
        }
      }
    }
  }

  // using TenMatrix<std::vector<Tensor>>::operator=;
  // Using explicit definition below for compatibility with lower version of g++
  SplitIndexTPS &operator=(const SplitIndexTPS &rhs) {
    TenMatrix<std::vector<Tensor>>::operator=(rhs);
    return *this;
  }

  /**
   * @brief Convert split-index TPS back to regular TPS format
   * 
   * Combines the split tensor components back into regular tensors with
   * the specified physical index. This is the inverse operation of the
   * constructor from TPS.
   * 
   * @param phy_idx The physical index to use for the resulting TPS
   * @return Regular TPS with combined physical indices
   * 
   * @note For fermionic tensors, uses Expand operation to combine components
   * @note For bosonic tensors, uses weighted sum with projection tensors
   */
  TPST GroupIndices(const Index<QNT> &phy_idx) const {
    TPST tps(this->rows(), this->cols());
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        const std::vector<Tensor> &split_tens = (*this)({row, col});
        Tensor combined_ten;
        if constexpr (Tensor::IsFermionic()) {
          combined_ten = split_tens.front();
          for (size_t i = 1; i < split_tens.size(); i++) {
            Tensor tmp;
            Expand(&combined_ten, &split_tens[i], {4}, &tmp);
            combined_ten = tmp;
          }
        } else {
          for (size_t i = 0; i < split_tens.size(); i++) {
            Tensor leg_ten({phy_idx});
            leg_ten({i}) = TenElemT(1.0);
            std::vector<std::vector<size_t>> empty_vv = {{}, {}};
            const Tensor &split_ten_comp = split_tens[i];
            Tensor split_ten_with_phy_leg;
            Contract(&split_ten_comp, &leg_ten, empty_vv, &split_ten_with_phy_leg);
            if (i == 0) {
              combined_ten = split_ten_with_phy_leg;
            } else {
              combined_ten += split_ten_with_phy_leg;
            }
          }
        }
        tps({row, col}) = combined_ten;
      }
    }
    return tps;
  }

  /**
   * @brief Get the physical dimension at a specific site
   * 
   * @param site The site to query (defaults to {0,0})
   * @return Physical dimension (number of tensor components) at the site
   */
  size_t PhysicalDim(const SiteIdx &site = {0, 0}) const {
    return (*this)(site).size();
  }

  /**
   * @brief Project the SplitIndexTPS with a given configuration
   * 
   * Creates a 2D tensor network by selecting specific tensor components
   * at each site according to the configuration.
   * 
   * @param config Configuration specifying which component to select at each site
   * @return TensorNetwork2D representing the projected state
   * 
   * @pre config.rows() == this->rows() && config.cols() == this->cols()
   */
  TN2D Project(const Configuration &config) const {
    assert(config.rows() == this->rows());
    assert(config.cols() == this->cols());
    return TN2D((*this), config);
  }

  /**
   * @brief Get minimal inner bond dimension across the network
   * @return Minimal dimension among all inner bonds
   */
  size_t GetMinBondDimension(void) const;

  /**
   * @brief Get maximal inner bond dimension across the network
   * @return Maximal dimension among all inner bonds
   */
  size_t GetMaxBondDimension(void) const;

  /**
   * @brief Get both minimal and maximal inner bond dimensions
   * @return A pair {min_dim, max_dim}
   */
  std::pair<size_t, size_t> GetMinMaxBondDimension(void) const;

  /**
   * @brief Collect all inner bond dimensions
   * @return Vector of inner bond dimensions in scan order
   */
  std::vector<size_t> GetAllInnerBondDimensions(void) const;

  /**
   * @brief Check if inner bond dimensions are uniform (even) on the bulk bonds
   * @return true if all inner bonds share the same dimension
   */
  bool IsBondDimensionEven(void) const;

  /**
   * @brief Scalar multiplication operator
   * 
   * Multiplies all tensor components by a scalar value.
   * Only non-default tensors are affected.
   * 
   * @param scalar The scalar to multiply by
   * @return New SplitIndexTPS with scaled tensors
   */
  SplitIndexTPS operator*(const TenElemT scalar) const {
    SplitIndexTPS res = CreateInitializedResult_();
    ForEachValidTensor_([&res, scalar](size_t row, size_t col, size_t i, const Tensor& ten) {
      res({row, col})[i] = ten * scalar;
    });
    return res;
  }

  /**
   * @brief In-place scalar multiplication operator
   * 
   * Multiplies all tensor components by a scalar value in-place.
   * Only non-default tensors are affected.
   * 
   * @param scalar The scalar to multiply by
   * @return Reference to this object
   */
  SplitIndexTPS operator*=(const TenElemT scalar) {
    ForEachValidTensor_([scalar](size_t, size_t, size_t, Tensor& ten) {
      ten *= scalar;
    });
    return *this;
  }

  /**
   * @brief Addition operator
   * 
   * Adds corresponding tensor components from two SplitIndexTPS objects.
   * Handles cases where one or both tensors may be in default state.
   * 
   * @param right The SplitIndexTPS to add
   * @return New SplitIndexTPS containing the sum
   */
  SplitIndexTPS operator+(const SplitIndexTPS &right) const {
    return ApplyBinaryOp_(right, [](const Tensor& left, const Tensor& right, Tensor& result) {
      if (!left.IsDefault() && !right.IsDefault())
        result = left + right;
      else if (!left.IsDefault())
        result = left;
      else if (!right.IsDefault())
        result = right;
    });
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

  /**
   * @brief Inner product operator
   * 
   * Computes the inner product between this SplitIndexTPS and another one.
   * The result is equivalent to Dag(*this) * right, summing over all
   * tensor components and lattice sites.
   * 
   * @param right The SplitIndexTPS to compute inner product with
   * @return The complex/real valued inner product
   * 
   * @note For fermionic tensors, proper fermion parity operations are applied
   * @note Only non-default tensor components contribute to the result
   */
  TenElemT operator*(const SplitIndexTPS &right) const {
    TenElemT res(0);
    ForEachValidTensor_([&res, &right](size_t row, size_t col, size_t i, const Tensor& ten) {
      if (right({row, col})[i].IsDefault()) return;
      
      Tensor ten_dag = Dag(ten);
      Tensor scalar;
      if constexpr (Tensor::IsFermionic()) {
        ten_dag.ActFermionPOps();
        Contract(&ten_dag, {0, 1, 2, 3, 4}, &right({row, col})[i], {0, 1, 2, 3, 4}, &scalar);
      } else {
        Contract(&ten_dag, {0, 1, 2, 3}, &right({row, col})[i], {0, 1, 2, 3}, &scalar);
      }
      res += TenElemT(scalar());
    });
    return res;
  }

  void ActFermionPOps() {
    if constexpr (Tensor::IsFermionic()) {
      ForEachValidTensor_([](size_t, size_t, size_t, Tensor& ten) {
        ten.ActFermionPOps();
      });
    }
  }

  SplitIndexTPS operator-() const {
    SplitIndexTPS res = CreateInitializedResult_();
    ForEachValidTensor_([&res](size_t row, size_t col, size_t i, const Tensor& ten) {
      res({row, col})[i] = -ten;
    });
    return res;
  }

  SplitIndexTPS operator-(const SplitIndexTPS &right) const {
    return (*this) + (-right);
  }

  SplitIndexTPS &operator-=(const SplitIndexTPS &);

  /**
   * @brief Calculate sum of squared quasi-2-norms of all tensor components
   * 
   * Computes \f$\sum_{r,c,i} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2\f$ where
   * \f$\|A\|_{2,\mathrm{quasi}} = \sqrt{\sum_j |a_j|^2}\f$ is the quasi-2-norm.
   * 
   * @warning This is NOT the physical wave function norm!
   * @return Sum of squared quasi-2-norms across all sites and components
   * @note For fermionic tensors, uses quasi-norm (always well-defined)
   *       vs. graded norm which can be ill-defined
   */
  double NormSquare() const;

  /**
   * @brief Normalize all tensor components at a site using quasi-2-norm
   * 
   * Normalizes tensors such that \f$\sum_{i=0}^{d-1} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2 = 1\f$.
   * For fermionic tensors, this uses the robust quasi-2-norm rather than 
   * the potentially ill-defined graded 2-norm.
   *
   * @param site The lattice site to normalize
   * @return The normalization factor that was applied
   * @note Uses quasi-2-norm for fermionic tensor stability
   */
  double NormalizeSite(const SiteIdx &site);
  /**
   * @brief Scale tensor components at a site to achieve target maximum absolute value
   * 
   * Finds the maximum absolute value among all tensor elements at the given site,
   * then scales all tensor components by \f$\frac{\text{aiming\_max\_abs}}{\text{current\_max\_abs}}\f$.
   * 
   * @param site The lattice site to scale  
   * @param aiming_max_abs Target maximum absolute value
   * @return The inverse of the scaling factor applied (i.e., original_max_abs/aiming_max_abs)
   * @note Only non-default tensors are affected
   */
  double ScaleMaxAbsForSite(const SiteIdx &site, double aiming_max_abs);

  /**
   * @brief Normalize all sites using NormalizeSite()
   * 
   * Applies quasi-2-norm normalization to every site in the lattice.
   * Each site will have \f$\sum_{i} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2 = 1\f$ after this operation.
   */
  void NormalizeAllSite();
  
  /**
   * @brief Scale all sites using ScaleMaxAbsForSite()
   * 
   * Applies maximum absolute value scaling to every site in the lattice.
   * 
   * @param aiming_max_abs Target maximum absolute value for all sites
   */
  void ScaleMaxAbsForAllSite(double aiming_max_abs);

  /**
   * @brief Return a SplitIndexTPS with element-wise square applied
   *
   * Applies for every site (r,c) and component i the transform
   * \( T_{r,c}^{(i)} \leftarrow T_{r,c}^{(i)} \odot T_{r,c}^{(i)} \),
   * where \(\odot\) denotes element-wise product. Delegates to
   * `QLTensor::ElementWiseSquare()` for each tensor.
   */
  SplitIndexTPS ElementWiseSquare() const;

  /**
   * @brief Return a SplitIndexTPS with element-wise square root applied
   *
   * Applies for every site (r,c) and component i the transform
   * \( T_{r,c}^{(i)} \leftarrow \sqrt{T_{r,c}^{(i)}} \)
   * element-wise. Delegates to `QLTensor::ElementWiseSqrt()`.
   */
  SplitIndexTPS ElementWiseSqrt() const;

  /**
   * @brief Return a SplitIndexTPS with element-wise safe inverse applied
   *
   * Applies for every site (r,c) and component i the transform
   * element-wise safe reciprocal with epsilon guard. Delegates to
   * `QLTensor::ElementWiseInv(epsilon)`.
   *
   * @param epsilon Small positive guard to avoid division by near-zero values
   */
  SplitIndexTPS ElementWiseInverse(double epsilon) const;

  /**
   * @brief In-place per-element magnitude clipping (complex-safe)
   *
   * Clip each element's magnitude to clip_value while preserving phase/sign.
   * Delegates to `QLTensor::ElementWiseBoundTo(clip_value)`.
   *
   * @param clip_value Per-element magnitude clip threshold (>0)
   */
  void ElementWiseClipTo(double clip_value);

  /**
   * @brief In-place global L2 norm clipping
   *
   * Let r = sqrt(Σ |g_j|^2) across all elements and components. If r > clip_norm,
   * uniformly scale all tensors by (clip_norm / r); otherwise unchanged.
   *
   * For fermion tensor, norm use quasi-norm.
   * @param clip_norm Global L2 norm threshold (>0)
   */
  void ClipByGlobalNorm(double clip_norm);

  /**
   * @brief In-place element-wise square on all non-default tensors
   */
  void ElementWiseSquareInPlace();

  /**
   * @brief In-place element-wise square root on all non-default tensors
   */
  void ElementWiseSqrtInPlace();

  /**
   * @brief In-place element-wise safe inverse with epsilon guard
   * @param epsilon Small positive guard to avoid division by near-zero values
   */
  void ElementWiseInverseInPlace(double epsilon);

  // (Names aligned with free functions: member versions are in-place)

  /**
   * @brief Dump one split tensor component to a file
   * @param row Lattice row
   * @param col Lattice column
   * @param compt Physical component index (0..d-1)
   * @param file Target file path
   * @return true if succeeded
   */
  bool DumpTen(const size_t row, const size_t col, const size_t compt, const std::string &file) const;

  // Assumes memory has been allocated
  /**
   * @brief Load one split tensor component from a file
   * @param row Lattice row
   * @param col Lattice column
   * @param compt Physical component index (0..d-1)
   * @param file Source file path
   * @return true if succeeded
   */
  bool LoadTen(const size_t row, const size_t col, const size_t compt, const std::string &file);

  /**
   * @brief Dump the whole SplitIndexTPS to a directory
   * 
   * Layout:
   * - Tensor files: kTpsTenBaseName + "row_col_compt" + "." + kQLTenFileSuffix
   * - Metadata:     tps_meta.txt containing: rows cols phy_dim
   *
   * @param tps_path Target directory path (created if absent)
   * @param release_mem Whether to deallocate site tensors after dumping
   */
  void Dump(const std::string &tps_path = kTpsPath, const bool release_mem = false);

  /**
   * @brief Load the whole SplitIndexTPS from a directory
   * 
   * Accepts both new format (with tps_meta.txt) and old format (with file "phys_dim").
   * For the old format, the matrix size must be already set by the caller.
   *
   * @param tps_path Source directory path
   * @return true if succeeded
   */
  bool Load(const std::string &tps_path = kTpsPath);

private:
  /// Helper function to apply operation to all valid tensors
  template<typename Func>
  void ForEachValidTensor_(Func&& func) const;
  
  /// Helper function to apply operation to all valid tensors (non-const version)
  template<typename Func>
  void ForEachValidTensor_(Func&& func);
  
  /// Helper function for element-wise operations between two SplitIndexTPS
  template<typename Func>
  SplitIndexTPS ApplyBinaryOp_(const SplitIndexTPS &right, Func&& func) const;
  
  /// Helper function to create a properly initialized result SplitIndexTPS
  SplitIndexTPS CreateInitializedResult_() const;
};

// Out-of-place free functions for element-wise operations
// These are defined in the impl header after the class template definitions
// to avoid circular dependencies in templates.

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseSquare(const SplitIndexTPS<TenElemT, QNT> &tps);

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseSqrt(const SplitIndexTPS<TenElemT, QNT> &tps);

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseInverse(const SplitIndexTPS<TenElemT, QNT> &tps, double epsilon);

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ElementWiseClipTo(const SplitIndexTPS<TenElemT, QNT> &tps, double clip_value);

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT>
ClipByGlobalNorm(const SplitIndexTPS<TenElemT, QNT> &tps, double clip_norm);

}//qlpeps

#include "qlpeps/two_dim_tn/tps/split_index_tps_impl.h"
#endif //QLPEPS_VMC_PEPS_SPLIT_INDEX_TPS_H
