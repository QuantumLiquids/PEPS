/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: QuantumLiquids/PEPS project. Configuration Class.
*/
#ifndef QLPEPS_VMC_BASIC_CONFIGURATION_H
#define QLPEPS_VMC_BASIC_CONFIGURATION_H

#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <stdexcept>
#include "mpi.h"                // MPI BroadCast
#include "qlten/qlten.h"        // Showable, Streamable
#include "qlpeps/utility/filesystem_utils.h"  // IsPathExist, CreatePath
#include "qlpeps/two_dim_tn/framework/duomatrix.h"

namespace qlpeps {
using qlten::Showable;
using qlten::Streamable;

/**
 * @brief Occupancy number vector for U1 conserved systems
 * 
 * A non-negative integer number vector with size = local Hilbert space dimension.
 * occupancy_num[i] represents how many sites occupy the i-th state.
 * The summation of all elements in the vector equals the total site number.
 * 
 * Examples:
 * - For t-J model: vector of length 3 (spin up, spin down, empty)
 * - For Heisenberg model: vector of length 2 (spin up, spin down)
 * - For Hubbard model: vector of length 4 (empty, spin up, spin down, double occupied)
 * 
 * This is specifically useful when randomly generating configurations for
 * U1 conserved systems where particle number/spin number is conserved.
 */
using OccupancyNum = std::vector<size_t>;

/**
 * @brief Configuration class for quantum lattice models
 *
 * Represents a configuration of (direct-product) quantum states on a 2D square lattice.
 * Each site is labeled by a non-negative integer representing the local quantum state.
 *
 * For spin models: represents spin configurations (e.g., Ising configurations in Heisenberg model)
 * For fermion systems: represents fermion occupation patterns
 *
 * The configuration can be used for:
 * - Monte-Carlo sampling configurations in VMC (Variational Monte Carlo)
 * - MC-based measurement of correlation functions
 */
class Configuration : public DuoMatrix<size_t>, public Showable, public Streamable {
 public:

  using DuoMatrix<size_t>::DuoMatrix;
  using DuoMatrix<size_t>::operator();
  using DuoMatrix<size_t>::operator==;

  /// @brief Constructor from a 2D vector of size_t
  /// @param config_data 2D vector representing the configuration data
  Configuration(std::vector<std::vector<size_t>> config_data) : DuoMatrix<size_t>(config_data.size(),
                                                                                  config_data[0].size()) {
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        (*this)({row, col}) = config_data.at(row).at(col);
      }
    }
  }
  
  /// @brief Constructor with random initialization using Hilbert space dimension
  /// @param rows Number of rows in the lattice
  /// @param cols Number of columns in the lattice
  /// @param dim Dimension of local Hilbert space (0 to dim-1)
  /// @note The applied system does not preserve U1 conservation
  Configuration(size_t rows, size_t cols, const size_t dim) : DuoMatrix<size_t>(rows, cols) {
    this->Random(dim);
  }

  /// @brief Constructor with random initialization using occupancy numbers
  /// @param rows Number of rows in the lattice
  /// @param cols Number of columns in the lattice
  /// @param occupancy_num Vector specifying how many sites occupy each state
  Configuration(size_t rows, size_t cols, const OccupancyNum &occupancy_num) : DuoMatrix<size_t>(rows, cols) {
    this->Random(occupancy_num);
  }

  /// @brief Constructor with random initialization using configuration map
  /// @param rows Number of rows in the lattice
  /// @param cols Number of columns in the lattice
  /// @param config_map_to_occupancy_num Map from state index to number of sites
  Configuration(size_t rows, size_t cols, const std::map<size_t, size_t> config_map_to_occupancy_num) : DuoMatrix<size_t>(rows, cols) {
    this->Random(config_map_to_occupancy_num);
  }

  /// @brief Randomly generate a configuration with given Hilbert space dimension
  /// @param dim Dimension of local Hilbert space (states will be 0 to dim-1)
  /// @note This method does not preserve U1 conservation - each site is independently randomized
  void Random(const size_t dim) {
    std::random_device rd;
    std::mt19937 rand_num_gen(rd());
    for(size_t& state : *this){
      state = rand_num_gen() % dim;
    }
  }

  /**
   * @brief Randomly generate a configuration with specified occupancy numbers
   *
   * Creates a configuration where exactly occupancy_num[i] sites occupy state i.
   * This preserves U1 conservation by maintaining the specified particle numbers.
   *
   * @param occupancy_num Vector where occupancy_num[i] = number of sites in state i
   *                      The sum of all elements must equal total_sites = rows * cols
   * 
   * Examples:
   * - For 4x4 lattice with 8 spin-up and 8 spin-down: {8, 8}
   * - For 3x3 lattice with 5 empty, 3 single-occupied, 1 double-occupied: {5, 3, 1}
   * 
   * @throws std::invalid_argument if sum of occupancy numbers != total sites
   */
  void Random(const OccupancyNum &occupancy_num) {
    size_t dim = occupancy_num.size();
    size_t rows = this->rows();
    size_t cols = this->cols();
    size_t total_sites = rows * cols;
    
    // Calculate sum of occupancy numbers
    size_t sum_occupancy = 0;
    for (size_t i = 0; i < dim; i++) {
      sum_occupancy += occupancy_num[i];
    }
    
    // Validate that sum of occupancy numbers equals total sites
    if (sum_occupancy != total_sites) {
      throw std::invalid_argument("Random: Sum of occupancy numbers (" + 
                                 std::to_string(sum_occupancy) + 
                                 ") must equal total sites (" + 
                                 std::to_string(total_sites) + ")");
    }
    
    std::vector<size_t> configuration_list(total_sites);
    size_t offset = 0;
    
    // Fill the list with the specified number of each state
    for (size_t i = 0; i < dim; i++) {
      std::fill(configuration_list.begin() + offset, 
                configuration_list.begin() + offset + occupancy_num[i], i);
      offset += occupancy_num[i];
    }

    // Shuffle the configuration randomly
    std::random_device rd;
    std::mt19937 rand_num_gen(rd());
    std::shuffle(configuration_list.begin(), configuration_list.end(), rand_num_gen);

    // Apply the shuffled configuration
    this->SetFromList(configuration_list);
  }

   /**
   * @brief Randomly generate a configuration using a sparse occupancy map
   *
   * Advanced version of Random(occupancy_num) for large Hilbert spaces where
   * only a few states are occupied. This is more friendly to code.
   *
   * @param config_map_to_occupancy_num Map from state index to number of sites
   * 
   * Example: {{0, 10}, {2, 4}, {5, 8}} means:
   * - 10 sites in state 0
   * - 4 sites in state 2  
   * - 8 sites in state 5
   * - All other states have 0 sites
   * 
   * @throws std::invalid_argument if sum of occupancy numbers != total sites
   */
  void Random(const std::map<size_t, size_t> config_map_to_occupancy_num) {
    size_t rows = this->rows();
    size_t cols = this->cols();
    size_t total_sites = rows * cols;
    
    // Calculate sum of occupancy numbers
    size_t sum_occupancy = 0;
    for (auto [state, occupancy_num] : config_map_to_occupancy_num) {
      sum_occupancy += occupancy_num;
    }
    
    // Validate that sum of occupancy numbers equals total sites
    if (sum_occupancy != total_sites) {
      throw std::invalid_argument("Random: Sum of occupancy numbers (" + 
                                 std::to_string(sum_occupancy) + 
                                 ") must equal total sites (" + 
                                 std::to_string(total_sites) + ")");
    }
    
    std::vector<size_t> configuration_list(total_sites);
    size_t offset = 0;
    
    // Fill the list with the specified states
    for (auto [state, occupancy_num] : config_map_to_occupancy_num) {
      std::fill(configuration_list.begin() + offset, 
                configuration_list.begin() + offset + occupancy_num, state);
      offset += occupancy_num;
    }
    
    // Shuffle the configuration randomly
    std::random_device rd;
    std::mt19937 rand_num_gen(rd());
    std::shuffle(configuration_list.begin(), configuration_list.end(), rand_num_gen);

    // Apply the shuffled configuration
    this->SetFromList(configuration_list);
  }

  /// @brief Calculate the sum of all configuration values
  /// @return Sum of all site values
  size_t Sum(void) const {
    size_t sum = 0;
    for (auto elem : *this) {
      sum += elem;
    }
    return sum;
  }

  /**
   * @brief Count occupancy numbers for each species in the configuration
   * 
   * Returns a vector where occupancy[i] = number of sites in state i.
   * This is useful for validating U1 conservation in quantum systems.
   * 
   * @return Vector of occupancy numbers for each species
   */
  std::vector<size_t> CountOccupancy() const {
    if (this->size() == 0) return {};
    
    // Find maximum state value to determine vector size
    size_t max_state = 0;
    for (size_t elem : *this) {
      max_state = std::max(max_state, elem);
    }
    
    std::vector<size_t> occupancy(max_state + 1, 0);
    for (size_t elem : *this) {
      occupancy[elem]++;
    }
    
    return occupancy;
  }

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  /// @brief Check if all sites have valid (non-null) values
  /// @return true if all sites are valid, false otherwise
  bool IsValid(void) const {
    for(size_t row = 0; row < this->rows(); row++) {
      for(size_t col = 0; col < this->cols(); col++) {
        if((*this)(row, col) == nullptr){
          return false;
        }
      }
    }
    return true;
  }
  
  /**
   * @brief Save configuration to a binary file
   * @param path Directory path where to save the file
   * @param label Label for the file (e.g., MPI rank number)
   */
  void Dump(const std::string &path, const size_t label) {
    EnsureDirectoryExists(path);
    std::string file = path + "/configuration" + std::to_string(label);
    std::ofstream ofs(file, std::ofstream::binary);
    ofs << (*this);
    ofs.close();
  }

  /**
   * @brief Load configuration from a binary file
   * @param path Directory path where the file is located
   * @param label Label of the file to load (e.g., MPI rank number)
   * @return true if file was successfully loaded, false otherwise
   */
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
  /// @brief Private function to set configuration from a pre-filled list
  /// @param configuration_list Vector of configuration values (row-major order)
  void SetFromList(const std::vector<size_t> &configuration_list) {
    for (size_t row = 0; row < this->rows(); row++) {
      for (size_t col = 0; col < this->cols(); col++) {
        (*this)({row, col}) = configuration_list.at(row * this->cols() + col);
      }
    }
  }

};

void Configuration::StreamRead(std::istream &is) {
  for (size_t row = 0; row < this->rows(); row++) {
    for (size_t col = 0; col < this->cols(); col++) {
      if (!(is >> (*this)({row, col}))) {
        throw std::runtime_error("Configuration::StreamRead: Failed to read data from stream (row " +
                                 std::to_string(row) + ", col " + std::to_string(col) + ")");
      }
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
#endif //QLPEPS_VMC_BASIC_CONFIGURATION_H
