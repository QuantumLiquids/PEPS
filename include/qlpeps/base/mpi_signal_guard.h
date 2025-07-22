/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-07-22
*
* Description: MPI-aware signal guard for emergency stop in Monte Carlo and other parallel computations.
*              Provides a utility for handling SIGINT/SIGTERM and synchronizing stop across MPI ranks.
*/

#ifndef QLPEPS_BASE_MPI_SIGNAL_GUARD_H
#define QLPEPS_BASE_MPI_SIGNAL_GUARD_H

#include <csignal>
#include <atomic>
#include <mpi.h>
#include <iostream>

namespace qlpeps {

class MPISignalGuard {
 public:
  // Returns true if any rank has received SIGINT/SIGTERM
  static bool EmergencyStopRequested(MPI_Comm comm) {
    int local_flag = g_emergency_stop_flag_.load() ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, comm);
    return global_flag != 0;
  }

  // Register the signal handler (call once per process, e.g. at the start of Execute)
  static void Register() {
    std::signal(SIGINT, &MPISignalGuard::SignalHandler);
    std::signal(SIGTERM, &MPISignalGuard::SignalHandler);
  }

  // Reset the emergency stop flag (optional, for repeated runs)
  static void Reset() {
    g_emergency_stop_flag_.store(false);
  }

  // Returns true if this rank received the signal
  static bool LocalStopRequested() {
    return g_emergency_stop_flag_.load();
  }

 private:
  static void SignalHandler(int) {
    g_emergency_stop_flag_.store(true);
    // Optionally print a message (only on first signal)
    static std::atomic_flag printed = ATOMIC_FLAG_INIT;
    if (!printed.test_and_set()) {
      std::cerr << "\n[MPISignalGuard] Emergency stop signal received. Will stop at next check.\n";
    }
  }

  static std::atomic<bool> g_emergency_stop_flag_;
};

// Definition of the static member
inline std::atomic<bool> MPISignalGuard::g_emergency_stop_flag_{false};

} // namespace qlpeps

#endif // QLPEPS_BASE_MPI_SIGNAL_GUARD_H