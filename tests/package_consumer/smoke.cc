// SPDX-License-Identifier: LGPL-3.0-only

#include "qlpeps/qlpeps.h"

#if defined(QLPEPS_VERIFY_EXPECT_SCALAPACK)
#include "qlpeps/optimizer/minsr_scalapack.h"
#endif

#if defined(QLPEPS_VERIFY_EXPECT_SCALAPACK) && !defined(QLPEPS_HAS_SCALAPACK)
#error "QLPEPS_HAS_SCALAPACK must be exported by ScaLAPACK-enabled PEPS packages."
#endif

int main() {
#if defined(QLPEPS_VERIFY_EXPECT_SCALAPACK)
  auto *scalapack_symbol = &numroc_;
  auto *blacs_symbol = &Cblacs_pinfo;
  return (scalapack_symbol == nullptr || blacs_symbol == nullptr) ? 1 : 0;
#else
  return 0;
#endif
}
