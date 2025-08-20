// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-27
*
* Description: QuantumLiquids/PEPS project. The generic PEPS class.
*/

#ifndef QLPEPS_TWO_DIM_TN_PEPS_PEPS_H
#define QLPEPS_TWO_DIM_TN_PEPS_PEPS_H


/**
 * @file peps.h
 * @brief Placeholder for the future lattice-agnostic PEPS base interface.
 *
 * @details
 * Status and intent:
 * - Today: the production code supports Square lattice with Open Boundary Conditions (OBC) only
 *   through concrete implementations in other headers.
 * - Near term: extend Square lattice to Periodic Boundary Conditions (PBC).
 * - Longer term: introduce a lattice-agnostic PEPS base class that cleanly separates core tensor
 *   storage/operations from lattice geometry and boundary policies, enabling multiple lattices
 *   (e.g., Square, Triangle, Honeycomb, Kagome) and multiple boundary conditions (OBC, PBC, ...).
 *
 * Design direction (non-breaking):
 * - Define an abstract base interface (name TBD, e.g., `IPeps`) for lattice/topology-agnostic
 *   functionality:
 *   - tensor container and layout abstraction
 *   - lattice geometry/neighbor access via traits or strategy objects
 *   - boundary handling (OBC/PBC/...) as a policy
 * - Provide concrete specializations that compose geometry and boundary policies, e.g.,
 *   `SquareOBCPeps`, `SquarePBCPeps`, and later lattice variants.
 *
 * Rationale:
 * - Eliminate scattered special-case branching in algorithms by moving variability into explicit
 *   data/trait objects.
 * - Preserve backward compatibility for existing Square OBC workflows.
 *
 * Migration plan:
 * 1) Keep existing Square OBC APIs stable.
 * 2) Introduce the base interface and adapters that wrap current implementations.
 * 3) Incrementally port algorithms to consume the base interface without breaking userspace.
 *
 * @note This header intentionally contains no public API yet. It documents the architectural
 *       direction and reserves the location for the future PEPS base class and related concepts.
 * @todo Define the minimal base interface and trait concepts. Implement Square PBC first to
 *       validate the boundary policy design, then generalize to additional lattices.
 */






#endif //QLPEPS_TWO_DIM_TN_PEPS_PEPS_H
