---
title: Two-Phase MPI Broadcast for SplitIndexTPS
last_updated: 2026-03-19
status: Implemented
scope: TensorToolkit (QLTensor) + PEPS (SplitIndexTPS)
---

# Two-Phase MPI Broadcast for SplitIndexTPS

## Problem

`SplitIndexTPS` is broadcast across MPI ranks multiple times per VMC
iteration (energy/gradient evaluation, optimizer synchronization, etc.).
The broadcast goes through a single overload:
`qlpeps::MPI_Bcast(SplitIndexTPS&, comm, root)` in `split_index_tps_impl.h`.

Each `SplitIndexTPS` holds `rows × cols × phys_dim` tensors. The original
implementation broadcasts each `QLTensor` individually with 4 MPI collective
calls per tensor (is_default flag, shell size, shell bytes, raw data).
On a 16×16 lattice with d=2 this totals **2048 sequential MPI_Bcast calls
per broadcast** — excessive collective overhead that increases latency and
stresses MPI implementations.

## Design

Split the broadcast into two phases that match `QLTensor`'s internal
separation of metadata ("shell") from numerical data ("raw data").

### Phase 1 — Shell metadata (2 MPI calls)

On root, pack all tensor shells plus a small header into one contiguous
`std::vector<char>` buffer. Broadcast the buffer size (1 call), then the
buffer itself (1 call). Receivers unpack shells and allocate tensor memory.

Buffer layout:
```
[uint64 rows][uint64 cols][uint64 phys_dim][uint64 bc]  — 32-byte header
[tensor_0 shell record]                                  — is_default + shell + raw_data_count
[tensor_1 shell record]
...
```

For a 16×16 D=8 d=2 state this buffer is ~150 KB.

### Phase 2 — Raw data (≤ N_tensor MPI calls, zero copy)

Iterate all tensors in the same deterministic order on every rank. For each
non-default tensor, call `RawDataMPIBcast` which broadcasts directly
from/to the tensor's own memory — no intermediate buffer, no `memcpy`.
MPI implementations can use shared-memory or RDMA transport on the tensor's
storage.

## API

### QLTensor (TensorToolkit) additions

| Method | Purpose |
|--------|---------|
| `PackShellForMPI(std::vector<char>&)` | Append shell metadata (is_default, shell, raw_data_count) to a byte buffer. No raw data. |
| `UnpackShellForMPI(const char*&, const char*)` | Read shell from buffer, reconstruct indexes, allocate raw data memory. Does **not** fill raw data — caller must follow with `RawDataMPIBcast`. |

These complement the existing `PackForMPI`/`UnpackForMPI` (which include
raw data for single-buffer use cases) and `RawDataMPIBcast` (unchanged).

Supporting additions in `mpi_fun.h`:
- `GetMPIDataType<char>()` — enables `hp_numeric::MPI_Bcast` for `char` buffers.
- `GetMPIDataType<size_t>()` — platform-correct mapping (was previously hardcoded to `MPI_UNSIGNED_LONG_LONG`).
- `detail::AppendPod` / `detail::ReadPod` — shared by all Pack/Unpack methods.

### PEPS

The internal implementation of `qlpeps::MPI_Bcast(SplitIndexTPS&, ...)` is
replaced. The public signature is unchanged — no call-site modifications.

## Alternatives Considered

### Per-tensor broadcast (original)

2048 MPI calls for 16×16 d=2. Simple, no extra memory, but excessive
collective overhead.

### Single-buffer pack

Pack shells **and** raw data into one `std::vector<char>`. 2 MPI calls total.

Rejected:
- Copies all raw data into and out of the pack buffer (~66 MB for D=8,
  ~2 GB for D=16).
- Doubles peak memory during the broadcast.
- Defeats intra-node shared-memory MPI transport, which works best when
  sending directly from the source buffer.

### MPI derived datatypes (future)

Use `MPI_Type_create_hindexed` to describe the scattered raw-data layout
across all tensors, reducing phase 2 to a single `MPI_Bcast` call.

Deferred: `MPI_Type_commit` has non-trivial overhead, the type must be
recreated each broadcast (tensor pointers change), and some MPI
implementations internally copy non-contiguous types.

## Comparison

| Approach | MPI calls | Extra memcpy | Peak memory overhead |
|---|---|---|---|
| Per-tensor (original) | 4 × N | None | None |
| Single-buffer | 2 | 2× raw data | Raw data duplicated |
| **Two-phase (current)** | **2 + N** | **None** | **~150 KB shell buffer** |
| MPI derived types (future) | 3 | None | None |

where N = number of non-default tensors (≤ rows × cols × phys_dim).

## Edge Cases

**Default tensors.** Recorded as `is_default=1` with zero shell/data sizes
in phase 1. Skipped in phase 2. Both sides iterate in the same order,
guaranteed by the shared header.

**Scalar tensors (zero raw data).** `UnpackShellForMPI` reconstructs the
block-sparse structure so `RawDataMPIBcast` finds a matching zero-size
state. Both ranks enter `RawDataMPIBcast`'s scalar branch — the collective
matches, avoiding deadlock.

**Physical dimension uniformity.** `SplitIndexTPS::PhysicalDim()` reads
from site {0,0} and assumes uniform d. The two-phase approach preserves
this assumption.

## References

- MPI ownership contracts: `docs/dev/design/arch/mpi-contracts.md`
- SplitIndexTPS data structure: `docs/dev/design/data-structures/split-index-tps.md`
- Implementation: `include/qlpeps/two_dim_tn/tps/split_index_tps_impl.h`
