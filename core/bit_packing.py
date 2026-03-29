"""Bit-level packing of N-bit quantized codes into compact byte arrays.

Reduces storage from 1 byte per code to N/8 bytes per code by packing
arbitrary-width signed integer codes into dense uint8 buffers.

Storage savings vs int8:
    Q2: 2/8 = 25%       Q5: 5/8 = 62.5%
    Q3: 3/8 = 37.5%     Q6: 6/8 = 75%
    Q4: 4/8 = 50%        Q7: 7/8 = 87.5%

Public API:
    pack_codes          -- generic N-bit packing via numpy vectorized bit ops
    unpack_codes        -- generic N-bit unpacking
    pack_codes_fast     -- optimised paths for Q4/Q5, falls back to generic
    unpack_codes_fast   -- optimised paths for Q4/Q5, falls back to generic
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offset(bits: int) -> int:
    """Unsigned offset: maps signed [-2^(bits-1), 2^(bits-1)-1] to [0, 2^bits - 1]."""
    return 1 << (bits - 1)


def _validate_bits(bits: int) -> None:
    if not (2 <= bits <= 8):
        raise ValueError(f"bits must be in [2, 8], got {bits}")


# ===================================================================
# Generic pack / unpack  (numpy-vectorised bit scatter/gather)
# ===================================================================

def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed N-bit integer codes into a uint8 byte array.

    Args:
        codes: int8/int16/int32 tensor with values in [-2^(bits-1), 2^(bits-1)-1].
        bits:  Bit width, one of 2, 3, 4, 5, 6, 7, 8.

    Returns:
        uint8 tensor with ceil(n * bits / 8) bytes.
    """
    _validate_bits(bits)

    if bits == 8:
        # No packing needed -- just reinterpret as unsigned.
        return (codes.flatten().to(torch.int16) + _offset(8)).to(torch.uint8)

    flat = codes.flatten()
    n = flat.numel()
    n_packed_bytes = math.ceil(n * bits / 8)

    # Convert to numpy for fast vectorised bit operations.
    unsigned = flat.to(torch.int16).numpy().astype(np.int16) + np.int16(_offset(bits))
    unsigned = unsigned.astype(np.uint16)

    # Pre-compute which bits of unsigned values go into which byte/bit of output.
    # Each value contributes `bits` individual bits.
    out = np.zeros(n_packed_bytes, dtype=np.uint8)

    # Bit position of the start of value i in the output bitstream.
    bit_positions = np.arange(n, dtype=np.int64) * bits  # (n,)

    for b in range(bits):
        # Extract the b-th bit of every unsigned value.
        bit_val = ((unsigned >> b) & 1).astype(np.uint8)  # (n,)

        # Target position in the output bitstream.
        target_bit = bit_positions + b  # (n,)
        target_byte = (target_bit >> 3).astype(np.intp)   # target_bit // 8
        target_shift = (target_bit & 7).astype(np.uint8)  # target_bit % 8

        # Scatter-add the bits into the output buffer.
        np.add.at(out, target_byte, (bit_val << target_shift).astype(np.uint8))

    return torch.from_numpy(out).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """Unpack a uint8 byte array back to signed N-bit integer codes.

    Args:
        packed:     uint8 tensor produced by :func:`pack_codes`.
        bits:       Same bit width used during packing.
        n_elements: Original number of elements.

    Returns:
        int8 tensor with ``n_elements`` values in [-2^(bits-1), 2^(bits-1)-1].
    """
    _validate_bits(bits)

    if bits == 8:
        return (packed[:n_elements].to(torch.int16) - _offset(8)).to(torch.int8)

    buf = packed.numpy()
    out = np.zeros(n_elements, dtype=np.uint16)

    bit_positions = np.arange(n_elements, dtype=np.int64) * bits  # (n,)

    for b in range(bits):
        target_bit = bit_positions + b
        src_byte = (target_bit >> 3).astype(np.intp)
        src_shift = (target_bit & 7).astype(np.uint8)

        bit_val = ((buf[src_byte] >> src_shift) & 1).astype(np.uint16)
        out |= bit_val << b

    # Convert back to signed.
    signed = out.astype(np.int16) - np.int16(_offset(bits))
    return torch.from_numpy(signed).to(torch.int8)


# ===================================================================
# Fast path: Q4  (nibble packing -- 2 values per byte)
# ===================================================================

def _pack_q4_fast(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(4)).to(torch.uint8)  # [0, 15]

    n = shifted.numel()
    if n % 2 != 0:
        shifted = torch.cat([shifted, torch.zeros(1, dtype=torch.uint8, device=shifted.device)])

    lo = shifted[0::2]  # even indices
    hi = shifted[1::2]  # odd indices
    return (hi << 4) | lo


def _unpack_q4_fast(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([lo, hi], dim=1).flatten()[:n_elements]
    return (interleaved - _offset(4)).to(torch.int8)


# ===================================================================
# Fast path: Q5  (8 values per 5 bytes, explicit bit layout)
# ===================================================================
#
# Layout for 8 five-bit unsigned values v0..v7:
#   byte0 = v0[4:0] | v1[2:0] << 5
#   byte1 = v1[4:3] | v2[4:0] << 2 | v3[0] << 7
#   byte2 = v3[4:1] | v4[3:0] << 4
#   byte3 = v4[4] | v5[4:0] << 1 | v6[1:0] << 6
#   byte4 = v6[4:2] | v7[4:0] << 3

def _pack_q5_fast(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(5)).to(torch.int16)  # [0, 31]

    n = shifted.numel()
    pad = (8 - n % 8) % 8
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.int16, device=shifted.device)])

    v = shifted.view(-1, 8)
    v0, v1, v2, v3, v4, v5, v6, v7 = [v[:, i] for i in range(8)]

    byte0 = (v0 | (v1 << 5)).to(torch.uint8)
    byte1 = ((v1 >> 3) | (v2 << 2) | (v3 << 7)).to(torch.uint8)
    byte2 = ((v3 >> 1) | (v4 << 4)).to(torch.uint8)
    byte3 = ((v4 >> 4) | (v5 << 1) | (v6 << 6)).to(torch.uint8)
    byte4 = ((v6 >> 2) | (v7 << 3)).to(torch.uint8)

    return torch.stack([byte0, byte1, byte2, byte3, byte4], dim=1).flatten()


def _unpack_q5_fast(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    p = packed.to(torch.int16)
    groups = p.view(-1, 5)
    b0, b1, b2, b3, b4 = [groups[:, i] for i in range(5)]

    mask5 = 0x1F

    v0 = b0 & mask5
    v1 = ((b0 >> 5) | (b1 << 3)) & mask5
    v2 = (b1 >> 2) & mask5
    v3 = ((b1 >> 7) | (b2 << 1)) & mask5
    v4 = ((b2 >> 4) | (b3 << 4)) & mask5
    v5 = (b3 >> 1) & mask5
    v6 = ((b3 >> 6) | (b4 << 2)) & mask5
    v7 = (b4 >> 3) & mask5

    interleaved = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten()[:n_elements]
    return (interleaved - _offset(5)).to(torch.int8)


# ===================================================================
# Fast dispatch
# ===================================================================

def pack_codes_fast(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Optimised packing for common bit widths (Q4, Q5).

    Falls back to the generic :func:`pack_codes` for other widths.

    Args:
        codes: int8/int16/int32 tensor of signed codes.
        bits:  Bit width in [2, 8].

    Returns:
        uint8 tensor with ceil(n * bits / 8) bytes.
    """
    _validate_bits(bits)
    if bits == 4:
        return _pack_q4_fast(codes)
    if bits == 5:
        return _pack_q5_fast(codes)
    return pack_codes(codes, bits)


def unpack_codes_fast(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """Optimised unpacking for common bit widths (Q4, Q5).

    Falls back to the generic :func:`unpack_codes` for other widths.

    Args:
        packed:     uint8 tensor from :func:`pack_codes_fast`.
        bits:       Same bit width used during packing.
        n_elements: Original number of elements.

    Returns:
        int8 tensor with ``n_elements`` values.
    """
    _validate_bits(bits)
    if bits == 4:
        return _unpack_q4_fast(packed, n_elements)
    if bits == 5:
        return _unpack_q5_fast(packed, n_elements)
    return unpack_codes(packed, bits, n_elements)


# ===================================================================
# GPU-accelerated unpack / dequant
# ===================================================================

def unpack_codes_gpu(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """GPU-accelerated unpacking. Works on CUDA tensors.

    For Q4: simple nibble unpacking with torch ops (no numpy)
    For other bits: bitwise operations on GPU

    Returns int8 tensor on same device as input.
    """
    device = packed.device
    offset = 1 << (bits - 1)

    if bits == 4:
        # Fast Q4: nibble unpack
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        result = torch.empty(packed.numel() * 2, dtype=torch.int8, device=device)
        result[0::2] = (low - offset).to(torch.int8)
        result[1::2] = (high - offset).to(torch.int8)
        return result[:n_elements]

    if bits == 8:
        return packed.to(torch.int8)[:n_elements]

    # General GPU path for Q2, Q3, Q5, Q6, Q7
    result = torch.zeros(n_elements, dtype=torch.int16, device=device)
    bit_positions = torch.arange(n_elements, dtype=torch.int64, device=device) * bits

    for b in range(bits):
        target_bit = bit_positions + b
        byte_idx = (target_bit >> 3).long()
        bit_idx = (target_bit & 7).to(torch.uint8)
        bit_vals = (packed[byte_idx] >> bit_idx) & 1
        result += bit_vals.to(torch.int16) << b

    return (result - offset).to(torch.int8)


def dequant_packed_gpu(
    packed: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    n_elements: int,
    shape: tuple,
    block_size: int = 128
) -> torch.Tensor:
    """Full GPU pipeline: unpack + dequantize in one function.

    packed: uint8 tensor on GPU
    scales: fp16 tensor on GPU
    Returns: fp16 tensor with original shape
    """
    # Unpack on GPU
    codes = unpack_codes_gpu(packed, bits, n_elements).float()

    # Dequantize
    n = codes.numel()
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        codes = F.pad(codes, (0, pad))
    blocks = codes.view(-1, block_size)
    result = (blocks * scales.float().unsqueeze(1)).flatten()[:n]
    return result.view(shape).half()


def benchmark_unpack_speed():
    """Compare CPU vs GPU unpack speed."""
    import time

    for bits in [3, 4, 5, 6]:
        n = 1_000_000
        qmax = (1 << (bits-1)) - 1
        codes_orig = torch.randint(-qmax, qmax+1, (n,), dtype=torch.int8)
        packed = pack_codes_fast(codes_orig, bits)

        # CPU
        t0 = time.perf_counter()
        for _ in range(10):
            cpu_result = unpack_codes_fast(packed, bits, n)
        cpu_time = (time.perf_counter() - t0) / 10

        # GPU (if available)
        if torch.cuda.is_available():
            packed_gpu = packed.cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                gpu_result = unpack_codes_gpu(packed_gpu, bits, n)
                torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - t0) / 10

            # Verify correctness
            assert torch.equal(cpu_result, gpu_result.cpu())

            print(f'Q{bits}: CPU {n/cpu_time/1e6:.1f} Melem/s | GPU {n/gpu_time/1e6:.1f} Melem/s | Speedup {cpu_time/gpu_time:.1f}x')
        else:
            print(f'Q{bits}: CPU {n/cpu_time/1e6:.1f} Melem/s | GPU: N/A')


# ===================================================================
# Self-tests and benchmarks
# ===================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    np.random.seed(42)

    PASS_COUNT = 0
    FAIL_COUNT = 0

    def check(condition: bool, label: str) -> None:
        global PASS_COUNT, FAIL_COUNT
        if condition:
            PASS_COUNT += 1
            print(f"  PASS  {label}")
        else:
            FAIL_COUNT += 1
            print(f"  FAIL  {label}")

    # ------------------------------------------------------------------
    # 1. Round-trip correctness for every bit width (2-8), generic path
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Round-trip tests: pack_codes / unpack_codes (generic)")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in [1, 2, 3, 7, 8, 9, 15, 16, 100, 1024]:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)
            packed = pack_codes(codes, bits)
            recovered = unpack_codes(packed, bits, n)

            expected_bytes = math.ceil(n * bits / 8)

            ok_values = torch.equal(codes, recovered)
            ok_size = packed.numel() == expected_bytes

            check(ok_values and ok_size,
                  f"Q{bits}  n={n:5d}  packed_bytes={packed.numel()} "
                  f"(expected {expected_bytes})  values_ok={ok_values}")

    # ------------------------------------------------------------------
    # 2. Round-trip for fast path (Q4, Q5)
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Round-trip tests: pack_codes_fast / unpack_codes_fast")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in [1, 2, 3, 7, 8, 9, 15, 16, 100, 1024]:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)
            packed = pack_codes_fast(codes, bits)
            recovered = unpack_codes_fast(packed, bits, n)

            ok = torch.equal(codes, recovered)
            check(ok, f"fast Q{bits}  n={n:5d}  ok={ok}")

    # ------------------------------------------------------------------
    # 3. Edge values: min, max, zero
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Edge-value tests")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        edge = torch.tensor([lo, hi, 0, lo, hi, 0, lo, hi], dtype=torch.int8)
        packed = pack_codes(edge, bits)
        recovered = unpack_codes(packed, bits, edge.numel())
        check(torch.equal(edge, recovered), f"Q{bits} edge values")

    # ------------------------------------------------------------------
    # 4. Single element
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Single-element tests")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for val in [lo, hi, 0]:
            codes = torch.tensor([val], dtype=torch.int8)
            packed = pack_codes(codes, bits)
            recovered = unpack_codes(packed, bits, 1)
            check(torch.equal(codes, recovered), f"Q{bits} single val={val}")

    # ------------------------------------------------------------------
    # 5. Odd number of elements
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Odd-element tests")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in [1, 3, 5, 7, 13, 99, 1023]:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)
            packed = pack_codes(codes, bits)
            recovered = unpack_codes(packed, bits, n)
            check(torch.equal(codes, recovered), f"Q{bits} odd n={n}")

    # ------------------------------------------------------------------
    # 6. Packed size verification
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Packed-size verification")
    print("=" * 72)

    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in [1, 8, 100, 1000]:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)
            packed = pack_codes(codes, bits)
            expected = math.ceil(n * bits / 8)
            check(packed.numel() == expected,
                  f"Q{bits} n={n}  packed_size={packed.numel()} expected={expected}")

    # ------------------------------------------------------------------
    # 7. Cross-check: generic vs fast path produce same results
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Cross-check: generic vs fast path")
    print("=" * 72)

    for bits in [4, 5]:
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in [1, 7, 8, 15, 16, 100, 1024]:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)

            packed_generic = pack_codes(codes, bits)
            recovered_generic = unpack_codes(packed_generic, bits, n)

            packed_fast = pack_codes_fast(codes, bits)
            recovered_fast = unpack_codes_fast(packed_fast, bits, n)

            # The packed bytes may differ in padding bits for the fast path
            # (which groups into fixed-size chunks), but the round-trip must match.
            check(torch.equal(recovered_generic, recovered_fast),
                  f"Q{bits} n={n}  generic_vs_fast values match")

    # ------------------------------------------------------------------
    # 8. Speed comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Speed benchmarks")
    print("=" * 72)

    NUM_WARMUP = 5
    NUM_TRIALS = 30

    def bench(fn, *args):
        for _ in range(NUM_WARMUP):
            fn(*args)
        times = []
        for _ in range(NUM_TRIALS):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[NUM_TRIALS // 2]  # median

    SIZES = [10_000, 100_000, 1_000_000]

    header = (f"  {'Bits':>4s}  {'N':>10s}  "
              f"{'Pack(ms)':>10s}  {'Unpack(ms)':>12s}  "
              f"{'FastPk(ms)':>11s}  {'FastUnpk(ms)':>13s}  "
              f"{'Speedup Pk':>10s}  {'Speedup Unpk':>12s}")
    print(header)
    print(f"  {'-' * len(header)}")

    for bits in [3, 4, 5, 6]:
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        for n in SIZES:
            codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)

            # Generic
            t_pack = bench(pack_codes, codes, bits)
            packed_g = pack_codes(codes, bits)
            t_unpack = bench(unpack_codes, packed_g, bits, n)

            # Fast
            t_pack_f = bench(pack_codes_fast, codes, bits)
            packed_f = pack_codes_fast(codes, bits)
            t_unpack_f = bench(unpack_codes_fast, packed_f, bits, n)

            sp_pack = t_pack / t_pack_f if t_pack_f > 0 else float("inf")
            sp_unpack = t_unpack / t_unpack_f if t_unpack_f > 0 else float("inf")

            print(f"  {bits:4d}  {n:10d}  "
                  f"{t_pack * 1000:10.3f}  {t_unpack * 1000:12.3f}  "
                  f"{t_pack_f * 1000:11.3f}  {t_unpack_f * 1000:13.3f}  "
                  f"{sp_pack:10.2f}x  {sp_unpack:12.2f}x")

    # Throughput summary
    print()
    print("Throughput summary (1M elements):")
    n = 1_000_000
    for bits in range(2, 9):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        codes = torch.randint(lo, hi + 1, (n,), dtype=torch.int8)

        t_pack = bench(pack_codes_fast, codes, bits)
        packed = pack_codes_fast(codes, bits)
        t_unpack = bench(unpack_codes_fast, packed, bits, n)

        pack_mps = n / t_pack / 1e6
        unpack_mps = n / t_unpack / 1e6
        storage_pct = bits / 8 * 100

        print(f"  Q{bits}:  pack {pack_mps:7.1f} Melem/s   "
              f"unpack {unpack_mps:7.1f} Melem/s   "
              f"storage {storage_pct:.1f}% of int8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Tests: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All tests PASSED.")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 72)
