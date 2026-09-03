import pytest
import torch
import numpy as np

# Test round-trip for each bit width
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_roundtrip(bits):
    """Pack then unpack should return original codes."""
    from core.bit_packing import pack_codes, unpack_codes
    qmax = (1 << (bits - 1)) - 1
    n = 1000
    codes = torch.randint(-qmax-1, qmax+1, (n,), dtype=torch.int8)
    codes = codes.clamp(-qmax, qmax)  # ensure valid range
    packed = pack_codes(codes, bits)
    unpacked = unpack_codes(packed, bits, n)
    assert torch.equal(codes, unpacked)

def test_packed_size_q4():
    """Q4: 1000 codes should pack to 500 bytes."""
    from core.bit_packing import pack_codes
    codes = torch.zeros(1000, dtype=torch.int8)
    packed = pack_codes(codes, 4)
    assert packed.numel() == 500

def test_packed_size_q3():
    """Q3: 1000 codes should pack to ceil(1000*3/8) = 375 bytes."""
    from core.bit_packing import pack_codes
    codes = torch.zeros(1000, dtype=torch.int8)
    packed = pack_codes(codes, 3)
    assert packed.numel() == 375

def test_packed_size_q5():
    """Q5: 1000 codes should pack to 625 bytes."""
    from core.bit_packing import pack_codes
    codes = torch.zeros(1000, dtype=torch.int8)
    packed = pack_codes(codes, 5)
    assert packed.numel() == 625

def test_single_element():
    from core.bit_packing import pack_codes, unpack_codes
    for bits in [3, 4, 5]:
        qmax = (1 << (bits - 1)) - 1
        for value in [qmax, -qmax, 0]:
            codes = torch.tensor([value], dtype=torch.int8)
            packed = pack_codes(codes, bits)
            unpacked = unpack_codes(packed, bits, 1)
            assert codes[0] == unpacked[0]

def test_odd_elements():
    from core.bit_packing import pack_codes, unpack_codes
    codes = torch.tensor([1, -2, 3], dtype=torch.int8)
    packed = pack_codes(codes, 4)
    unpacked = unpack_codes(packed, 4, 3)
    assert torch.equal(codes, unpacked)

def test_extreme_values():
    """Test min/max values for each bit width."""
    from core.bit_packing import pack_codes, unpack_codes
    for bits in [2, 3, 4, 5, 6, 7, 8]:
        qmax = (1 << (bits - 1)) - 1
        codes = torch.tensor([-qmax, qmax, 0, -1, 1], dtype=torch.int8)
        packed = pack_codes(codes, bits)
        unpacked = unpack_codes(packed, bits, 5)
        assert torch.equal(codes, unpacked)

def test_large_tensor():
    """Test with realistic tensor size."""
    from core.bit_packing import pack_codes, unpack_codes
    n = 1_000_000
    codes = torch.randint(-7, 8, (n,), dtype=torch.int8)  # Q4 range
    packed = pack_codes(codes, 4)
    assert packed.numel() == n // 2
    unpacked = unpack_codes(packed, 4, n)
    assert torch.equal(codes, unpacked)

def test_compression_ratio():
    """Verify actual compression ratios."""
    from core.bit_packing import pack_codes
    n = 10000
    codes = torch.zeros(n, dtype=torch.int8)

    for bits, expected_ratio in [(3, 8/3), (4, 2.0), (5, 8/5), (6, 8/6)]:
        packed = pack_codes(codes, bits)
        actual_ratio = n / packed.numel()
        assert abs(actual_ratio - expected_ratio) < 0.01, f"bits={bits}: expected {expected_ratio}, got {actual_ratio}"

def test_dtype():
    """Packed output should be uint8."""
    from core.bit_packing import pack_codes, unpack_codes
    codes = torch.zeros(100, dtype=torch.int8)
    packed = pack_codes(codes, 4)
    assert packed.dtype == torch.uint8
    unpacked = unpack_codes(packed, 4, 100)
    assert unpacked.dtype == torch.int8
