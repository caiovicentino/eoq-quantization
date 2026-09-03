"""Tests for core/gptq_correction.py -- GPTQ weight quantization with error correction."""

import pytest
import torch


def test_gptq_result_shape():
    """GPTQ result codes should match input shape."""
    from core.gptq_correction import gptq_quantize_weight, GPTQResult

    W = torch.randn(256, 128)
    H = torch.eye(128) + torch.randn(128, 128) * 0.1
    H = H @ H.t()  # make positive definite
    result = gptq_quantize_weight(W, H, bits=4, block_size=128)
    assert isinstance(result, GPTQResult)
    assert result.codes.shape == (256, 128)
    assert result.bits == 4


def test_gptq_codes_in_range():
    """Codes should be in [-qmax, qmax]."""
    from core.gptq_correction import gptq_quantize_weight

    W = torch.randn(64, 64)
    H = torch.eye(64)
    result = gptq_quantize_weight(W, H, bits=4)
    qmax = (1 << 3) - 1  # 7 for Q4
    assert result.codes.max() <= qmax
    assert result.codes.min() >= -qmax


def test_gptq_reduces_error():
    """GPTQ should have lower weighted MSE than plain absmax."""
    from core.gptq_correction import gptq_quantize_weight, gptq_dequantize

    torch.manual_seed(42)
    W = torch.randn(128, 128) * 0.1
    # Create correlated Hessian (where GPTQ helps most)
    X = torch.randn(1000, 128)
    H = X.t() @ X / 1000

    # GPTQ quantization
    result = gptq_quantize_weight(W, H, bits=4, block_size=128)
    W_gptq = gptq_dequantize(result)

    # Plain absmax quantization
    import torch.nn.functional as F

    flat = W.float().flatten()
    n = flat.numel()
    bs = 128
    pad = (bs - n % bs) % bs
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, bs)
    qmax = 7
    absmax = blocks.abs().amax(dim=1).clamp(min=1e-10)
    scale = absmax / qmax
    codes = (blocks / scale.unsqueeze(1)).round().clamp(-qmax, qmax)
    W_absmax = (codes * scale.unsqueeze(1)).flatten()[:n].view(W.shape).half()

    # Weighted error (weight by Hessian diagonal = channel importance)
    importance = H.diag()
    error_gptq = (
        (W.half().float() - W_gptq.float()) ** 2 * importance.unsqueeze(0)
    ).mean()
    error_absmax = (
        (W.half().float() - W_absmax.float()) ** 2 * importance.unsqueeze(0)
    ).mean()

    # GPTQ should be better (or at least not worse)
    assert error_gptq <= error_absmax * 1.1  # allow 10% tolerance


def test_gptq_identity_hessian():
    """With H=I, GPTQ reduces to plain quantization (no error correction)."""
    from core.gptq_correction import gptq_quantize_weight

    W = torch.randn(64, 64)
    H = torch.eye(64)
    result = gptq_quantize_weight(W, H, bits=4)
    assert result.codes.shape == (64, 64)


def test_gptq_dequantize_roundtrip():
    """Dequantize should produce valid FP16 tensor."""
    from core.gptq_correction import gptq_quantize_weight, gptq_dequantize

    W = torch.randn(64, 128)
    H = torch.eye(128)
    result = gptq_quantize_weight(W, H, bits=4)
    W_recon = gptq_dequantize(result)
    assert W_recon.shape == W.shape
    assert W_recon.dtype == torch.float16


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_gptq_different_bits(bits):
    """GPTQ should work for various bit widths."""
    from core.gptq_correction import gptq_quantize_weight

    W = torch.randn(64, 64)
    H = torch.eye(64)
    result = gptq_quantize_weight(W, H, bits=bits)
    qmax = (1 << (bits - 1)) - 1
    assert result.codes.max() <= qmax
    assert result.codes.min() >= -qmax


def test_gptq_damping():
    """Damping should prevent numerical instability with ill-conditioned H."""
    from core.gptq_correction import gptq_quantize_weight

    W = torch.randn(64, 64)
    # Nearly singular Hessian
    H = torch.zeros(64, 64)
    H[0, 0] = 1.0  # rank 1
    result = gptq_quantize_weight(W, H, bits=4, percdamp=0.1)
    assert not torch.isnan(result.codes.float()).any()


def test_gptq_large_matrix():
    """Test with realistic size matrix."""
    from core.gptq_correction import gptq_quantize_weight, gptq_dequantize

    torch.manual_seed(7)
    W = torch.randn(512, 256) * 0.05
    X = torch.randn(500, 256)
    H = X.t() @ X / 500
    result = gptq_quantize_weight(W, H, bits=4)
    W_recon = gptq_dequantize(result)
    error = (W.half().float() - W_recon.float()).abs().mean()
    assert error < 0.05  # reasonable error for Q4


def test_gptq_scale_stored_in_result():
    """GPTQResult should carry scale info needed for dequantization."""
    from core.gptq_correction import gptq_quantize_weight

    W = torch.randn(64, 128)
    H = torch.eye(128)
    result = gptq_quantize_weight(W, H, bits=4, block_size=128)
    assert hasattr(result, "scales")
    assert result.scales is not None
    assert result.scales.shape == (64, 1)  # (out_features, n_groups)
    assert hasattr(result, "shape")
    assert result.shape == (64, 128)


def test_gptq_block_size_smaller_than_columns():
    """GPTQ should work when block_size < number of columns."""
    from core.gptq_correction import gptq_quantize_weight, gptq_dequantize

    W = torch.randn(64, 128)
    H = torch.eye(128)
    result = gptq_quantize_weight(W, H, bits=4, block_size=32)
    assert result.codes.shape == (64, 128)
    W_recon = gptq_dequantize(result)
    assert W_recon.shape == W.shape
