"""Tests for core.kld_sensitivity — tensor classification and bit allocation."""

import pytest

from core.kld_sensitivity import classify_tensor


# ── Embedding ────────────────────────────────────────────────────────────

def test_classify_tensor_embedding():
    assert classify_tensor("model.embed_tokens.weight") == "embedding"


def test_classify_tensor_embedding_wte():
    assert classify_tensor("transformer.wte.weight") == "embedding"


# ── LM head ──────────────────────────────────────────────────────────────

def test_classify_tensor_lm_head():
    assert classify_tensor("lm_head.weight") == "lm_head"


def test_classify_tensor_lm_head_output():
    assert classify_tensor("output.weight") == "lm_head"


# ── Attention Q/K/V ──────────────────────────────────────────────────────

def test_classify_tensor_attn_qkv():
    assert classify_tensor("model.layers.0.self_attn.q_proj.weight") == "attn_qkv"
    assert classify_tensor("model.layers.0.self_attn.k_proj.weight") == "attn_qkv"
    assert classify_tensor("model.layers.0.self_attn.v_proj.weight") == "attn_qkv"


# ── Attention O ──────────────────────────────────────────────────────────

def test_classify_tensor_attn_o():
    assert classify_tensor("model.layers.0.self_attn.o_proj.weight") == "attn_o"


# ── MLP ──────────────────────────────────────────────────────────────────

def test_classify_tensor_mlp_gate_up():
    assert classify_tensor("model.layers.0.mlp.gate_proj.weight") == "mlp_gate_up"
    assert classify_tensor("model.layers.0.mlp.up_proj.weight") == "mlp_gate_up"


def test_classify_tensor_mlp_down():
    assert classify_tensor("model.layers.0.mlp.down_proj.weight") == "mlp_down"


# ── Norm ─────────────────────────────────────────────────────────────────

def test_classify_tensor_norm():
    assert classify_tensor("model.layers.0.input_layernorm.weight") == "norm"


def test_classify_tensor_norm_rms():
    assert classify_tensor("model.layers.5.post_attention_layernorm.weight") == "norm"


def test_classify_tensor_norm_final():
    assert classify_tensor("model.norm.weight") == "norm"


# ── SSM (Mamba / Jamba) ─────────────────────────────────────────────────
# The current implementation requires "mamba" in the name or an explicit
# \bssm\b / \.s6\. pattern for SSM classification.  Names that only use
# the generic "mixer" prefix (without "mamba") fall through to "other".

def test_classify_tensor_ssm_keyword():
    """Names containing '.mamba.' are classified as SSM."""
    assert classify_tensor("backbone.layers.0.mamba.A_log") == "ssm"
    assert classify_tensor("backbone.layers.0.mamba.D") == "ssm"
    assert classify_tensor("backbone.layers.0.mamba.dt_proj.weight") == "ssm"


def test_classify_tensor_ssm_block():
    """Names containing the literal 'ssm' token are classified as SSM."""
    assert classify_tensor("model.layers.0.ssm.weight") == "ssm"


@pytest.mark.xfail(
    reason="mixer-only names lack 'mamba' keyword; classify_tensor returns 'other'",
    strict=True,
)
def test_classify_tensor_ssm_mixer_names():
    """Spec: mixer-style names should eventually map to SSM."""
    assert classify_tensor("backbone.layers.0.mixer.A_log") == "ssm"
    assert classify_tensor("backbone.layers.0.mixer.D") == "ssm"
    assert classify_tensor("backbone.layers.0.mixer.dt_bias") == "ssm"


# ── Conv1d ───────────────────────────────────────────────────────────────

def test_classify_tensor_conv1d():
    assert classify_tensor("backbone.layers.0.mixer.conv1d.weight") == "conv1d"


# ── Other / fallback ────────────────────────────────────────────────────

def test_classify_tensor_other():
    assert classify_tensor("some.random.tensor.weight") == "other"


# ── get_bit_allocation ──────────────────────────────────────────────────

def _import_get_bit_allocation():
    """Try to import get_bit_allocation; return None if missing."""
    try:
        from core.kld_sensitivity import get_bit_allocation
        return get_bit_allocation
    except ImportError:
        return None


def test_get_bit_allocation_basic():
    get_bit_allocation = _import_get_bit_allocation()
    assert get_bit_allocation is not None, "get_bit_allocation not found"

    kld = {"a": 10.0, "b": 5.0, "c": 1.0, "d": 0.1}
    alloc = get_bit_allocation(kld, target_avg_bits=4.0)
    # Most sensitive tensor should get more bits
    assert alloc["a"] >= alloc["d"]
    # All allocations should be in a valid range
    assert all(2 <= v <= 16 for v in alloc.values())


def test_get_bit_allocation_target():
    get_bit_allocation = _import_get_bit_allocation()
    assert get_bit_allocation is not None, "get_bit_allocation not found"

    kld = {f"tensor_{i}": float(i) for i in range(100)}
    alloc = get_bit_allocation(kld, target_avg_bits=4.0)
    avg = sum(alloc.values()) / len(alloc)
    # Average should be close to target (within 1 bit)
    assert abs(avg - 4.0) < 1.0
