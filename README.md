# EOQ: Entropy-Optimal Quantization for LLMs

**PolarQuant** -- optimal Gaussian quantization via Walsh-Hadamard rotation + Lloyd-Max centroids. Achieves near-lossless compression (PPL 6.39 vs 6.37 FP16) with 2.75x VRAM reduction.

**arXiv preprint**: [arXiv:2603.7424577](https://arxiv.org/abs/2603.7424577)

---

## PolarQuant v5: Final Results

### Benchmark (Qwen3.5-9B, RTX PRO 6000 Blackwell)

| Method | tok/s | VRAM | PPL (WikiText-2) | Platform |
|--------|-------|------|-------------------|----------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 | RTX PRO 6000 Blackwell |
| **PolarQuant Q5 + torchao INT4** | **43.1** | **6.5 GB** | **6.56** | RTX PRO 6000 Blackwell |
| torchao INT4 (absmax) | 43.3 | 6.3 GB | 6.68 | RTX PRO 6000 Blackwell |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 | RTX PRO 6000 Blackwell |
| PolarQuant Q5 dequant FP16 | 45.9 | 18.1 GB | 6.39 | RTX PRO 6000 Blackwell |
| PolarQuant MLX Q4 | 19.7 | 4.8 GB | 6.90 | Mac mini M4 16 GB |

**PolarQuant Q5 + torchao INT4** is the recommended inference path: 43 tok/s at 6.5 GB VRAM with PPL 6.56 (+0.19 vs FP16). PolarQuant dequantized to FP16 achieves PPL 6.39 (+0.02), confirming near-lossless quantization quality.

### Ablation Study (Q5, Qwen3.5-9B)

| Configuration | PPL | Delta vs FP16 |
|---------------|-----|---------------|
| Absmax Q5 (baseline) | 6.9030 | +0.53 |
| + Hadamard rotation | 6.4010 | +0.03 |
| + Lloyd-Max centroids | 6.9139 | +0.54 |
| + Both (PolarQuant Q5) | 6.3909 | +0.02 |

**Hadamard rotation accounts for 98% of the improvement.** The Walsh-Hadamard transform makes weight distributions approximately Gaussian, enabling near-optimal uniform quantization. Lloyd-Max centroids provide a small additional gain when combined with rotation, but are ineffective alone.

### Apple Silicon (MLX)

PolarQuant runs natively on Apple Silicon via MLX with 4-bit quantization:

| Platform | tok/s | VRAM | PPL |
|----------|-------|------|-----|
| Mac mini M4 16 GB | 19.7 | 4.8 GB | 6.90 |

Model: [caiovicentino1/Qwen3.5-9B-PolarQuant-MLX-4bit](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-MLX-4bit)

---

## How PolarQuant Works

```
FP16 weights --> Normalize (L2) --> Hadamard Rotate --> Lloyd-Max Quantize --> INT codes + norms
                                    (Gaussian dist)    (optimal centroids)
```

1. **Block normalization**: divide weight blocks by L2 norm
2. **Walsh-Hadamard rotation**: transforms weights to approximate N(0,1) distribution (this is 98% of the quality gain)
3. **Lloyd-Max quantization**: optimal centroids for Gaussian distribution (small additional gain)
4. **Compact storage**: INT codes (1 byte per weight) + FP16 per-block norms

For inference, two paths are supported:
- **Dequant + torchao**: Dequantize to FP16, apply torchao INT4 -- 43 tok/s, 6.5 GB VRAM, PPL 6.56 (recommended)
- **PolarEngine Triton**: Custom kernel keeps weights quantized -- 34 tok/s, 7.8 GB VRAM, PPL 6.89

---

## EOQ Compression History

The project evolved through several generations:

### EOQ v1-v3 Results (Qwen3.5-9B)

| Version | Technique | PPL | Delta |
|---------|-----------|-----|-------|
| v1 | Absmax + rANS entropy coding | 7.26 | +0.89 |
| v2 | + AWQ activation-aware scaling | 7.05 | +0.68 |
| v3 | + PolarQuant (Hadamard + Lloyd-Max) | 6.43 | +0.06 |
| **v5** | **PolarQuant Q5 dequant (final)** | **6.39** | **+0.02** |

### Multi-Model Results (EOQ Q5, RTX PRO 6000 Blackwell)

| Model | FP16 Size | EOQ Q5 Size | PPL FP16 | PPL Q5 | Delta | tok/s |
|-------|-----------|-------------|----------|--------|-------|-------|
| Qwen2.5-0.5B | 988 MB | 279 MB | 10.87 | 11.69 | +0.83 | 145.0 |
| Qwen2.5-3B | 6,172 MB | 1,724 MB | 6.54 | 6.77 | +0.23 | 97.1 |
| Qwen3.5-4B | 8,412 MB | 2,398 MB | 7.58 | 7.77 | +0.18 | 54.1 |
| Qwen3.5-9B | 17,908 MB | 9,100 MB | 6.37 | 7.16 | +0.79 | 46.0 |
| Qwen3.5-27B | 53,792 MB | 15,353 MB | 5.65 | 5.64 | -0.01 | 6.2 |
| Qwen3.5-35B-A3B | 69,321 MB | 19,680 MB | 5.19 | 5.39 | +0.20 | 30.2 |

---

## EOQ Dynamic (Mixed-Bit)

Inspired by [Unsloth Dynamic 2.0](https://unsloth.ai), EOQ Dynamic assigns different bit widths per tensor based on quantization sensitivity:

| Tensor Type | Bits | Why |
|-------------|------|-----|
| MLP gate/up projections | 3 | Most robust to quantization |
| MLP down projection | 4 | Slightly more sensitive |
| Attention Q/K/V | 5 | Moderate sensitivity |
| Attention O/output proj | 6 | High sensitivity (no AWQ fix) |
| Embedding | 5 | Safe |
| LM head | 6 | Nearly lossless |
| Norms, biases, routing | FP16 | Must stay full precision |
| SSM tensors (Mamba) | FP16 | Catastrophic if quantized |

### Verified Results (Qwen3.5-9B)

| Format | Download | PPL | Delta | Compression |
|--------|----------|-----|-------|-------------|
| FP16 | 17.9 GB | 6.37 | --- | 1.0x |
| EOQ Q5 int8 | 9.1 GB | 7.09 | +0.72 | 2.0x |
| EOQ Dynamic BitPacked (v1) | 4.93 GB | 7.26 | +0.89 | 3.64x |
| EOQ Dynamic + AWQ (v2) | ~5 GB | 7.05 | +0.68 | 3.58x |

---

## PolarEngine v4: Quantized Inference

Custom Triton kernel that keeps weights quantized in GPU VRAM -- no dequantization to FP16 needed.

### Benchmark (Qwen3.5-9B, RTX PRO 6000 Blackwell)

| Method | tok/s | VRAM | PPL | Notes |
|--------|-------|------|-----|-------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 | Reference |
| **PolarQuant Q5 + torchao INT4** | **43.1** | **6.5 GB** | **6.56** | **Recommended** |
| torchao INT4 | 43.3 | 6.3 GB | 6.68 | Best speed/VRAM ratio |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 | |
| PolarEngine v4 | 34.2 | 7.8 GB | 6.89 | Custom Triton kernel |
| PolarEngine (package) | 33.7 | 7.8 GB | 6.89 | pip-installable plugin |

### Key Optimizations
- **Matmul FWHT**: 25x faster Walsh-Hadamard via `torch.matmul(x, H128)` (1 kernel vs 29)
- **FWHT cache**: Q/K/V projections reuse same result (69x total speedup)
- **Pre-scaled centroids**: Lloyd-Max centroids x 1/sqrt(block_size) baked into lookup table
- **INT4 nibble packing**: Half-block order, halves VRAM for Q3/Q4 layers (36% savings)
- **Triton tiled GEMV**: Fused centroid lookup + dot product, autotuned per layer shape

### vLLM Plugin

Install and use as a vLLM quantization method:

```bash
pip install git+https://github.com/caiovicentino/polarengine-vllm.git
```

```python
# Quantize
python -m polarengine_vllm.quantize --model Qwen/Qwen3.5-9B --output ./polar-9b/

# Serve (when vLLM supports the plugin)
vllm serve ./polar-9b/ --quantization polarengine
```

See [polarengine-vllm](https://github.com/caiovicentino/polarengine-vllm) for full documentation.

---

## Inference: Recommended Path

The recommended inference path is **PolarQuant Q5 + torchao INT4**:

```
Download PolarQuant Q5 (9.1 GB) --> Dequant to FP16 --> torchao INT4 --> 43 tok/s, 6.5 GB VRAM, PPL 6.56
```

### GPU-Accelerated Loading

EOQ loads **5x faster than FP16** thanks to GPU-accelerated dequantization:

| Step | PolarQuant | FP16 |
|------|------------|------|
| Download | 9.1 GB | 17.9 GB |
| Dequant | 4s (GPU) | 0s |
| torchao INT4 | ~3s | ~3s |

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int4WeightOnlyConfig
import torch

# Load PolarQuant Q5 codes (auto-dequantizes to FP16)
model = AutoModelForCausalLM.from_pretrained(
    "caiovicentino1/Qwen3.5-9B-PolarQuant-Q5",
    dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("caiovicentino1/Qwen3.5-9B-PolarQuant-Q5")

# Apply torchao INT4 for fast inference
quantize_(model, Int4WeightOnlyConfig(group_size=128))

# Generate
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Published Models

| Model | HuggingFace | Notes |
|-------|-------------|-------|
| **Qwen3.5-9B PolarQuant Q5** | [PolarQuant-Q5](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-Q5) | **Recommended** -- 9.1 GB codes, PPL 6.39/6.56 |
| **Qwen3.5-9B PolarQuant MLX 4-bit** | [PolarQuant-MLX-4bit](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-MLX-4bit) | Apple Silicon, 4.8 GB, PPL 6.90 |
| Qwen3.5-9B PolarEngine v4 | [PolarEngine-v4](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarEngine-v4) | Custom Triton kernel, 7.8 GB VRAM |
| Qwen3.5-9B EOQ v3 (PolarQuant+AWQ) | [v3](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-v3) | Legacy, PPL 6.43 |
| Qwen3.5-9B EOQ v2 (AWQ) | [v2](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-v2) | Legacy |
| Qwen3.5-9B EOQ Dynamic BitPacked | [Dynamic](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-Dynamic-BitPacked) | Legacy |
| Qwen2.5-0.5B EOQ Q4/Q5/Q6/Q8 | [Q4](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q6) [Q8](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q8) | |
| Qwen2.5-3B EOQ Q4/Q5/Q6 | [Q4](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q6) | |
| Qwen3.5-4B EOQ Q4/Q5/Q6 | [Q4](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q6) | |
| Qwen3.5-9B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-Q5-compressed) | |
| Qwen3.5-27B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-EOQ-Q5-compressed) | |
| Qwen3.5-35B-A3B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-35B-A3B-EOQ-Q5-compressed) | |
| GLM-4.7-Flash EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-EOQ-Q5-compressed) | |

---

## Experimental Findings

Eight experiments (A-H) were conducted to explore compression techniques:

| Technique | Result | Status |
|-----------|--------|--------|
| Delta coding (raw weights) | Adjacent layers have ~0 cosine similarity | Disproven |
| DCT/Wavelet 2D | Weight matrices lack spatial correlation | Disproven |
| SVD+Q above 3-bit | Direct quantization wins at Q4+ | Disproven |
| **Entropy coding** | **10-18% additional savings via rANS** | Confirmed |
| **SVD+Q sub-3-bit** | **100% win rate at Q2** | Confirmed |
| **Absmax + entropy = competitive with K-quants** | **Same quality, smaller size** | Confirmed |

---

## Project Structure

```
dct-quantization/
├── core/              # EOQ pipeline: rANS, quantization, format
│   ├── rans.py        # rANS encoder/decoder (19/19 tests)
│   ├── rans_blocked.py # Blocked rANS with random access (8/8 tests)
│   ├── eoq.py         # EOQ compression pipeline
│   ├── eoq_format.py  # .eoq file format (23/23 tests)
│   ├── svd_hybrid.py  # W = Q + LR for sub-2.5 bpw
│   ├── quantized_linear.py  # INT4 packed nn.Module (47/47 tests)
│   └── weight_loader.py     # Universal HuggingFace loader
├── kernels/           # 16 CUDA kernel variants + PolarEngine
│   ├── k01-k16        # Shared memory, half2, uint4, multi-row, etc.
│   ├── k05_combined.py # Ultimate kernel (all optimizations)
│   └── polar_engine.py # PolarQuant inference kernel (Triton + torch)
├── llamacpp/          # C rANS decoder + GGUF converter
│   ├── eoq_rans.c     # C decoder (31/31 tests, 120 MB/s)
│   ├── eoq_ggml.h     # GGML integration header
│   └── eoq_convert.py # GGUF compression tool
├── llamacpp_integration/  # Full llama.cpp integration
│   ├── patches/       # 5 patches for llama.cpp
│   ├── eoq_rans_v2.c  # Optimized decoder (114 MB/s, 83/83 tests)
│   ├── eoq_cuda_decompress.cu  # GPU decompression
│   └── PR_DESCRIPTION.md
├── experiments/       # 8 validated experiments (A-H) + EOQ sweep
├── benchmarks/        # Perplexity, speed, memory profiling
├── tools/             # CLI: compress, decompress, chat server
├── notebooks/         # Colab notebooks
├── research/          # 19 literature review documents
└── tests/             # 200+ tests across all modules
```

## llama.cpp Integration

EOQ can serve as a transport compression layer for GGUF files:
- Decode rANS at load time, use existing Q4_K kernels at runtime
- Zero runtime overhead
- 5 patches ready, C decoder (114 MB/s), CUDA decompressor
- See `llamacpp_integration/PR_DESCRIPTION.md`

## Citation

```bibtex
@article{vicentino2026polarquant,
    title={PolarQuant: Near-Lossless LLM Quantization via Walsh-Hadamard Rotation
           and Entropy-Optimal Coding},
    author={Vicentino, Caio},
    journal={arXiv preprint arXiv:2603.7424577},
    year={2026}
}
```

## Requirements

Python 3.10+, PyTorch 2.0+, transformers, numpy, scipy

```bash
pip install -r requirements.txt
```

## License

Apache 2.0
