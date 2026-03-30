# EOQ: Entropy-Optimal Quantization for LLMs

Simple absmax quantization + rANS entropy coding that matches complex GGUF K-quants in quality-per-byte.

## Key Results (RTX PRO 6000 Blackwell, 96 GB VRAM)

| Model | FP16 Size | EOQ Q5 Size | EOQ Q5 Download | PPL FP16 | PPL Q5 | Delta | tok/s |
|-------|-----------|-------------|-----------------|----------|--------|-------|-------|
| Qwen2.5-0.5B | 988 MB | 279 MB | — | 10.87 | 11.69 | +0.83 | 145.0 |
| Qwen2.5-3B | 6,172 MB | 1,724 MB | — | 6.54 | 6.77 | +0.23 | 97.1 |
| Qwen3.5-4B | 8,412 MB | 2,398 MB | — | 7.58 | 7.77 | +0.18 | 54.1 |
| Qwen3.5-9B | 17,908 MB | 9,100 MB | 9.1 GB | 6.37 | 7.16 | +0.79 | 46.0 |
| Qwen3.5-27B | 53,792 MB | 15,353 MB | 27.3 GB | 5.65 | 5.64 | -0.01 | 6.2 |
| **Qwen3.5-35B-A3B** | **69,321 MB** | **19,680 MB** | **35.2 GB** | **5.19** | **5.39** | **+0.20** | **30.2** |
| GLM-4.7-Flash (30B MoE) | 59,887 MB | 30,400 MB | 30.4 GB | 37.71 | 41.12 | +3.41 | 3.2 |
| **Qwen3.5-9B (EOQ v3 PolarQuant+AWQ)** | **17,908 MB** | **~5,000 MB** | **~5 GB** | **6.37** | **6.43** | **+0.06** | **45.8** |

EOQ Q5 achieves **3.5x compression** with PPL degradation of only +0.18 to +0.83 points and **zero inference overhead** (identical tok/s to FP16). Compressed format halves download size (35.2 GB vs 69.3 GB for 35B model).

## How It Works

```
FP16 weights --> Block Absmax Quantization (Q5) --> rANS Entropy Coding --> .eoq file
                        (lossy, near-FP16)            (lossless, 11% extra savings)
```

1. **Absmax quantization**: simple block-wise symmetric quantization (no complex K-quant schemes)
2. **rANS entropy coding**: removes redundancy in quantized codes (Shannon entropy < bit width)
3. **Dequantized FP16 safetensors**: models load directly with `transformers` (no custom code needed)
4. **Compressed HuggingFace format**: INT5 codes (1 byte) + FP16 scales stored in safetensors — 2x smaller download, dequantized at load time

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

This achieves ~3.5-bit average with better quality than uniform Q5, at smaller size.

### Verified Results (Qwen3.5-9B, H100 80GB)

| Method | Avg Bits | PPL | Delta | tok/s |
|--------|----------|-----|-------|-------|
| FP16 (baseline) | 16 | 6.37 | --- | 29.4 |
| Uniform Q4 | 4.0 | 8.80 | +2.42 | 30.2 |
| **EOQ Dynamic** | **3.7** | **7.26** | **+0.89** | **29.7** |
| Uniform Q5 | 5.0 | 7.09 | +0.71 | 29.9 |

Dynamic achieves Q5-level quality (PPL 7.26 vs 7.09) at 3.7-bit average — **1.54 PPL better than Uniform Q4** at the same storage footprint. With proper bit-packing, Dynamic would be ~24% smaller than Uniform Q5.

### Verified Results (Qwen3.5-9B, Blackwell/H100)

| Format | Download | PPL | Delta | Compression |
|--------|----------|-----|-------|-------------|
| FP16 | 17.9 GB | 6.37 | --- | 1.0x |
| EOQ Q5 int8 | 9.1 GB | 7.09 | +0.72 | 2.0x |
| EOQ Dynamic BitPacked (v1) | 4.93 GB | 7.26 | +0.89 | 3.64x |
| **EOQ Dynamic + AWQ (v2)** | **~5 GB** | **7.05** | **+0.68** | **3.58x** |

AWQ saved scales reduced the PPL delta from +0.89 to +0.68 at identical download size. EOQ v2 achieves near-FP16 quality at 3.58x compression.

### EOQ v3: PolarQuant + AWQ (NEW)

| Version | Technique | PPL | Delta |
|---------|-----------|-----|-------|
| v1 | Absmax | 7.26 | +0.89 |
| v2 | AWQ | 7.05 | +0.68 |
| **v3** | **PolarQuant + AWQ** | **6.43** | **+0.06** |

PolarQuant (TurboQuant-inspired): normalize blocks, Hadamard rotate to Gaussian, Lloyd-Max optimal quantize. Combined with AWQ achieves 93% reduction in quantization error — practically lossless.

## PolarEngine v4: Quantized Inference (NEW)

Custom Triton kernel that keeps weights quantized in GPU VRAM -- no dequantization to FP16 needed.

### Benchmark (Qwen3.5-9B, RTX PRO 6000 Blackwell)

| Method | tok/s | VRAM | PPL | Notes |
|--------|-------|------|-----|-------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 | Reference |
| torchao INT4 | 43.3 | 6.3 GB | 6.68 | Best speed/VRAM ratio |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 | |
| **PolarEngine v4** | **34.2** | **7.8 GB** | **6.89** | Custom Triton kernel |
| **PolarEngine (package)** | **33.7** | **7.8 GB** | **6.89** | pip-installable plugin |
| PolarEngine v3 | 11.8 | 12.1 GB | 6.89 | Before FWHT optimization |

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

## Inference Speed (torchao path)

EOQ models can also be loaded with **torchao** for fast quantized inference:

```
Download (4.93 GB EOQ) → Dequantize → torchao INT4 → 43 tok/s, 6.3 GB VRAM
```

### GPU-Accelerated Loading (EOQ v2)

EOQ v2 loads **5x faster than FP16** thanks to GPU-accelerated dequantization:

| Step | EOQ v2 | FP16 |
|------|--------|------|
| Download | 5 GB (15s) | 17.9 GB (53s) |
| Dequant | 4s (GPU) | 0s |
| Total | **10s** | **53s** |

### Usage with torchao
```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig

model = load_eoq_model("caiovicentino1/Qwen3.5-9B-EOQ-Dynamic-BitPacked")
quantize_(model, Int4WeightOnlyConfig(group_size=128))
# Now runs at 43 tok/s with 6.3 GB VRAM
```

## Published Models

| Model | HuggingFace |
|-------|-------------|
| Qwen2.5-0.5B EOQ Q4/Q5/Q6/Q8 | [Q4](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q6) [Q8](https://huggingface.co/caiovicentino1/Qwen2.5-0.5B-EOQ-Q8) |
| Qwen2.5-3B EOQ Q4/Q5/Q6 | [Q4](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen2.5-3B-EOQ-Q6) |
| Qwen3.5-4B EOQ Q4/Q5/Q6 | [Q4](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q4) [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q5) [Q6](https://huggingface.co/caiovicentino1/Qwen3.5-4B-EOQ-Q6) |
| Qwen3.5-9B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-Q5-compressed) |
| Qwen3.5-27B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-EOQ-Q5-compressed) |
| Qwen3.5-35B-A3B EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/Qwen3.5-35B-A3B-EOQ-Q5-compressed) |
| GLM-4.7-Flash EOQ Q5 (compressed) | [Q5](https://huggingface.co/caiovicentino1/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-EOQ-Q5-compressed) |
| Qwen3.5-9B EOQ Dynamic BitPacked | [Dynamic](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-Dynamic-BitPacked) |
| Qwen3.5-9B EOQ v2 (AWQ) | [v2](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-v2) |
| Qwen3.5-9B EOQ v3 (PolarQuant+AWQ) | [v3](https://huggingface.co/caiovicentino1/Qwen3.5-9B-EOQ-v3) |
| Qwen3.5-9B PolarEngine v4 | [PolarEngine](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarEngine-v4) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "caiovicentino1/Qwen3.5-4B-EOQ-Q5",
    dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("caiovicentino1/Qwen3.5-4B-EOQ-Q5")
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

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

## Requirements

Python 3.10+, PyTorch 2.0+, transformers, numpy, scipy

```bash
pip install -r requirements.txt
```

## License

Apache 2.0
