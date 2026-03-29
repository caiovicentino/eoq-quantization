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

## Inference Speed

EOQ models are stored as dequantized FP16 safetensors. Inference speed is **identical to FP16** (no quantized kernels). The advantage is **smaller file size** at comparable quality, not speed.

For speed improvement with reduced RAM, we developed custom CUDA INT4 kernels:
- RAM: 2.8x less than FP16 (2,214 MB vs 6,172 MB for Qwen2.5-3B)
- Speed: 0.76x of FP16 (75.5 vs 98.7 tok/s) -- kernel optimization ongoing
- See `kernels/` directory for 16 CUDA kernel variants

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
├── kernels/           # 16 CUDA kernel optimization variants
│   ├── k01-k16        # Shared memory, half2, uint4, multi-row, etc.
│   └── k05_combined.py # Ultimate kernel (all optimizations)
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
