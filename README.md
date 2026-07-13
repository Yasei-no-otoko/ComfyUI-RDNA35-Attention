# ComfyUI RDNA35 Attention Research

![ComfyUI RDNA35 Attention](assets/rdna35-attention.png)

**English documentation (canonical).**

[日本語版](README_JA.md)

ComfyUI custom nodes for fixed 64-token block-diagonal self-attention on PyTorch ROCm with an optional Triton forward kernel. This is not a port of NVIDIA Blackwell TLX code.

The package also contains two isolated gfx1151 research paths. Neither replaces normal ComfyUI attention globally:

- Exact full attention with an online-softmax Triton kernel for `[BH,Q,D] x [BH,K,D]`.
- A training-free PISA HYD path using a compiled CK Tile statistics wheel, FlexAttention, and WMMA correction.

## What This Implements

The operation splits the sequence into fixed blocks of 64 tokens. Tokens in block `i` attend only to keys and values from block `i`. This is exact for fixed block-diagonal attention, but it is not equivalent to normal full attention because cross-block attention is removed.

The Triton path is forward/inference only. There is no custom autograd or backward path.

## Nodes

- `RDNA35 Block Attention Diagnostics`: reports PyTorch, HIP, device, best-effort gfx target, Triton availability, and RDNA3.5 detection.
- `RDNA35 Patch Model Attention`: installs a model-local `optimized_attention_override` on a cloned MODEL. It never globally monkey-patches ComfyUI attention.
- `RDNA35 Patch PISA Attention`: installs a model-local PISA override. Anima keeps its validated `T=9216,D=128` spatial path. Explicitly marked SD1.5, SDXL, Wan, and LTX self-attention with `T>=8192` uses the generic path; cross-attention, masks, short sequences, and unsupported devices chain to the previous ComfyUI backend.
- `RDNA35 Fixed Block Attention Benchmark`: creates synthetic Q/K/V tensors, compares reference, dispatch, PyTorch SDPA with a block-diagonal mask, and normal PyTorch full SDPA. Full SDPA is reported as a semantic contrast, not as an exact replacement.
- `RDNA35 Exact Full Attention Benchmark`: compares the gfx1151 online-softmax kernel with PyTorch SDPA for Anima-like self- and cross-attention shapes.
- `RDNA35 PISA Attention Benchmark`: separates first-use compile time from GPU-event steady-state time and compares the CK/Flex hybrid with dense SDPA.
- `RDNA35 Generic PISA Benchmark`: compares generic PISA, ComfyUI PyTorch SDPA, Flash Attention, and the optional SageAttention ROCm7 backend on one synthetic input. SageAttention supports only its advertised head dimensions; unsupported shapes are reported without aborting the other measurements.

## Measured gfx1151 results

The production comparison uses `rdna35-pisa-ck` 0.7.0 API 5 on the local PyTorch 2.14 ROCm 7.15 stack. The attention measurement is BF16 `B=2,H=16,T=9216,D=128` with the validated 23/144 exact-block profile:

| Spatial self-attention backend | Complete call | Relative speed | Time reduction |
|---|---:|---:|---:|
| ComfyUI Flash Attention | 46.411 ms | 1.000x | baseline |
| RDNA35 PISA CK/Flex fused | **25.121 ms** | **1.848x faster** | **21.290 ms (45.9%)** |

The end-to-end comparison uses Anima INT8_ConvRot at 1536x1536, Spectrum 30 steps, CFG 5, Euler/simple, batch 1, and 17 actual model forwards. The positive and negative prompts exactly match the metadata embedded in the official [Anima `example.png`](https://huggingface.co/circlestone-labs/Anima/blob/main/example.png). `mod_w_profile=off`, so Spectrum does not add another quality-conditioning pass.

| Selected ComfyUI backend | PISA patch | Seed 856853657535148 | Seed 856853657535149 | Prompt E2E median | Difference from Flash |
|---|---:|---:|---:|---:|---:|
| Flash Attention | on | 64.337 s | 66.217 s | **65.277 s** | **-11.939 s (-15.5%)** |
| Flash Attention | off | 71.816 s | 82.616 s | 77.216 s | baseline |
| SageAttention ROCm7 1.0.6 | on | 78.609 s | 80.541 s | 79.575 s | +2.359 s (+3.1%) |
| PyTorch SDPA | on | 98.163 s | 101.673 s | 99.918 s | +22.702 s (+29.4%) |
| SageAttention ROCm7 1.0.6 | off | 149.090 s | 151.760 s | 150.425 s | +73.209 s (+94.8%) |
| PyTorch SDPA | off | 302.027 s | 302.158 s | 302.093 s | +224.877 s (+291.2%) |

Each backend was measured in a resident ComfyUI process after an excluded warm-up, and Prompt E2E includes sampling, VAE decode, and preview output. Flash/PISA used ABBA order in one process; SDPA and Sage each used a dedicated process and measured the unpatched and patched paths after their respective warm-ups. The first SDPA and Sage runs took 719.036 s and 490.612 s because they included backend-specific Inductor compilation and are not included in the steady-state table.

The runtime verifier recorded `24/24` eligible self-attention calls, zero cross-attention calls, zero fallbacks, and no runtime failure for both PISA + SDPA and PISA + Sage in a one-forward validation run. The verifier itself triggers per-layer Dynamo recompilation and is therefore excluded from the performance table. The reproducible workflow is [workflows/Anima_INT8_ConvRot_PISA_E2E_Benchmark.json](workflows/Anima_INT8_ConvRot_PISA_E2E_Benchmark.json); disable its PISA node for the selected backend baseline. The native Q/K/V spatial pack is included in the complete attention call. The fused gfx1151 epilogue combines LSE weighting, FP32 correction, BF16 conversion, and block-major-to-raster output, while the spatial exact FlexAttention tile uses `BLOCK_N=32`.

### BF16 isolation run without INT8 ConvRot, model compile, or Spectrum

To separate the quality regression from the optimized workflow, `anima_aestheticV10.safetensors` was run with the standard KSampler and no model compile node. Resolution, prompts, seeds, 30 steps, CFG 5, and Euler/simple were kept unchanged. These are diagnostic process-order results, not warm steady-state release numbers:

| Selected backend | PISA | Seed 856853657535148 | Seed 856853657535149 |
|---|---:|---:|---:|
| Flash Attention | off | 147.178 s | 130.267 s |
| Flash Attention | on | 122.257 s | 126.271 s |
| PyTorch SDPA | off | 709.650 s | 567.443 s |
| PyTorch SDPA | on | 184.900 s | 184.358 s |
| SageAttention ROCm7 1.0.6 | off | 289.465 s | 267.070 s |
| SageAttention ROCm7 1.0.6 | on | 148.298 s | 146.293 s |

The pure Flash, SDPA, and Sage images retained the same composition and did not reproduce the severe noise or extra limb. All three PISA combinations produced a similar large composition shift and body/sign intersection, so the remaining geometry regression follows the PISA approximation rather than the fallback backend. INT8 ConvRot, Spectrum, or model compilation may amplify the earlier failures, but they are not required for PISA to change the composition.

The progress bar is not a valid GPU timing source on these asynchronous paths. Pure Flash reported about 2 seconds for 30 steps while Prompt E2E was 128.80-146.06 seconds; pure SDPA similarly reported about 2 seconds while E2E was 566.64-708 seconds. PISA made the step loop blocking enough that its 119-181 second progress time closely tracked E2E. Use Prompt E2E or GPU events after synchronization, not tqdm iteration speed.

The reproducible pure-BF16 workflow is [workflows/Anima_BF16_PISA_Pure_E2E_Benchmark.json](workflows/Anima_BF16_PISA_Pure_E2E_Benchmark.json). Both benchmark workflows include a disabled `SaveImage` node for matched-seed quality runs; enable it and set a backend-specific prefix when persistence is required. It remains disabled by default so disk I/O is excluded from timing.

## TODO

- Quantify the PISA composition and geometry regression before recommending any PISA + fallback combination. The BF16 isolation run removed INT8 ConvRot, model compile, and Spectrum: severe noise and the extra arm did not reproduce, but body/sign intersections followed PISA with Flash, SDPA, and Sage. Expand the matched-seed matrix and compare perceptual metrics plus per-layer attention error. PISA remains experimental until this quality gate passes.

The generic benchmark below uses identical synthetic Q/K/V tensors and GPU-event medians after warm-up. Sage is [SageAttention ROCm7 1.0.6](https://github.com/guinmoon/SageAttention-Rocm7/releases/tag/v1.0.6_rocm7). These are kernel-level measurements, not end-to-end image quality results:

| Shape | Generic PISA | PyTorch SDPA | Flash | SageAttention ROCm7 |
|---|---:|---:|---:|---:|
| FP16 SD1.5-like `B=1,H=8,T=8192,D=40` | 4.394 ms | 5.791 ms | **3.896 ms** | unsupported D=40 |
| FP16 `B=1,H=8,T=8192,D=64` | **4.912 ms** | 5.278 ms | 5.296 ms | 5.689 ms |
| BF16 `B=1,H=4,T=16384,D=128` | **11.588 ms** | 19.870 ms | 16.993 ms | 21.638 ms |

Generic PISA is intentionally approximate; its cosine similarity against dense SDPA on the three random inputs was 0.701518, 0.713715, and 0.805782. Kernel speed must therefore be paired with model-level quality validation before enabling a profile.

An LTX 2.3 image-to-video run at 768x768, 73 frames, and batch 2 completed normally but did not execute PISA. Runtime accounting observed 768 self-attention and 1536 cross-attention calls; the relevant self-attention sequence was `T=1440`, below the generic `T>=8192` threshold, while text attention used `T=77`. Those calls correctly remained on the existing ComfyUI backend. This is compatibility evidence, not an LTX PISA speed result.

For unfiltered LTX video, ComfyUI now passes the patchified `F,H,W` token grid to the model-local attention override. Filtered guide tokens, guide masks, and mixed audio/video streams do not claim a spatial grid and remain on their safe fallback paths. The local LTX 2.3 checkpoint declares `H=32,D=128`; a synthetic BF16 `B=1,T=8192` check on a `2x64x64` video grid measured 49.867 ms for HYD PISA, 492.331 ms for ComfyUI PyTorch SDPA, 40.981 ms for Flash, and 42.695 ms for SageAttention ROCm7. All outputs were finite; PISA cosine similarity against SDPA was 0.813123. PISA is substantially faster than the measured SDPA path at this large shape, but Flash remained 21.7% faster.

Wan self-attention now uses `transformer_options["grid_sizes"]` to group each frame into local 8x8 spatial blocks before generic PISA. Both square and rectangular spatial grids are supported when each dimension is divisible by 8; other shapes retain the generic linear-block path. A synthetic Wan-shaped BF16 check at `B=1,H=12,T=8192,D=128`, with a `2x64x64` token grid, produced finite output. Median kernel times were 21.936 ms for HYD PISA, 16.865 ms for ComfyUI PyTorch SDPA, and 13.553 ms for Flash; PISA cosine similarity against SDPA was 0.816548. This shape is therefore compatible but not performance-enabled by default evidence. No local Wan checkpoint was available for a model-level quality run. Causal Wan KV-cache attention is explicitly rejected by PISA and remains on the existing backend because cached Q and K/V lengths differ.

The generic Triton path was also checked for batch isolation with FP16 `B=4,H=8,T=8192,D=64`. All outputs were finite, and batch 0 from the four-image call exactly matched the same input executed alone.

An actual SDXL v-prediction workflow at 1536x1536 and 12 steps used dense attention in the four encoder-side high-resolution layers and generic 0th-order PISA in six decoder-side layers. The first real PISA call measured cosine 0.999624 and MAE 0.006393 against dense SDPA. The resulting image was coherent and free of stripes or non-finite pixels; against the same-seed Flash image it measured SSIM 0.679058 and RGB cosine 0.971958. Two warm paired runs averaged 20.193 seconds for PISA and 20.087 seconds for Flash, so this SDXL profile is compatible but does not demonstrate an end-to-end speedup on the measured workflow.

PISA is approximate and deliberately opt-in. Against Flash Attention, the coherent same-seed output measured SSIM 0.961379 and RGB cosine 0.999484. The spatial path accepts only the validated 23-block sparse profile: 32 became non-finite during 30 steps, 33/36 were non-finite on the first step, and other sparse budgets are not production-validated. The 144-block profile remains available only as the dense SDPA validation path.

## Install

### ComfyUI Registry / Manager

After the first Registry release is approved, search for `RDNA35 Attention` in ComfyUI-Manager, or run:

```powershell
comfy node install rdna35-attention
```

### Git

Place this folder under:

```powershell
C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-Attention
```

Restart ComfyUI. Fixed-block nodes appear under `RDNA35/Fixed Block Attention`; the opt-in PISA patch appears under `RDNA35/Attention Research`.

The PISA patch additionally requires the wheel under `native/rdna35_pisa_ck`. Build it with the exact PyTorch/ROCm runtime and `MAX_JOBS=32`, then install it with `pip install --no-deps`. The wheel is BF16-only and rejects a different PyTorch/ROCm nightly before loading its extension.

Recommended runtime for the generic Triton research nodes:

- PyTorch ROCm build
- Triton compatible with that PyTorch ROCm build
- AMD RDNA3.5 target such as `gfx1150`, `gfx1151`, or `gfx1152`

The optional fourth benchmark backend is SageAttention ROCm7 1.0.6. Install its release wheel into the same Python environment as ComfyUI:

```powershell
python -m pip install --no-deps https://github.com/guinmoon/SageAttention-Rocm7/releases/download/v1.0.6_rocm7/sageattention-1.0.6-py3-none-any.whl
```

The prebuilt PISA wheel is narrower: Windows, BF16, and `gfx1151` only.

PyTorch ROCm still uses `torch.cuda` APIs and `device="cuda"` strings. ROCm detection is based on `torch.version.hip`.

## Patch Safety

`RDNA35 Patch Model Attention` defaults to `exact_only`. In that mode, ordinary ComfyUI attention calls are left unchanged unless a call explicitly declares:

```python
rdna35_attention_semantics = "fixed_block_diagonal"
```

`experimental_force_block_local` is opt-in. It still refuses calls that are not explicitly marked or otherwise proven self-attention. Cross-attention is never intentionally converted to block-local attention.

If a safe local model patch cannot be installed, the node returns the original model and includes the reason in the `info` output.

`RDNA35 Patch Anima PISA Attention` requires the `is_self_attention=True` marker and the initial-block marker supplied by the Cosmos/Anima attention owner. The `anima_1536_spatial` policy never converts the first four transformer blocks, cross-attention, masked calls, FP16 calls, or another sequence length. Its sparse profile is fixed at the validated 23/144 blocks. Once a native/Flex call starts, failures are surfaced instead of retrying another backend on the same asynchronous stream.

## Optimized Dispatch Conditions

The Triton kernel is used only when all of these are true:

- PyTorch ROCm/HIP is detected through `torch.version.hip`
- Q/K/V are on the same `cuda` device, which is the PyTorch ROCm device type
- Triton imports successfully
- dtype is `float16` or `bfloat16`
- head dimension is 32, 64, or 128
- `block_size == 64`
- Q/K/V shapes match and represent self-attention
- layout is supported and normalized to contiguous `[BH,T,D]`
- no arbitrary mask is passed
- no input has `requires_grad=True`

Otherwise dispatch falls back to the PyTorch reference implementation with a reason. The ComfyUI patch falls back to the original attention backend for calls that are not known to be fixed block-diagonal.

## Limitations

- Forward/inference only
- The optimized PISA paths are gfx1151-only
- Generic PISA prioritizes the fused CK HYD statistics profile for BF16 `D=128`; FP16 and other head dimensions up to 256 use the fused Triton statistics fallback
- SD1.5/SDXL, Wan, Wan AR, and LTX use generic PISA only for explicitly marked, unmasked self-attention with matching Q/K/V and `T>=8192`. Cross-attention, GQA, short sequences, Wan KV-cache attention, and LTX guide masks chain to the previous ComfyUI backend
- SDXL `T=4032,D=64` fallback execution is verified. Wan dispatch is covered by synthetic contract tests. LTX 2.3 is verified to complete an actual 768x768, 73-frame workflow with safe short-sequence fallback, but neither model has a production PISA quality/performance profile
- A generic CK/Triton/Flex compile failure falls back to the previous attention backend; an out-of-memory error remains fatal so a second large allocation is not attempted
- First use compiles the selected-block FlexAttention kernel and the dense centroid approximation through Inductor/Triton; benchmark cold and steady-state runs separately
- PISA output is approximate and can change composition relative to Flash Attention
- Spatial sparse PISA accepts only the validated 23 exact blocks; other sparse budgets are rejected after real-model non-finite results at 32/33/36
- Fixed `block_size=64`
- No arbitrary mask in the Triton path
- No cross-attention conversion
- No CUDA extensions
- No NVIDIA Blackwell TLX features such as async dot, TMA, TMEM, mBarrier, or tcgen05
- No silent replacement of normal full attention

## Tests

The runtime Python on this machine may not have `pytest`; the tests are `unittest` compatible:

```powershell
$py = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
& $py -m unittest discover -s C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-Attention\tests -v
```

Reference-only tests run on CPU. Triton tests skip automatically when ROCm/Triton is unavailable.

## Benchmark

```powershell
$py = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
& $py C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-Attention\scripts\bench_fixed_block_attention.py --tokens 256 --head-dim 64 --dtype float16 --mode auto
```

The benchmark reports:

- PyTorch reference loop latency for fixed block-diagonal attention
- PyTorch SDPA with a block-diagonal mask, which has the same fixed-block semantics
- PyTorch full SDPA, which is normal full attention and not semantically equivalent
- Triton latency and speedups only if the Triton backend actually runs
- The full-SDPA semantic delta vs fixed-block reference, so the output difference is visible

## Primary References

- [PyTorch TLX Block Attention blog](https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/)
- [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.12/notes/hip.html)
- [ROCm Triton install docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html)
- [ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [ROCm RDNA3.5 optimization docs](https://rocm.docs.amd.com/en/7.13.0-preview/reference/system-optimization/rdna3-5.html)
- [ComfyUI custom node backend docs](https://docs.comfy.org/custom-nodes/backend/server_overview)
- [PISA paper (arXiv:2602.01077)](https://arxiv.org/abs/2602.01077)
- [Official PISA implementation](https://github.com/xie-lab-ml/piecewise-sparse-attention)
