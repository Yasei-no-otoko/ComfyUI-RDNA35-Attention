# Changelog

## 0.1.5 - 2026-08-12

- Select PyTorch SDPA/AOTriton for the gfx1151 MiniMax-H3 workflow instead of the CUDA-only Comfy Kitchen SageAttention path.

## 0.1.4 - 2026-08-12

- Add the MiniMax-H3 gfx1151 layout optimization README and dedicated R2V workflow.

## 0.1.2 - 2026-07-14

- Fuse Anima HYD centroids, first-order covariance statistics, and routing bias in the gfx1151 native kernel.
- Add inclusive Anima PISA layer controls and select layers 20-27 as the quality-tuned default.
- Document the SD3.5, FLUX.1, and Cosmos-Predict2 layer schedules and matched-seed Anima layer search.

## 0.1.1 - 2026-07-12

- Add the project image used by the Comfy Registry and ComfyUI-Manager.

## 0.1.0 - 2026-07-12

- Add safe model-local fixed 64-token block-diagonal attention dispatch.
- Add ROCm/Triton diagnostics and correctness benchmarks.
- Add gfx1151 exact full-attention research kernel.
- Add opt-in Anima 1536 PISA CK/Flex attention with the validated 23/144 profile.
- Add English and Japanese documentation.
