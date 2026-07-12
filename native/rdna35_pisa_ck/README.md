# rdna35-pisa-ck

`gfx1151`専用のPISA HYD first-order forward attention wheelです。Windows、Python 3.13、PyTorch 2.14 ROCm 7.15向けに事前buildし、ComfyUI実行中のJIT extension buildやnetwork accessは行いません。

## 対応範囲

- dtype: bfloat16
- head dimension: 128
- block size: 64 tokens
- sequence length: `T <= 9216`（最大144 blocks）
- attention: non-causal self-attention、forward-only
- architecture: `gfx1151`
- backward、float16、CUDA/NVIDIAは非対応

通常APIは連続な`[BH,T,128]`を受け取ります。Anima 1536x1536統合用の`forward_spatial_bhtd`は、連続な`[B,T,H,D]` storageに対する`[B,H,9216,128]` viewを直接受け取り、Python側のlayout copyを行いません。

## 実装

- C++/HIPとCK Tile型を使う`block_stats` kernelが、64-token blockごとのQ centroid、FP32 K centroid、V meanを計算します。入力を`1 / length`倍してからFP32へ累積し、有限BF16のsum overflowを避けます。
- C++/HIPの`pack_spatial_qkv` kernelが、Animaのraster tokenを公式PISAと同じ8x8空間block順へ並べ替えます。8 tokens x 8 heads x D128の16 KiB shared-memory tileを使い、Q/K/Vを1 launchで転置します。
- PyTorch 2.14 FlexAttentionが、routingで選択したtoken-level exact blocksと、未選択blockのlength-weighted centroid tailを共通softmaxへ統合します。
- WMMA `torch.bmm(..., out_dtype=torch.float32)`がfirst-order correctionを適用します。`H_sum`はKをFP32 centroidで先にcenter化してから計算し、大きなGEMM結果同士の減算を避けます。
- spatial outputはC++/HIP kernelでraster順の`[B,T,H*D]`へ戻します。
- package全体をfake implementation付きcustom opとして公開するため、外側の`torch.compile(fullgraph=True)`から呼び出せます。

Composable Kernelはbuild時だけ必要です。runtime AITER依存はありません。

## API

```python
import rdna35_pisa_ck

output = rdna35_pisa_ck.forward(
    q,
    k,
    v,
    exact_blocks=23,
    scale=None,
    sink_block=None,
)

output_bthd = rdna35_pisa_ck.forward_spatial_bhtd(
    q_bhtd,
    k_bhtd,
    v_bhtd,
    exact_blocks=23,
)
```

`forward_spatial_bhtd`では、実Anima 30-stepで有限性を確認した`exact_blocks=23`だけをsparse PISAとして許可します。32 blocksは30-step途中、33/36 blocksは初stepでNaN化し、0〜22および24〜143 blocksをproduction未検証域として明示エラーにします。144 blocksだけは近似を使わないdense SDPAとして許可します。ComfyUI nodeは23/144（15.625% budget）へ固定します。

`capabilities()`は対応条件と`spatial_sparse_exact_blocks`を返します。`build_info()`はpackage/API version、PyTorch・ROCm version、CK commit、target architectureを返します。Python API 5はimport時にnative API、PyTorch C++ ABI、ROCm、CK commit、gfx targetを照合し、不一致なら実行しません。

## Build

検証済み環境:

- PyTorch `2.14.0a0+rocm7.15.0a20260704`
- ROCm `7.15.26263`
- Composable Kernel commit `4975bd0c8e17a54bdc27c746527a385e7383bb07`
- Visual Studio 2022 C++ toolchain、Ninja

```powershell
$env:CK_DIR = 'C:\Users\HarutoWatanabe\composable_kernel'
$env:MAX_JOBS = '32'
$env:PYTORCH_ROCM_ARCH = 'gfx1151'
python -m pip wheel . --no-build-isolation --no-deps
python -m pip install --force-reinstall --no-deps .\rdna35_pisa_ck-0.7.0-cp313-cp313-win_amd64.whl
```

`setup.py`はCK commitを検証し、`MAX_JOBS`を最低32、code objectを`gfx1151`へ固定します。

## 実測

PyTorch 2.14 / ROCm 7.15 / gfx1151、BF16、`B=2,H=16,T=9216,D=128`、warmup後のGPU Event medianです。

| 処理 | latency |
| --- | ---: |
| native spatial Q/K/V pack | 2.412 ms |
| PISA 23/144 blocks | 37.917 ms |
| ComfyUI Flash Attention | 46.411 ms |

PISA attention全体はFlashより8.495 ms、18.3%高速でした。以前のhead固定packは19.54 msだったため、shared tile版でpackを約8.1倍高速化しています。

同一seedのAnima INT8_ConvRot、1536x1536、Spectrum euler/simple、30 steps、CFG 5、SEA off、17/30 actual forwardsでは次の結果でした。

| backend | Sampler | Prompt total |
| --- | ---: | ---: |
| Flash Attention | 69.12 s | 69.17 s |
| PISA 23/144 | 68.63 s | 68.68 s |

API 5最終wheelを各backendのwarm-up後に測定し、SamplerとPrompt totalを0.49秒（0.7%）短縮しました。PISAは近似attentionなので、同seed画像はFlashと完全一致しません。測定画像は破綻せず、Flash比でSSIM 0.961379、RGB cosine 0.999484でした。近似差を許容する明示opt-in用途として扱います。

## Compile

native extensionは事前build済みですが、FlexAttentionは最初のshape/device/scaleでInductor kernelをcompileします。初回1-stepとsteady-state 30-stepを分けて測定してください。step数またはPISA/Flashを切り替えると`sample_sigmas`長とoverride有無に対する一度限りの特殊化が発生しますが、同じ30-step設定の反復では追加recompile、`cache_size_limit`、graph breakは発生しませんでした。ComfyUI統合は最初の4 transformer blocksをFlashのまま維持し、後段24 self-attention blocksだけをPISAへ送ります。cross-attention、FP16、別解像度、mask付きattentionは既存backendへ戻ります。

## Test

```powershell
Set-Location C:\ComfyUI
python -m unittest discover -s custom_nodes\ComfyUI-RDNA35-FixedBlockAttention\native\rdna35_pisa_ck\tests -v
python -m unittest discover -s custom_nodes\ComfyUI-RDNA35-FixedBlockAttention\tests -v
```

0.7.0ではnative 11件、custom-node 35件がgfx1151環境で合格しています。
