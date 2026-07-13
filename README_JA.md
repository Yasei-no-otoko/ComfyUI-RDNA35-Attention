# ComfyUI RDNA35 Attention Research

[English](README.md)

PyTorch ROCm向けのComfyUIカスタムノードです。固定64-token block-diagonal self-attention、gfx1151用exact full-attention研究kernel、Anima 1536x1536向けPISA HYD経路を提供します。

通常のComfyUI attentionをグローバル置換しません。MODELをcloneし、明示的なself-attention markerと検証済みshapeを満たす呼び出しだけをmodel-localに切り替えます。PISAは近似計算でありopt-inです。

## インストール

Registry公開後はComfyUI-Managerで`RDNA35 Attention`を検索するか、次を実行します。

```powershell
comfy node install rdna35-attention
```

Gitから導入する場合は、このrepositoryを`ComfyUI/custom_nodes`へcloneしてComfyUIを再起動してください。

PISA CK/Flex経路はWindows・gfx1151・BF16専用です。`native/rdna35_pisa_ck`を使用中のPyTorch/ROCm環境でビルドし、wheelを`pip install --no-deps`で導入する必要があります。generic fixed-block reference/Triton nodesはこのnative wheelなしでも使用できます。

## gfx1151実測

`rdna35-pisa-ck` 0.7.1 API 6、PyTorch 2.14 ROCm 7.15、BF16 `B=2,H=16,T=9216,D=128`、23/144 exact-block profileで測定しました。

| Spatial self-attention backend | 完全な1 call | 相対速度 | 短縮 |
|---|---:|---:|---:|
| ComfyUI Flash Attention | 46.411 ms | 1.000倍 | baseline |
| RDNA35 PISA CK/Flex fused | **25.121 ms** | **1.848倍高速** | **21.290 ms (45.9%)** |

CK kernelはQ/K/V centroid、first-order covariance行列、公式HYDのrouting biasに使うcovariance normを融合して計算します。

### PISA論文とAnimaのlayer構造

[PISA v2 Appendix E](https://arxiv.org/html/2602.01077v2)はSD3.5とFLUX.1で先頭4層をDenseに保ちます。ただし論文の70%/85% sparsityは、PISA対象層内で近似するKV blockの比率であり、Sparse化するtransformer層の割合ではありません。[公式FLUX processor](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/main/piecewise_attn/models/flux/flux_processor.py)もprocessor ID 0-3をDense SDPA、4以降をPISAへ送ります。

| モデル | Transformer構造 | Dense層 | PISA層 | 層内sparsity |
|---|---|---:|---:|---:|
| SD3.5 Medium | 24 joint MMDiT blocks | 0-3 | 4-23 | 70% |
| SD3.5 Large Turbo | 38 joint MMDiT blocks | 0-3 | 4-37 | 85% |
| FLUX.1-dev | 19 double-stream + 38 single-stream blocks | 0-3 | 4-56 | 85% |
| Cosmos-Predict2 2B / Anima | 同型のself-attention、cross-attention、MLP blockを28個 | **0-19** | **20-27** | **84.03% (23/144 exact)** |

AnimaはCosmos-Predict2 2Bのfine-tuneであり、FLUX transformerではありません。28層はすべてimage self-attention、独立したtext cross-attention、MLPの同じ構造で、FLUXの19/38 phase境界はありません。従来の4-27設定は、構造の異なるFLUXの閾値を移してAnimaの28層中24層をSparse化していたため、大きな構図変化が発生しました。この設定は既定値および推奨benchmarkから外しました。

### BF16 Anima layer search

`anima_aestheticV10.safetensors`を1536x1536、標準KSampler、INT8 ConvRotなし、model compileなし、Spectrumなしで測定しました。プロンプトは公式[Anima `example.png`](https://huggingface.co/circlestone-labs/Anima/blob/main/example.png)と同一、30 steps、CFG 5、Euler/simple、batch 1、同一2 seedです。SSIM、RGB cosine、PSNRは同seedのDense Flash画像との比較です。時間はprocess順序を含む診断値で、warm release medianではありません。

| Sparse self-attention層 | Seed 856853657535148 E2E | SSIM / cosine / PSNR | Seed 856853657535149 E2E | SSIM / cosine / PSNR | 判定 |
|---|---:|---:|---:|---:|---|
| なし (Flash) | 142.974秒 | 1 / 1 / inf | 128.265秒 | 1 / 1 / inf | Dense基準 |
| **20-27** | **129.650秒** | .723942 / .963230 / 14.935 | **132.249秒** | .630565 / .896732 / 10.794 | 審美性重視の既定profile |
| 24-27 | 129.220秒 | **.908147 / .993142 / 22.138** | 128.670秒 | **.781257 / .958885 / 14.717** | Flash近似重視profile |
| 4-11、20-27 | 129.900秒 | .607146 / .929997 / 12.196 | 132.720秒 | .596093 / .885909 / 10.325 | text/layout低下で棄却 |
| 0-3、8-11、16-19 | 147.100秒 | .575142 / .885335 / 10.003 | 145.910秒 | .571344 / .894699 / 10.619 | 低速かつtext低下で棄却 |

20-27は知覚的な審美性で選択したため、数値上Flashへ最も近い設定ではありません。同seedのFlash忠実度を優先する場合は24-27を使用してください。nodeの`first_pisa_layer`と`last_pisa_layer`はinclusiveです。1 nodeは連続範囲を表し、複数nodeのchainで非連続な研究用設定も作れます。

既定profileのruntime verifierはmodel forwardごとにPISA self-attention 8 call、cross-attention 0、対象fallback 0を要求します。再現workflowは[純BF16](workflows/Anima_BF16_PISA_Pure_E2E_Benchmark.json)と[INT8 ConvRot](workflows/Anima_INT8_ConvRot_PISA_E2E_Benchmark.json)で、どちらも20-27を明示保存しています。disk I/Oを性能値へ含めないため`SaveImage`は既定無効です。

非同期attention経路ではprogress barをGPU時間として利用できません。比較にはtqdmのiteration速度ではなく、同期後のPrompt E2EまたはGPU eventを使用します。

## TODO

- 20-27 profileをexperimentalから昇格する前に、公式example以外のpromptと2 seedを超える品質matrixへ拡張し、知覚指標、prompt adherence、anatomy、文字、layer別attention誤差を比較します。

## 安全性と制限

- inference forward専用
- block sizeは64固定
- arbitrary maskとcross-attentionをblock-localへ変換しない
- PISAはAnima `T=9216`、BF16、検証済み23/144 sparse profileのみ
- 対象外の呼び出しは既存ComfyUI backendへchainする
- PISA開始後のnative例外を同じ非同期stream上で隠して再実行しない

node一覧、計測値、詳細なdispatch条件、テスト方法は[英語README](README.md)を参照してください。
