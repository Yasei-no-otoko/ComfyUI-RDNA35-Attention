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

Anima INT8_ConvRot、1536x1536、Spectrum 30 steps、CFG 5、Euler/simple、batch 1、17 actual forwardsで再測定しました。ポジティブ・ネガティブは公式[Anima `example.png`](https://huggingface.co/circlestone-labs/Anima/blob/main/example.png)の埋め込みmetadataと完全に同一です。`mod_w_profile=off`のため、Spectrumによる追加quality conditioningはありません。

| ComfyUI選択backend | PISA | Seed 856853657535148 | Seed 856853657535149 | Prompt E2E中央値 | Flash比 |
|---|---:|---:|---:|---:|---:|
| Flash Attention | 有効 | 64.337秒 | 66.217秒 | **65.277秒** | **-11.939秒 (-15.5%)** |
| Flash Attention | 無効 | 71.816秒 | 82.616秒 | 77.216秒 | baseline |
| SageAttention ROCm7 1.0.6 | 有効 | 78.609秒 | 80.541秒 | 79.575秒 | +2.359秒 (+3.1%) |
| PyTorch SDPA | 有効 | 98.163秒 | 101.673秒 | 99.918秒 | +22.702秒 (+29.4%) |
| SageAttention ROCm7 1.0.6 | 無効 | 149.090秒 | 151.760秒 | 150.425秒 | +73.209秒 (+94.8%) |
| PyTorch SDPA | 無効 | 302.027秒 | 302.158秒 | 302.093秒 | +224.877秒 (+291.2%) |

各backendは専用または同一resident processでwarm-up後に測定し、Prompt E2EにはSampler、VAE decode、preview出力を含めています。Flash/PISAは同一processのABBA順、SDPAとSageはそれぞれ専用processで未適用・PISA適用経路をwarm-up後に測定しました。backend固有Inductor compileを含むSDPA 719.036秒、Sage 490.612秒の初回runはsteady-state表から除外しています。

PISA + SDPAおよびPISA + Sageの1-forward verifierは、対象self-attention `24/24`、cross-attention 0、fallback 0、runtime failure 0でした。verifierはlayerごとのDynamo再compileを誘発するため性能値には含めていません。再現用workflowは[workflows/Anima_INT8_ConvRot_PISA_E2E_Benchmark.json](workflows/Anima_INT8_ConvRot_PISA_E2E_Benchmark.json)です。PISA nodeの`enabled`を無効にすると、起動時に選択したbackend単体のbaselineになります。

### INT8 ConvRot・model compile・Spectrumを除外したBF16切り分け

品質低下の原因を分離するため、`anima_aestheticV10.safetensors`、標準KSampler、model compile nodeなしで再実行しました。解像度、プロンプト、seed、30 steps、CFG 5、Euler/simpleは同一です。以下はprocess順序を含む診断値であり、warm steady-stateのrelease benchmarkではありません。

| 選択backend | PISA | Seed 856853657535148 | Seed 856853657535149 |
|---|---:|---:|---:|
| Flash Attention | 無効 | 147.178秒 | 130.267秒 |
| Flash Attention | 有効 | 122.257秒 | 126.271秒 |
| PyTorch SDPA | 無効 | 709.650秒 | 567.443秒 |
| PyTorch SDPA | 有効 | 184.900秒 | 184.358秒 |
| SageAttention ROCm7 1.0.6 | 無効 | 289.465秒 | 267.070秒 |
| SageAttention ROCm7 1.0.6 | 有効 | 148.298秒 | 146.293秒 |

純Flash、SDPA、Sageは同じ構図を維持し、深刻なノイズや余分な腕は再現しませんでした。一方、3種類すべてのPISA併用結果で、同方向の大きな構図変化と身体・看板の交差が発生しました。残るgeometry低下はfallback backend固有ではなくPISA近似に追従しています。INT8 ConvRot、Spectrum、model compileは以前の破綻を増幅した可能性がありますが、PISAによる構図変化の必須条件ではありません。

非同期経路ではprogress barをGPU時間として利用できません。純Flashは30 stepsを約2秒と表示した一方でPrompt E2Eは128.80-146.06秒、純SDPAも約2秒表示に対してE2Eは566.64-708秒でした。PISA経路はstep loop内の待機が増えるため、119-181秒のprogress表示がE2Eへ近づきます。比較にはtqdmのiteration速度ではなく、同期後のPrompt E2EまたはGPU eventを使用します。

純BF16の再現workflowは[workflows/Anima_BF16_PISA_Pure_E2E_Benchmark.json](workflows/Anima_BF16_PISA_Pure_E2E_Benchmark.json)です。2つのbenchmark workflowには同一seed品質比較用の`SaveImage` nodeを既定無効で追加しています。保存時だけ有効化してbackend別prefixを設定してください。disk I/Oを性能値へ含めないため、既定では無効です。

## TODO

- PISAとfallback backendの組み合わせを推奨する前に、構図・geometry低下を定量化する。BF16切り分けでINT8 ConvRot、model compile、Spectrumを除外すると深刻なノイズと余分な腕は再現しませんでしたが、身体・看板の交差はFlash、SDPA、Sageの全PISA経路に追従しました。同一seed matrixを拡張し、知覚指標とlayer別attention誤差を比較します。この品質gateを通過するまでPISAはexperimental扱いです。

## 安全性と制限

- inference forward専用
- block sizeは64固定
- arbitrary maskとcross-attentionをblock-localへ変換しない
- PISAはAnima `T=9216`、BF16、検証済み23/144 sparse profileのみ
- 対象外の呼び出しは既存ComfyUI backendへchainする
- PISA開始後のnative例外を同じ非同期stream上で隠して再実行しない

node一覧、計測値、詳細なdispatch条件、テスト方法は[英語README](README.md)を参照してください。
