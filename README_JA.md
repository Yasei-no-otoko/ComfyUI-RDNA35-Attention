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

Anima INT8_ConvRot、1536x1536、Spectrum 30 steps、CFG 5、17 actual forwardsのABBA測定では、FlashのSampler中央値76.08秒に対して融合PISAは65.94秒で、13.3%短縮しました。Prompt中央値は78.54秒から68.38秒へ12.9%短縮しています。runtime verifierで対象self-attention 24/24、cross-attention 0、fallback 0を確認済みです。

## 安全性と制限

- inference forward専用
- block sizeは64固定
- arbitrary maskとcross-attentionをblock-localへ変換しない
- PISAはAnima `T=9216`、BF16、検証済み23/144 sparse profileのみ
- 対象外の呼び出しは既存ComfyUI backendへchainする
- PISA開始後のnative例外を同じ非同期stream上で隠して再実行しない

node一覧、計測値、詳細なdispatch条件、テスト方法は[英語README](README.md)を参照してください。
