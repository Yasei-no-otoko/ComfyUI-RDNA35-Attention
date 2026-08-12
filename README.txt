ComfyUI RDNA35 Attention - MiniMax-H3 gfx1151
================================================

MiniMax-H3 専用の gfx1151 BF16 attention layout 最適化を追加しています。

実装
----

`RDNA35 Patch MiniMax-H3 gfx1151 Attention` は、MiniMax-H3 の fused QKV
projection から生じる interleaved Q/K/V view を、既存の ComfyUI exact
attention backend に渡す直前だけ連続 layout へ pack します。attention の
数式、選択済み backend、モデル全体の global attention dispatch は変更しません。
処理は model-local clone にだけ適用され、既存の model-local override は chain
されます。

対象条件は ROCm gfx1151、BF16、B=1、H=56、D=128、mask/GQA/causal なし、
forward-only です。`4608 <= T < 12000` は KV、`12000 <= T <= 16500`
は QKV を一回の `torch.stack` allocation に pack します。T<4608、T>16500、
既に packed された入力、非対応 dtype/device/attention 呼び出しは既存経路へ
戻ります。

有無比較の実測結果
------------------

PyTorch 2.14 / ROCm 7.15、gfx1151、AOTriton、BF16、warmup 5 回後に raw と
packed を交互に GPU event 20 回測定した中央値です。ここで patchなしは native
interleaved layout、patchありは `RDNA35PatchMiniMaxH3GFX1151Attention` が
選ぶ packed layout に対応します。同じ乱数seed、同じ入力、同じ exact SDPA
backendを使用しています。

  shape                         patchなし    patchあり     speedup       短縮率
  B=1,H=56,T=9170,D=128 (KV)    317.423 ms    86.242 ms    3.681x        72.83%
  B=1,H=56,T=16500,D=128 (QKV)  2112.046 ms   279.717 ms   7.551x        86.76%

両 shape の実測出力は BF16 で最大絶対誤差 0、cosine 1.0 でした。correctness
gate は allclose(atol=rtol=5e-2) です。pack allocation 単体は T=9170 で約
251 MiB、T=16500 で約677 MiBであり、モデル tensor、出力、backend workspace
のメモリは別途必要です。これは attention call の測定で、sampling workflow
全体の時間ではありません。

ワークフロー
------------

`workflows/video_minimax_h3_r2v_gfx1151_mixed_fp16_audio_spectrum_ck_attention_rdna35_h3_attention.json`
は、ComfyUI の現行 MiniMax-H3 R2V gfx1151 + CK/Spectrum workflow の専用コピーです。
元の workflow は変更せず、次の model chain にパッチ node を追加しています。

  UNETLoader -> ModelAttentionBackend -> RDNA35 Patch MiniMax-H3 gfx1151 Attention
             -> SolAttnPatch -> SpectrumApplyMiniMaxH3 -> BasicGuider

ComfyUI の `user/default/workflows` にコピーして読み込み、patch node の
`enabled` を true にしてください。現在のモデルが FP16 attention を選ぶ場合は
BF16専用条件に一致せず、node は既存 backend へ fallback します。T2V の一体型
MiniMax-H3 node は MODEL socket を公開しないため、この model-local patch node を
直接接続できません。対象は MODEL chain が見える R2V workflow です。

検証
----

専用 unit/integration test 70件、gfx1151 BF16 integration、Magpie correctness
2/2 を通過しています。今回の有無比較CLIも両shapeでcorrectness passです。
Magpie の performance は skip し、速度比較は直接のGPU-event medianを正と
しています。
