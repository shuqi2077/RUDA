# 共通 stack autotuning

[目次](README.md) · [runtime](runtime-api.md) · [方針とサンプル](../en/stack-autotuning.md)

共通 controller は同じ意味の候補を検証し、完了時間を比較して device/workload ごとに cache。matmul、attention、forward convolution、fused matmul、packed greedy generation の接続です。FFT/sparse/solver/communication や multi-GPU placement、全 model 組合せ探索を自動で扱いません。

enable_stack_autotune(policy,cache_directory) を model loading/worker/初回演算前に一度呼びます。None は memory only。device backend の stack-autotune、fusion の device-stack-autotune、LLM の stack-autotune feature が必要。std native 向けで no-std/WASM ではありません。

| Mode | 動作 |
| --- | --- |
| Explore | valid cache、それ以外は検証と時間探索。 |
| CacheOnly | 新しい探索をせず miss は明示 reference。disk 初回は検証する。 |
| Disabled | この controller の cache/trial を使わず reference。旧 LocalTuner route とは別。 |

未 install のときに旧 route を維持します。既定 EndToEnd、検証必須、warmups2、paired samples7、候補最大32、soft budget30秒、min_speedup1.05、relative MAD .15。容量1024、TTL7日、parallel1、regression pairs7/ratio1.15、workspace_limit None。容差 abs1e-4/rel1e-3、合計readback64MiB。

試行ごと reference/candidate を再測定し順序を交互に変え、ratio median で選びます。EndToEnd は候補内の allocation/layout/dispatch/completion wait を含む。準備した isolated input は外。budget は完了 trial 間でのみ確認し kernel を中断しません。1.05 は選択閾値で実測 speedup ではありません。

reference と candidate の writable state は分離、matmul は stride/offset を保持、generation は専用 KV cache。NaN/Inf や不正結果は拒否。validator 不足は reference、require_validation=False は明示的 unverified 選択。generation は校正 prompt の token/終了一致で、全 prompt の保証ではありません。

key は operation/candidate revision、device/driver/build、実 shape/stride/dtype/精度/options/context。driver が不明なら memory only。起動前の RUDA_AUTOTUNE_DRIVER_TAG/BUILD_TAG/CONTEXT_TAG は caller の immutable deployment 情報で、既定 context 名が隔離を検出するわけではありません。

disk は専用 stack-autotune-v1、完全 key/checksum/version/TTL を照合。digest は非暗号学的で署名ではありません。memory hit は強制 sync/readback なし、disk cold hit と探索は latency を増やすため起動/offline で校正します。

contender と nested miss は reference、cross-process exclusion はありません。完了不明は tuning lane を fault し、GPU reset/現在要求の再実行はしません。workspace_limit は未知 estimate を拒否し reference も拒否し得ます。record_comparison は caller の同条件・正解確認済み paired time を使い、無言の shadow model 実行をしません。

StackTuner の new/policy/stats/reports/select/invalidate/lower_level_fingerprint が controller API。stats は GPU util ではなく、report は bounded local diagnostic。invalidate は将来の選択だけ、保持済み GenerationPlan は明示的に再校正します。cache の内部 DiskCache は公開 runtime API ではありません。
