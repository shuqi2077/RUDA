# 一般 PyTorch モデルのコンパイル

[目次](README.md) · [詳細参照](../en/model-compiler.md)

ruda_torch.compile は通常の nn.Module/関数を wrap し、AOTAutograd が forward/backward を生成します。対応領域は StaticGraph、他は元の device dispatch。すべての実行演算に device kernel が必要です。StaticGraph.from_model は同じ callable wrapper で replay オブジェクトではありません。

| オプション | 既定と契約 |
| --- | --- |
| capture | aot。eager は明示的な非コンパイル実行。 |
| native | auto、off は AOT のみ、required は未対応 captured operator/guard を拒否。 |
| device_type | ruda。ruda:0 ではなく type。cpu は明示参照。 |
| fullgraph / dynamic | False / None。break 禁止は全演算 native を意味しない。 |
| min_native_ops | 2、1–256。required は最小1。 |
| cache_size | 4、1–64、**領域ごと**。 |
| decompositions | None。正確な overload→callable の明示変換。 |

eager は required、fullgraph/dynamic、非空 decompositions と組み合わせません。AOT wrapper は suppress_errors=True を拒否し、失敗を暗黙 eager 再実行へ変えません。

native overload は clone、UNARY_CODES の unary、add/sub/mul/div Tensor/Scalar、mm/bmm、keepdim sum/mean、softmax/log-softmax と backward、SiLU/sigmoid/tanh backward。二項 broadcast/混合 dtype、batch broadcast、暗黙 half→FP32 softmax、reduction dtype override は native 領域に入りません。

native 入力は dense strided ruda:0 FP32/FP16/BF16、非空 rank1–8、uint32 indexing。連続 staging へ copy、operation output は contiguous。領域最大256 node/512 tensor。view、mutation、RNG、未対応演算は auto で元 device、required でエラー。CPU fallback や失敗後再試行ではありません。

model と入力を明示移動します。parameter identity と直接 state_dict の元 key を保持、子 module に入れた wrapper は _original prefix を持ちます。AOT は一階のみ、eager の高階微分は個々の operator 次第。

cache は shape/stride/dtype/device/stream/inference state で区別する bounded LRU。native output は返却前に clone し、次 replay が以前の結果を上書きしません。copy/staging は追加メモリと時間を必要とし、capture 自体は高速化の保証ではありません。

info の plan と実行を区別し、native_replays/native_nodes_executed/reference_reasons を確認。NativeCoverageError は required 契約、GraphExecutionError は元 device 上の実行失敗です。close は全 backward 完了後。make_backend は外部 torch.compile 用で、その global 設定は caller の責任。optimizer や Python side effect の全 capture は保証しません。
