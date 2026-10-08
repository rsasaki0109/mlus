# 競合・関連技術調査

調査日: 2026-10-08。取得した一次資料に基づくMVP設計の判断です。GPU性能比較や全製品・全バージョンの機能保証ではありません。GitHubのmain/masterは変化するため導入時に対象バージョンを再確認してください。

| 技術 | 既存の解決領域 | MLusでの判断・残る課題 |
|---|---|---|
| NVIDIA MPS | CUDAの同時実行・共有。NVIDIA device plugin資料ではMPSのメモリ/compute fractionとtime-slicingの非隔離を区別 | MPSを再実装しない。プラグインの制約を全MPS利用に一般化しない。独立した任意アロケーションの移動・モデル状態復元は別問題 |
| NVIDIA MIG | 対応GPUの固定メモリ/compute分割。mig-partedが構成管理を提供 | 強いハードウェア分割が必要なら併用候補。対象ハードウェアとprofile粒度の制約があり、任意モデルの動的offloadを提供するものではない |
| HAMi | Kubernetesのdevice sharing、対応backendのmemory/compute制限、heterogeneous device scheduling | CUDA interceptionやKubernetes schedulerを再実装しない。単一ローカルホストの軽量な協調プロファイル連携を調査対象にする |
| Run:ai | GPUオーケストレーションとの比較対象 | 公式scheduler/fractionページの取得が403でブロック。現行のquota・preemption・fraction仕様は未確認。下記の公開Model Streamerはモデル読込SDKでありschedulerの代替資料とは扱わない |
| SkyPilot | 複数cloud/clusterの配置、binpacking、failover、idle cleanup、既存MLコードとの連携 | fleet配置層を再実装しない。モデル内ピーク推定とホスト内協調制御の連携余地を検証する |
| vLLM | PagedAttention、continuous batching、prefix caching等の推論エンジン内メモリ効率化 | LLM servingでは既存エンジンを利用する。ジョブ間予約とエンジン内KV cache管理は別の層 |
| PyTorch allocator | caching allocator、allocated/reserved/peak観測、allocator設定、メモリpool | empty_cacheは未使用cacheを解放し、生きているtensorの占有量を減らさない。MLusは観測値を将来の協調APIに利用する候補 |

## 実際に取得した資料

- [NVIDIA Kubernetes device plugin](https://github.com/NVIDIA/k8s-device-plugin#shared-access-to-gpus): sharing、MPS、MIG制約。プラグイン固有のexperimental表示等を製品全体の状態とは解釈しない。
- [NVIDIA mig-parted](https://github.com/NVIDIA/mig-parted): 固定メモリ・compute partition、既存構成ツール。
- [HAMi README](https://github.com/Project-HAMi/HAMi): sharing/isolation/schedulingの対象範囲とbackend依存。
- [SkyPilot README](https://github.com/skypilot-org/skypilot): fleet schedulingとbinpacking。
- [vLLM README](https://github.com/vllm-project/vllm): PagedAttentionとcontinuous batching。
- [PyTorch CUDA notes](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md): caching allocator、empty_cache、UVM。資料ではDLのアクセス傾向とUVMのpage-fault/transfer overheadを理由に明示的な配置を推奨している。
- [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer): tensorファイルからGPUへの並行streamingを行うSDK。schedulerやfractionの仕様を示す資料ではない。

## 初回セットアップ時にアクセスできなかった一次資料

- https://docs.nvidia.com/deploy/mps/index.html
- https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html
- https://docs.pytorch.org/docs/stable/notes/cuda.html （GitHubの原稿で補完）
- https://run-ai-docs.nvidia.com/self-hosted/platform-management/runai-scheduler/resource-optimization/fractions

初回はHTTPS proxyが403を返したため取得できませんでした。環境公開後、MPS/MIGの公式indexは200で取得できました。PyTorch公開docsとRun:ai schedulerページはなお403でした。詳細な対応matrixの確認は継続課題です。推測で詳細を補完せず、Run:aiの現行scheduler機能、MPS/MIGの正確な対応matrix、CUDA IPC/UVMの詳細制約は実機導入前の追加確認事項とします。

## 差別化の仮説

既存schedulerや監視ツールが存在するため、CLIやbest-fitだけを独自技術として主張しません。MLusの候補は「モデル/入力形状別の実測メモリprofile」「ローカル予約台帳」「フレームワークの安全点とoffload可否」をつなげる協調プロトコルです。0.3では終了時報告をGPU別プロファイルと予約へ反映し、アプリが明示的に保存・終了するチェックポイントからの再開も実装しました。PyTorchラッパーは模擬APIで検証した段階で、CPU offloadと透過的VRAM共有は未実装です。

この仮説は、同じモデル・同じGPUで未管理並列/直列/MPSまたはMIG/MLusを比較し、OOM率・成功ジョブ/秒・p95待ち時間・総完了時間・GPU利用率・監視overheadから検証します。未管理並列がすでに問題なく動く負荷や、申告量が過大な負荷も含め、不利な結果も保存します。シミュレーションだけでは差別化や速度改善を証明できません。
