# 検証結果 — MVP 0.5

Linux x86_64 / Rust 1.90.0。NVIDIA deviceおよびnvidia-smi、実PyTorchはありません。GPU上のモデル・OOM・実性能は未測定です。

| チェック | 結果 | 検証内容 |
|---|---|---|
| cargo test --locked | 14件成功、失敗0 | 予約/配置/報告/保存の12件に加え、checkpoint契約と予約返却/FIFO再投入 |
| cargo clippy --locked --all-targets -- -D warnings | 成功 | Rust lint |
| cargo build --locked / --release | 成功 | CLI/HTTP実行バイナリ |
| scripts/smoke.py | 成功 | HTTP/CLI、dashboard配信、複数GPU配置、待機、終了状態、ログ、不正要求、監視fixture、SIGTERM |
| scripts/profile_smoke.py | 成功 | 合成報告の取込、GPU別予約増加、待機→起動、待機中の容量再評価、OOM下限、bad/missing report、CLI、lock、再起動、保存失敗で起動停止 |
| scripts/test_pytorch_wrapper.py | 6件成功、失敗0 | torch API fixtureでの引数/import/終了コード、MiB切上げ、OOM/通常エラー/checkpoint終了、mock/CUDAなし拒否 |
| scripts/checkpoint_smoke.py | 成功 | ワーカー終了前は予約維持、高優先度へ引渡し、同GPU再開、4ワーカー/200ステップのモデル状態継続、無効handoff拒否 |
| scripts/recovery_smoke.py | 成功 | durable応答、待機/履歴/ID/log/cwd復元、通常終了、実子を残すSIGKILL、起動意図の復旧抑止、明示recover、保存失敗503、破損台帳の保持、checkpoint待機の再起動 |
| scripts/recovery_demo.py | 成功 | 実デーモン再起動前後のCPU学習状態・同じjob/log・4 attempt/200 stepを記録、loss 4.9774e-19 |
| scripts/process_smoke.py | 成功 | 実子/孫のグループ継承、終了・回収後の次ジョブ起動、checkpoint handoff、通常終了、fatal-error cleanup、setsid離脱のadoption検出、recover 409と実cleanup後の解除 |
| scripts/process_demo.py | 成功 | 実親/子/孫の存在と終了を記録し、次ワーカー自身が旧3 PIDの回収を確認。GPU予約はmock |
| CPU線形回帰モデル | 成功 | 実CPU学習200 step、loss < 1e-12 |
| demo.py / --profiles | 成功 | それぞれ実mock daemonの状態記録、合成プロファイル学習、GIF |
| checkpoint_demo.py | 成功 | 実CPUモデル+mock GPUでwaiting_resumeを記録しGIF生成 |
| 0.4 releaseデーモンの実保存領域 | 成功 | 8787で起動、既存profileを維持、実CPUモデル・4-attemptモデルを完了、通常終了後に同じjob/log/historyを復元 |
| 0.5 releaseデーモンの実保存領域 | 成功 | 0.4履歴を維持して8787で起動、親/子/孫3 PIDを回収、4-attempt/200 stepのCPUモデル完了、cleanup_error=null |
| 公開0.5 snapshotの復元 | 成功 | 設定version切替・ソース・完了4件・profile・logs復元、全導入チェック再実行、実保存領域で3 PID回収とCPU 200 step完了 |
| CUDA/PyTorch probe | 未実行 | 実GPUと実PyTorchが必要 |

0件targetやdoc-testは成功件数に含めません。HTML/APIは検証しましたが、ブラウザJavaScript E2Eは未実施です。profile_smokeの報告値およびwrapper testのtorchモジュールは明示的なfixtureであり、実GPU測定ではありません。

## 比較シミュレーション

`cargo run --locked --example sim_benchmark` → [simulation-results.json](simulation-results.json)。8192 MiB、margin 512 MiB、4ジョブ各3000 MiB、実行時間3/2/4/1 tick。未管理合計12,000 MiBは予算超過、MLusはピーク予約6,000 MiBで6 tick、直列10 tick。人工的な固定時間モデルであり実性能ではありません。

`cargo run --locked --example feedback_benchmark` → [feedback-results.json](feedback-results.json)。申告量1000 MiB、合成footprint 5000 MiBの4ジョブを比較します。プロファイルなしでは同時合成footprint 20,000 MiB、校正値5000 MiBから学習した場合は予約5,128 MiBで1ジョブずつ実行し、合成footprint 5,000 MiB。これは申告不足を学ぶ配置ポリシーの検証です。実OOMの発生率を測定したものではなく、未管理側の不可能な並列実行を性能比較の基準にしてはいけません。合成tickでは学習側が遅くなるため、安全性と並列性のtradeoffも含めて記録しています。

## 環境公開後の確認

公開後の再接続環境でソース、Rust 1.90.0、ビルド出力、保存された導入/起動設定を確認し、0.1時点の単体8件と機能検証を再実行しました。ライブプロセスは再起動し、CPUモデルも再実行しました。その後0.2の公開versionへの再接続も確認し、ソース・保存プロファイル・単体12件・profile機能検証・ラッパー模擬5件を再検証しました。今回0.3の公開versionへの再接続を確認し、ソース・保存プロファイル・checkpointの復元と、Rust14件・checkpoint機能検証・ラッパー模擬6件を再検証しました。0.4はその復元環境で実装し、インストールスクリプト全体・デーモン再起動を検証しました。0.4単独のクラウド公開snapshotからの復元は未検証です。0.5の公開snapshotへの再接続を確認し、ソース・Rust・設定・完了ジョブ4件・プロファイル・ログ・checkpointを復元しました。保存済みの導入スクリプト全体（build/release、Rust14件、lint、5機能検証、wrapper模擬6件、2合成benchmark）を再実行して成功しました。復元した実保存領域でも親/子/孫の回収と4-attempt/200 stepのCPU学習を確認し、8787でデーモンを起動して検証後、通常終了し、完了履歴6件を保存しました。0.5は同じ作業環境で実装し、process group/subreaperが利用可能、cgroup書込みは不可であることを確認しました。

## 実機チェック手順 (未実施)

1. GPU型番/VRAM、driver/CUDA/PyTorch、OS、既存GPUプロセス、MIG/MPS設定を記録する。
2. 専用GPUで`examples/torch_probe.py`を直接起動し、256 MiB tensor生成・同期を確認する。
3. `mlus serve --state-dir PATH`を起動し、まずprofileなしでprobeを実行する。ログでvisible device/allocator peak/exitを確認する。
4. `--profile probe:256MiB:runtime-version`と`integrations/pytorch_profile.py`を使って同じprobeを実行する。報告値を直接取得したPyTorchカウンタと照合し、再実行時の予約量とGPU別profileを確認する。
5. model/seed/batch/shape/dtype固定の小型モデルを用意し、未管理並列・直列・MLus（未校正/校正済）をwarmup後5回以上測定する。同一の成功ジョブ数で比較し、OOM数、wall time、p50/p95待ち時間、peak VRAM、GPU利用率、CPU監視costを保存する。
6. `examples/torch_checkpoint_train.py`を`--cooperative`とラッパーで実行し、model/optimizer/RNGの復元と中断なしの結果を同条件で比較する。`nvidia-smi`で各ワーカー終了後のVRAM回収を実測する。これは未実施。
7. 申告不足、入力shape変化、外部GPU負荷の割込み、query失敗、報告欠落、OOM、再起動を試す。OOMはMLus管理中でも起こり得る。誘発は他の作業と分離したGPUで行う。
8. MPS/MIG比較は機種と公式制約を確認して既存ツールで構成し、元の構成を保存・復元する。ここでは自動実行しない。

生ログ、対象構成、測定回数、ばらつきが揃うまで性能改善・OOM回避保証・production readinessを主張しません。
