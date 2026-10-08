# Cooperative memory profiles (protocol v1)

このプロトコルは、信頼するジョブの自己申告を過去の観測として保存します。GPUドライバの制御API、CUDAアロケータの強制上限、プロセス間のメモリ共有APIではありません。

## キーと適用範囲

`submit --profile KEY`またはJSONの`profile`を指定します。キーは1..128 byte、制御文字なし。モデルversion、入力shape/batch、dtype、train/infer、フレームワークversion等を含め、異なる条件を混ぜないでください。自動のモデル同定や入力解析は行いません。

`KEY -> GPU UUID -> MemoryProfile`のmapを保持します。mockとCUDAは別ファイルに保存します。GPU UUIDが違う場合、サンプルを共有せず申告量に戻ります。同じGPUでもdriver/runtimeが変わった場合の互換性は保証しません。必要なら新しいキーを使って再校正します。

## 子プロセスへの契約

profile指定時、デーモンが以下を設定します。

- `MLUS_REPORT_PATH`: そのジョブ専用の最終報告JSONパス。
- `MLUS_PROFILE_KEY`: submitの明示的なキー。
- `MLUS_BACKEND`: `mock`または`nvidia`。

ワーカーと同グループの子孫の終了・回収確認後に、一度だけleaderの報告を読みます。途中の報告で実行中の予約を変更しません。子孫プロセスの報告やメモリ量を自動で集約しません。プロファイルなしの子ではこれらの変数を除去し、デーモン自身に継承された報告先を誤って流用しません。

```json
{
  "schema_version": 1,
  "measurement": "cuda",
  "outcome": "success",
  "peak_allocated_mib": 4200,
  "peak_reserved_mib": 4600
}
```

上は仕様例であり測定結果ではありません。mockでは`measurement: simulation`が必須です。`outcome`は`success / oom / error / checkpointed`。checkpointedは、オプトインしたワーカーのexit 75と有効なチェックポイント報告が確認できた場合にのみ受理します。successは子の正常終了と一致する必要があり、oom/errorは子の異常終了時にのみ受理します。全フィールド必須、未知のフィールドは拒否します。値は非負の整数MiB、allocated <= reserved <= 割り当てGPU容量が必要です。64 KiBを超える報告、通常ファイルでない報告、JSON不正、backend/version不一致を拒否します。

`integrations/mlus_report.py`は追加依存なしで一時ファイルからatomic replaceするhelperです。`examples/profile_simulation.py`は明示的な**合成値**を書き、mock以外で利用しようとすると失敗します。これは測定器ではありません。

不正/不足報告は`job.report_error`に記録します。アプリの成功を勝手に失敗扱いにはせず、`job.state`と`memory_report/report_error`を別に表示します。

## 予約量

GPU別に以下を適用します。

```
required = max(
    declared_mib,
    historical_peak_reserved_mib + 128 MiB,
    historical_oom_attempted_reservation_mib + 128 MiB
)
available = max(0, total - observed_used - gpu_margin - running_reservations)
```

未観測GPUではrequiredは申告量のみです。GPU marginは既定512 MiBで、128 MiBのプロファイル余裕とは別です。申告量は減らしません。過去ピークは小さな報告で減らしません。OOM時は失敗した予約量を不足需要の参考値として保持し、次の予約を少なくとも128 MiB増やします。これはOOM時の正確な不足バイト数の推定ではなく、再度OOMになる可能性があります。

実行中の予約は起動時に固定します。同じKEYの新しい報告が終了時に取り込まれたら、待機中ジョブをそのループで再評価します。全GPUの容量を超えれば`rejected`にし、実行されなかったことを表示します。新規submit時に既に超えている場合はHTTP 400です。

## 保存と障害

既定`.mlus-state`、または`serve --state-dir PATH`。`mock-profiles.json / cuda-profiles.json`にはschema/backendとprofile mapを保存します。ディレクトリにOS advisory lockを保持し、同じstate-dirは1デーモンのみ使用します。異なるportでも共有は拒否します。終了・クラッシュ時にlockはOSが解放します。

一時ファイルを書いてfsync、renameし、ディレクトリもfsyncします。新規ディレクトリ/ファイルは0700/0600。保存された内容が不正なら起動を失敗させ、黙って消去しません。デーモン停止後にバックアップして原因を調べてください。サイズ上限8 MiB。状態を消去して校正をやり直す場合も、既存の観測値は退避してください。

保存失敗時、観測値はメモリには残りますが永続化を保証できません。`profile_store_error`を表示し、profile指定ジョブの新規起動を止めます。既存ジョブとprofileなしのジョブは続行します。次の有効な終了時報告で保存を再試行します。状態ディレクトリの障害を修正後、必要なら静止したジョブ状態で再起動してください。

アプリのチェックポイントと終了時報告の契約は[checkpoints.md](checkpoints.md)を参照してください。0.3では各attemptに別の報告パスを用意し、前回のファイルを誤って再利用しません。

0.4ではジョブ履歴・待機列・attempt・checkpointを別のジョブ台帳へ保存します。受理済みの待機状態は復元できますが、クラッシュ後の子プロセスは引き継ぎません。起動途中/実行中の記録が残る場合は新規起動を止めます。[復旧契約](recovery.md)。

## PyTorchラッパーの制約

```bash
mlus submit --vram-mib 1024 --profile probe:256MiB:v1 -- python3 integrations/pytorch_profile.py examples/torch_probe.py --mib 256
```

`runpy`で既存Python scriptを`__main__`として実行し、引数、scriptの隣のmodule import、通常の標準出力と終了コードを維持します。`torch.cuda.max_memory_allocated / max_memory_reserved`をMiBへ切り上げて報告します。協調的なexit 75はcheckpointedとして数え、通常のアプリ異常終了とは区別します。CUDA OOMと通常エラーを区別します。mock、CUDAなし、可視GPUが1台でない場合には失敗し、合成値へ切り替えません。

PyTorch allocatorのカウンタにはCUDA context、NCCLや他のallocator、子プロセスの利用が含まれない場合があります。対象scriptが途中でpeakカウンタをresetした場合も正確な全体ピークになりません。ラッパーによるPyTorch/CUDA初期化・同期は元scriptの実行環境に影響し得ます。全コードに無変更で適用できる保証はありません。協調的な管理であり、測定値の真正性を別ユーザーから保証する認可機構ではありません。

現環境ではRust/HTTP/永続化は実動作で検証し、ラッパーは明示的なtorch API fixtureで検証しました。実PyTorch/CUDAは未検証です。
