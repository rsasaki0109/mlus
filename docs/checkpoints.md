# Cooperative checkpoint restart — MVP 0.5

アプリが安全な境界で状態を保存し、ワーカーが終了してからGPU予約を返す契約です。SIGSTOP、CUDAメモリの透過的移動、外部プロセスの強制checkpointは行いません。

## 実行と再開

1. `mlus submit --cooperative --vram-mib N -- COMMAND`でオプトインする。
2. アプリがmodel/optimizer/RNG/step等を自身のファイルへ保存する。
3. `yield_checkpoint(path, step=..., resume_vram_mib=...)`でhandoff報告を書き、exit 75で終了する。
4. デーモンが子の終了を回収し、報告を検証して`waiting_resume`へ移す。この時点で初めて予約を返す。
5. 優先度と待機順に従って同じGPU UUIDで再予約し、同じcommandを新しいワーカーとして起動する。
6. アプリは`MLUS_CHECKPOINT_PATH`から自分の状態を読み、続きから実行する。

通常のcommandは変更なく従来どおり実行できます。協調的な再開にはアプリの変更が必要です。コードを包むだけで任意の学習状態を取り出せるとは仮定しません。

## Python API

`integrations/mlus_checkpoint.py`は追加依存なし。

```python
from mlus_checkpoint import atomic_json_checkpoint, yield_checkpoint

# Application-owned CPU/JSON state; specification example, not GPU measurement.
path = atomic_json_checkpoint('/absolute/path/to/application-state.json', state)
yield_checkpoint(path, step=completed_step, resume_vram_mib=7000)
```

tensorの保存にはアプリ側で`torch.save`等を使い、optimizer/RNGも含めてください。`yield_checkpoint`は既に保存されたファイルを報告するだけです。ヘルパーのSystemExitを捕捉してGPU利用を続けないでください。仮に報告だけを書いて生存しても、デーモンは終了前に予約を返しません。

通常のfinally処理やPyTorchラッパーによる最終報告は実行されます。exit 75は、オプトインと有効な報告が両方ある時だけcheckpoint終了として扱います。報告欠落/不正ならfailed、非オプトインなら通常のアプリ失敗です。

## ワーカー環境変数

- `MLUS_COOPERATIVE=1`
- `MLUS_BACKEND=mock / nvidia`
- `MLUS_ATTEMPT`: 1から始まる今回の起動番号。
- `MLUS_CHECKPOINT_REPORT_PATH`: attempt固有のhandoff JSONパス。
- `MLUS_CHECKPOINT_PATH`: 再開時のみ、前ワーカーのチェックポイントファイル。

profile指定時の`MLUS_REPORT_PATH`もattemptごとに異なります。memory reportの`outcome: checkpointed`は、有効なhandoffと一致する場合だけ学習します。プロファイルには`checkpoint_samples`を記録し、通常エラーと混同しません。以前の保存プロファイルにこのカウンタがなければ0として読めます。

## Handoff JSON v1

```json
{
  "schema_version": 1,
  "measurement": "simulation",
  "attempt": 1,
  "step": 50,
  "checkpoint_path": "/absolute/path/to/model.json",
  "resume_vram_mib": 7000
}
```

上は仕様例です。CUDA backendではmeasurement=cuda。未知のフィールド、64 KiB超、通常ファイルでない報告を拒否します。attemptは終了したワーカーと一致、stepは前handoffより大きく、resume VRAMは正の整数、checkpoint pathは絶対パスかつ4096 byte以下の既存・読み取り可能な通常ファイルが必要です。内容の意味やモデル状態の完全性はアプリが検証します。

ワーカー起動回数は1ジョブ最大32回。32回目で再びyieldするとfailedにし、無限restartを防ぎます。失敗時もアプリの保存ファイルは削除しません。

## 配置と予約

再開は前のGPU UUIDに固定します。CPUにcheckpointを保存しても、任意GPUへ安全にmigrationできるとは仮定しません。対象UUIDが一時的に見えない場合は待機し、別GPUへ切り替えません。

`resume_vram_mib`はアプリが明示的に指定する次のphaseの申告量です。元のsubmit申告量は履歴に保存したまま、再開時はこの新しい要求と同GPUの過去profileを使います。プロファイルがある場合は過去ピーク/OOM余裕で保守的に増やします。容量を超える再開要求はrejected、再開前にcheckpointファイルが消えた場合はfailedです。

priorityは維持。同優先度ではyieldしたジョブを待機列の末尾へ移します。高優先度ジョブは次の安全境界まで待つ可能性があり、強制的なpreemptionや最大待ち時間を保証しません。新しいPIDでもjob IDは同じで、attempt番号と最新のhandoff stepをstatus/dashboardに表示します。job-ID.logは全attemptの出力を追記します。

GPU予約を返すことと、物理VRAMが即時に完全回収されることは同義ではありません。実GPUでは観測空き容量も引き続き確認します。0.5では同じグループの子孫の終了・回収まで待ってから予約を返します。グループを離れる子孫、外部allocator、MPS等に残る利用はこの契約で透過的に解放できません。[実行管理の境界](processes.md)。

## 保存・復旧の範囲

0.4ではcommand、queue、job ID、attempt、受理済みcheckpoint待機を永続ジョブ台帳へ保存します。同じ状態ディレクトリでデーモンを再起動すると待機を復元し、同じGPU UUIDへ再開します。実行中/起動途中の記録が残る場合は全体の新規起動を止め、残存ワーカーを確認後に明示的なrecoverが必要です。報告ファイルを書いただけではhandoffは受理済みになりません。終了確認・検証・台帳保存まで完了したcheckpointだけを自動再開します。[復旧契約](recovery.md)。新規exampleは既存のcheckpointを上書きしないため拒否します。

## 検証済み / 未検証

`checkpoint_smoke.py`では実HTTP・実プロセスで、報告だけでは予約が返らないこと、終了回収後の優先度制御、同GPU再開、無効handoff拒否を検証しました。CPU線形回帰を4回の起動に分け、50/100/150/200 stepの継続と、中断なしの重み・バイアスとの一致を確認しました。

`torch_checkpoint_train.py`はCUDA向けの小型モデル例です。model/optimizerとCPU/CUDA RNGを保存し、次のワーカーで復元します。実PyTorch/CUDAでは未実行です。ラッパーのcheckpoint分類は明示的なtorch API fixtureで検証しています。実GPUのVRAM回収、CUDA correctness、性能改善、CPU offloadは未検証です。
