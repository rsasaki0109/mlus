# Persistent jobs and restart recovery — MVP 0.5

## 保存内容

`--state-dir`（既定`.mlus-state`）内の`mock-jobs.json` / `cuda-jobs.json`にschema、backend、元の作業ディレクトリ、ログの絶対パス、`recovery_required`、ジョブ一覧を保存します。ジョブ一覧はcommand、申告量、優先度、profile key、状態、GPU UUID、履歴PID、exit、attempt、queue_ticket、受理済みcheckpointなどを含みます。環境変数は保存せず、新しいデーモンから継承します。コマンド引数には秘密情報を入れないでください。

ログとattempt別報告は`state-dir/runs/backend-PID-timestamp/`に保持します。ログは0600、生成ディレクトリは0700。元のcwdとログディレクトリが存在しない場合は起動を拒否します。ファイルを移動する場合は停止・バックアップして設定との整合を確認してください。0.3までのメモリ内ジョブは過去に遡って復元できません。

同じ状態ディレクトリのデーモンはOS lockで1つに制限します。mock/CUDAは別の台帳です。壊れたschema/backend/ID/状態を黙って消去しません。

## 保存と起動の順序

1. submitを検証し、`queued`を保存してからHTTP 200を返す。
2. placementを決め、`starting`とattempt・予約・GPU UUIDを保存する。
3. 子プロセスをspawnし、`running`とPIDを保存する。
4. leaderと同グループの子孫を終了・回収し、グループ消滅後に報告を検証する。profileを先に保存し、完了または`waiting_resume`を台帳に保存してから次のジョブを起動する。

毎回、一時ファイルをcreate_new/0600で作り、write+fsync、同じディレクトリでrename、ディレクトリfsyncを行います。`starting`も復旧確認が必要な状態です。spawn直前・直後の曖昧な区間でクラッシュしても、自動再実行を避けます。

32 MiB / 履歴込み10,000件のスナップショット方式です。更新ごとに全ジョブを直列化してfsyncするため、大量投入での性能・スケーラビリティは未評価です。ログ整理・履歴削除APIはありません。台帳とモデルprofileは別ファイルで、原子的な1トランザクションではありません。成功応答の喪失後に同じsubmitを再送すると別ジョブになるため、exactly-once/idempotencyは保証しません。

## 再起動時の扱い

| 保存状態 | 再起動後 |
|---|---|
| queued | 元の優先度・ticketで復元、起動可能なら開始 |
| waiting_resume | checkpointとattemptを復元、元のGPU UUIDへ再開 |
| starting / running | interruptedへ変換し、recovery_required=trueを保存。全体の新規起動を停止 |
| succeeded / failed / rejected / cancelled / interrupted | 履歴として維持。自動再試行なし |

`recovery_required`は次の再起動でも残ります。中断ジョブの保存予約は履歴であり、新しい実行の予約としては数えません。その代わり全体の新規起動を停止します。保存PIDは再利用され得るため、自動signal・実行継続判定に使いません。

SIGINT/SIGTERMでは、このデーモンが所有するジョブのグループをkill/waitして`cancelled`を保存します。未起動と受理済みcheckpoint待機は残します。回収を確認できない場合は`interrupted`・復旧待ちにします。グループ離脱した子孫や外部サービスまでの隔離は未対応です。[実行管理](processes.md)。クラウド公開はプロセス引継ぎではありません。公開前にジョブを完了させるか通常終了し、再接続後に状態を確認してください。

## 明示的な復旧

1. `mlus status`の`recovery_required`、中断ジョブ、GPUプロセス、ログを確認する。
2. OSのプロセス情報と実GPU監視を照合し、残存ワーカーがGPUを使わない状態にする。履歴PIDだけを信じてkillしない。
3. 確認後に`mlus recover --confirm-cleanup`を実行する（別ポートなら`--port N`）。
4. statusと待機ジョブの結果を確認する。中断ジョブは履歴に残り、自動再投入されない。

`POST /api/recover`もローカルJSON専用で、ボディは厳密に`{"confirm_cleanup":true}`が必要です。これはオペレータの明示的な確認であり、MLusが残存プロセスを検出・終了したことを意味しません。台帳を保存できなければ503で停止を解除しません。現在の子孫回収エラーやグループ外のlive adopted childがある間は409でrecoverを拒否します。

## 保存障害

`job_store_error`がある間は全体の新規起動を止めます。既に所有する子は監視を続けます。submit保存失敗は503と`id`・`state:queued`・`persistence:unconfirmed`を返します。rename後のfsync失敗などではディスクに残っている可能性があるので、失敗応答を「存在しない」と扱わずstatusと台帳を確認してください。障害を修正後、有効な状態変更時に再保存するか、`recover --confirm-cleanup`で明示的に再保存します。

破損台帳でデーモンが起動できない場合、台帳・ログ・profile・checkpointを先にバックアップして原因を調べます。台帳だけを削除するとジョブ追跡情報を失います。残存ワーカーを停止していない状態で新しいstate-dirを使うと二重起動の防止も失います。

## 検証

`python3 scripts/recovery_smoke.py`は実デーモン・実子プロセス、合成GPU、実ファイル障害で検証します。SIGKILL後の残存子はテストが起動した既知の子だけを終了します。実CUDAのメモリ回収・電源断やネットワークファイルシステムの耐久性は未検証です。
