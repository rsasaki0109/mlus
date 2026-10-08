# Linux worker lifecycle — MVP 0.5

## 対象と方式

MLusは同一ユーザーの協調的なジョブを専用プロセスグループで起動します。通常のfork/subprocessで子・孫はグループを継承します。デーモンはLinuxの`PR_SET_CHILD_SUBREAPER`を有効にし、親を失った子孫を引き取って回収します。管理の対象は、このデーモンが起動した現在のジョブです。台帳から読み込んだPID/PGIDにsignalは送りません。

leaderだけの終了でVRAM予約を返すと、子孫が使っているGPUへ別ジョブを重ねてしまう可能性があります。0.5ではグループが消えるまで予約を保持し、終了とcheckpoint handoffに同じ確認を使います。プロセスグループとsubreaperは既存Linux機能で、独自のGPU隔離技術とは扱いません。

## 終了順序

1. ワーカー本体を`waitid(WNOWAIT)`で観測する。まだPIDを回収しない。
2. ジョブのプロセスグループへSIGKILLを送り、残存メンバーを終了させる。
3. leaderの終了コードを取得し、引き取った同グループの子孫を`waitpid`で回収する。
4. グループ消滅を確認してから完了またはhandoffを検証・保存し、予約を返す。

待ち時間中はジョブの状態を`running`のまま保持します。`process_management.workers`の`leader_exited`と`cleanup_pending`で確認できます。leader終了コードをジョブ結果に使います。残存子孫を強制終了したことは、それらの処理結果が正しいことの保証ではありません。アプリはcheckpoint前・正常完了前に自身の子をjoin・終了し、必要な状態を保存してください。子孫のメモリピークはleaderのPyTorch報告へ自動集約しません。

SIGINT/SIGTERMを受けたデーモンは、すべての所有グループへ先にSIGKILLを送り、共通の3秒の待機枠で回収します。成功したジョブを`cancelled`として保存します。kill/wait/query失敗や回収timeoutでは、完了確認できない予約を成功扱いせず、復旧待ちにします。通常のエラーでデーモンを抜ける場合もDrop guardで同じグループcleanupを試みます。SIGKILLやOSクラッシュでguardが実行されない場合は、永続台帳の復旧契約に従います。

## グループを離れた子孫

`setsid`、`setpgid`、外部サービスへの依頼などはグループによる管理を外れます。Linuxの現在のadopted child一覧で、既知のグループ外にlive childがあれば、`process_management.escaped_children`と`cleanup_error`に表示し、`recovery_required=true`を保存して全体の新規起動を止めます。これらへ自動signalは送りません。

親がまだ生きていて引き取られていない子孫、外部サービス、別PID namespaceなどを網羅的に検出する仕組みではありません。cgroupによる封じ込めやマルチユーザーのセキュリティ境界ではありません。現在のクラウド環境ではcgroupディレクトリに書込み権限がなく、必要な委譲を設定したホストでのcgroup backendは今後の課題です。

live adopted childや現在の回収エラーが残っている間、`recover --confirm-cleanup`はHTTP 409で拒否します。実際のプロセスを確認・終了してから復旧してください。残存子が終了した場合はデーモンが回収しますが、永続的な復旧待ちの解除には明示的なrecoverが必要です。[復旧手順](recovery.md)。

## 起動条件と制約

Linux subreaperとprocfsのプロセス情報が使えることを起動前に確認します。このクラウド環境のカーネルには`task/PID/children`がないため、同一UIDの`/proc/PID/stat`の親PIDから直接の子一覧を取得します。このfallbackはプロセス一覧を走査するため、プロセスが多いホストでの監視コストは未評価です。参照失敗は新規起動を停止します。

同一UIDでsignal可能な、グループを離れないジョブが対象です。権限変更・強制隔離・実行時間制限・ジョブ取消APIは未対応です。SIGKILLによる子孫終了ではアプリ側のfinallyや共有メモリ等の後片付けが実行されないため、アプリが通常終了する責任は残ります。GPUドライバがメモリを即時回収する保証もなく、引き続き実GPUの空き容量を監視します。

`python3 scripts/process_smoke.py`は実子/孫プロセスと実ファイルを使って、回収前に次ジョブが起動しないこと、checkpoint、通常停止、fatal-error cleanup、setsid離脱と復旧拒否を確認します。GPUはモックです。実CUDA/PyTorch DataLoader、回収不能なD-state、cgroup委譲環境は未検証です。
