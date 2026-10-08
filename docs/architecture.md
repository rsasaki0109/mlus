# アーキテクチャ (MVP 0.5)

## 境界

```text
CLI / read-only dashboard
          | loopback HTTP
     single-thread daemon
          | submit validation / priority queue
     admission policy (Rust library)
          | GPU UUID + advisory lifetime reservation
     Linux child process / CUDA_VISIBLE_DEVICES
          |                     ^
     PyTorch etc.          NVIDIA telemetry
```

`src/lib.rs`の`Backend`、CSVパーサ、GPU選択、待機列順序はHTTPやLinux子プロセスから分離しています。`src/checkpoint.rs`は協調的handoffの検証と再開要求を担当します。`src/profiles.rs`は報告契約・GPU別需要推定・プロファイルのatomic保存とOS lockを担当します。`src/journal.rs`はジョブのatomicスナップショット・保存データの検証・異常終了検出を担当します。`src/processes.rs`はLinuxプロセスグループ・subreaper・子孫回収を担当します。`src/main.rs`は実行・終了回収・予約の状態遷移とHTTPを担当します。AMD向けバックエンドを追加する際はデバイス可視性設定も分離する必要があります。独立OS化にはこのポリシー部分の移植が候補になりますが、現時点でno_std対応やカーネルABIは設計・保証しません。

## 起動制御

GPUごとの有効空き容量:

`max(0, total - observed_used - margin - running_and_starting_reservations)`

実装は段階ごとに飽和減算し、加算も飽和して桁あふれを避けます。プロファイル指定時はGPU別の観測ピーク/OOM時予約量に128 MiBを加え、申告量との最大値を需要にします。この需要が有効空き以下のGPUから残容量が最小になるものを選びます。spawn前に起動意図と予約を保存し、同じイベントループ内で起動するため、このデーモン内では同じ容量を同時に予約しません。起動失敗、またはワーカー終了後のグループ全体の終了・回収確認で予約を解放します。監視失敗・終了確認失敗時には新規起動を止めます。既存GPUサンプルを表示する場合は`telemetry_error`が付くので最新情報とは扱えません。

観測使用量には管理ジョブの実使用も入ります。予約全体をさらに引くことで、その分を二重計上します。これは初期MVPの明示的な安全側の選択です。プロセス別使用量は表示しますが、子孫・共有コンテキスト・MPSによる帰属の不確実性があるため予約から差し引きません。

優先度は大きい値が先、同値はqueue_ticket順（初回は投入順、checkpoint再開は同優先度の末尾）です。容量が足りない先頭ジョブを飛ばして入るジョブを起動します。実行中のプリエンプションや優先度のagingはなく、継続的な高優先度投入では飢餓が発生し得ます。

## ライフサイクル

`queued -> starting -> running -> succeeded / failed`。協調的ワーカーは`running -> waiting_resume -> running`も可能です。待機中にプロファイルが更新され全GPU容量を超えた場合は`queued -> rejected`。spawn失敗は`queued -> failed`。申告量またはプロファイルからの需要が全GPUの物理容量から余裕を引いた上限を超える要求はHTTP 400で拒否します。GPUドライバが起動時に利用不能なら起動を失敗させ、黙ってモックへ切り替えません。モックは`--mock`でのみ有効です。

100 msのHTTP待機タイムアウトを挟み、各ループでGPUと子プロセスを観測します。nvidia-smiの起動時間があるため100 ms周期の保証はありません。HTTPリクエストボディの読込は同期的であり、信頼できないクライアントの遅延送信対策は未実装です。信頼したローカル開発用途に限定します。

ジョブ数は履歴込み10,000件、リクエストボディは64 KiBまで。永続スナップショットは32 MiBまで。長期運用では履歴整理、ログrotation、サブプロセス/cgroup管理、認証済みIPC、nvidia-smiタイムアウトが必要です。プロセスのPIDは識別・表示に使用し、GPU制御やメモリ移動に使いません。

## 協調的メモリ管理

0.4ではモデル/入力条件の明示的なキーとGPU UUIDごとに、終了時のピーク・成功/失敗/OOMの報告を受け取り、次回の需要へ反映します。実行中の予約を変えるAPIではありません。報告は子のexit status、backend、schema、GPU容量と照合します。不正/不足報告は`report_error`に記録し、学習しません。正常な子のexit statusと報告の可否は別の状態として保持します。

モデルプロファイルを`.mlus-state`（`--state-dir`指定可）へatomic rename+fsyncで保存し、同じディレクトリは1デーモンにOS lockで制限します。mock/CUDAは別ファイル。保存失敗はAPIに表示し、プロファイルを使う待機ジョブの新規起動を止めます。プロファイルなしのジョブは独立して続行できます。次に有効な報告を受け取ると保存を再試行します。復旧後の再起動でもプロファイルの読込を検証してください。

観測最大値とOOM時の予約量は将来の上限保証ではありません。入力・フレームワーク・ハードウェア差をキーに反映し、実機で妥当性を検証する必要があります。checkpointはジョブ自身が明示的な安全点で保存・終了し、デーモンが再起動を制御します。32回のワーカー起動を上限とし、再開は同じGPU UUIDに固定します。priorityは維持し、同優先度の再開ジョブは待機列の末尾へ置きます。チェックポイントファイルの内容、RNG/optimizerの保存・復元はアプリの責任です。CPU offloadは今後の検討事項です。任意のCUDAプロセスを停止してVRAMを開放したり、SIGSTOPでGPU予約を解放できるとは仮定しません。[プロトコル詳細](profiling.md)。

## 永続ジョブ台帳と復旧

`.mlus-state/{mock,cuda}-jobs.json`にジョブ・再開状態・元のcwd・ログディレクトリを保存します。ProfileStoreが取得した同じディレクトリのOS lockの下で、32 MiB/10,000件のスナップショットをatomic rename+fsyncで更新します。HTTP 200を返す前、spawn前の`starting`、spawn後の`running`、終了/handoff後を保存します。プロファイルは先に保存し、ジョブ台帳は次に保存します。両ファイルは単一トランザクションではありません。

復元時の`starting`/`running`は`interrupted`にし、`recovery_required`を永続化して新規起動を止めます。保存されたPIDを使ってsignalを送りません。明示的な`recover --confirm-cleanup`で停止を解除しても、中断したジョブは再試行しません。正常終了時は所有するグループをkill/waitして`cancelled`を保存し、未起動・checkpoint待機は残します。詳細は[recovery.md](recovery.md)。

## 子孫の実行管理

spawnで`process_group(0)`を指定し、Linux subreaperとして孤児になった同グループの子孫を回収します。leaderの終了をwaitid/WNOWAITで観測し、leader PIDを回収する前にグループへSIGKILLを送ります。leader・引き取った子孫をreapし、グループの消滅を確認してから完了/handoffへ遷移します。待っている間もジョブはrunningで予約を保持します。回収失敗・timeoutは新規起動を止めます。

グループ外へ離れたlive adopted childを検出すると`recovery_required`を永続化します。これらに自動signalは送りません。詳細と境界は[processes.md](processes.md)。cgroupが書込み可能でない現在の環境では、透過的な全子孫隔離は行いません。
