# 外部のモデル・コード・ライブラリ

この教材は、外部で学習・公開されたモデルとオープンソースのロボットシミュレーション環境を利用します。モデルをこの教材の作者や利用者が学習したものとして扱わないでください。

## 学習済みモデル

| 項目 | 内容 |
| --- | --- |
| 配布元 | robomimic / ARISE Initiative |
| モデル | Lift / Proficient-Human / BC-RNN / low-dim |
| 公式案内 | https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html |
| ファイル | `lift_ph_low_dim_epoch_1000_succ_100.pth` |
| ダウンロード元 | https://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth |
| SHA256 | `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` |
| ファイルサイズ | 7,959,022 bytes |

重みはリポジトリに同梱せず、実行時に公式配布元から取得します。ファイル名に含まれる成功率は元の実験での情報であり、この教材の条件・環境での成功率ではありません。

公式モデル集はrobomimic v0.1とrobosuiteの`offline_study`ブランチを指定しています。本教材は別の固定バージョン構成を利用します。使用バージョンは[READMEの「環境・依存関係」](README.md#環境依存関係)を参照してください。robomimic 0.3.0の方策復元機能を利用し、ロボット環境との接続は本教材のアダプターで行います。古い設定にない`algo.transformer.enabled=False`を補いますが、学習済み重みは変更しません。公式ベンチマークの厳密な再現を主張しません。

モデルの重みやデータの配布条件は、コードのライセンスと同じとは限りません。モデルを再配布・商用利用する場合は、配布元の条件を別途確認してください。本教材は上流モデルのライセンスを変更・付与するものではありません。

## 主な依存先

| プロジェクト | 使用目的 | 参照先 |
| --- | --- | --- |
| robomimic | チェックポイントの復元、学習済み方策による推論 | https://github.com/ARISE-Initiative/robomimic |
| robosuite | PandaアームとLift環境 | https://github.com/ARISE-Initiative/robosuite |
| MuJoCo | 物理シミュレーション | https://github.com/google-deepmind/mujoco |
| PyTorch / torchvision | ニューラルネットワークの実行 | https://pytorch.org/ |
| NumPy | 数値処理 | https://numpy.org/ |
| Matplotlib | 実験結果の図 | https://matplotlib.org/ |
| Pillow | 動画への文字表示、画像処理 | https://python-pillow.org/ |
| ImageIO / imageio-ffmpeg | 動画の書き出し・読み戻し | https://imageio.readthedocs.io/ |
| uv | Python環境と依存パッケージの導入 | https://docs.astral.sh/uv/ |

各依存先のライセンスは、そのプロジェクトの配布物・公式リポジトリで確認してください。依存パッケージをインストールすることと、そのコードを自分の作品として再配布することは別です。

## 教材で追加している部分

- ダウンロードするモデルの固定とハッシュ検証。
- 物体初期位置を変えた試行の実行。
- 高さと連続保持時間による独自の成功判定。
- CSV・動画・静止画・実行条件の保存。
- 位置ごとの集計と可視化。
- 動画の読み戻しと出力の検証。
- 日本語の実行手順と考察のための説明。

これらは学習済みモデルの再学習ではありません。提出時には、この教材を利用した部分と、利用者がさらに変更・検討した部分を区別して記載してください。
