## group6の評価関数

### 前準備
まずチェックポイント用のディレクトリを作成し、提出したチェックポイントを配置します。

```bash
mkdir checkpoints
cd checkpoints
pwd
# 出力例: .../SimplerEnv_group6/checkpoints
````

（ここに提出したチェックポイントを置く）

### 評価関数の実行

プロジェクトのルートに戻り、評価を実行します。

```bash
cd ..
pwd
# 出力例: .../SimplerEnv_group6/
```

にいることを確認してください。

#### google\_robot を評価する場合

例: `fractal_checkpoint-200000/` を配置していた場合

```bash
python scripts/gr00t/evaluate_fractal.py --ckpt-path ./checkpoints/fractal_checkpoint-200000/
```

#### widowx を評価する場合

例: `bridge_checkpoint-200000/` を配置していた場合

```bash
python scripts/gr00t/evaluate_fractal.py --ckpt-path ./checkpoints/bridge_checkpoint-200000/
```

### 補足

* `simpler_env/policies/gr00t/gr00t_model.py` 内の
  `data_config` と `embodiment_tag` は、**学習時と同じ設定**にする必要があります。