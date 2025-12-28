# Document Layout Detector V2

YOLOv8を使用したドキュメントレイアウト検出ツール。PDFやWordなどのドキュメント画像から、段落、表、画像などの構造要素を自動検出します。

## 特徴

- YOLOv8ベースのリアルタイム検出
- CPU推論に対応（GPUも利用可能）
- Dockerコンテナでの簡単デプロイ
- カスタムデータセットでの学習機能
- バッチ処理対応

## 検出可能な要素

- テキスト/段落（Text, Paragraph）
- 表（Table）
- 図・画像（Figure, Picture）
- タイトル（Title, Section-header）
- リスト（List, List-item）
- キャプション（Caption）
- その他（Footnote, Formula, Page-header, Page-footer）

## プロジェクト構造

```
document_layout_detector_v2/
├── src/                      # ソースコード
│   ├── detector.py          # メイン検出クラス
│   └── utils.py             # ユーティリティ関数
├── inference.py             # 推論スクリプト
├── train.py                 # 学習スクリプト
├── prepare_dataset.py       # データセット準備スクリプト
├── download_pretrained.py   # 学習済みモデルダウンロード
├── requirements.txt         # 推論用依存関係
├── requirements-training.txt # 学習用依存関係
├── Dockerfile               # Docker設定
├── docker-compose.yml       # Docker Compose設定
├── models/                  # モデル保存先
├── input/                   # 入力画像
└── output/                  # 出力結果
```

## インストール

### ローカル環境（WSL2/Ubuntu）

```bash
# リポジトリのクローン
git clone <repository-url>
cd document_layout_detector_v2

# 推論用依存関係のインストール
pip install -r requirements.txt

# 学習も行う場合
pip install -r requirements-training.txt
```

### Docker環境

```bash
# Dockerイメージのビルド
docker build -t document-layout-detector:latest .

# または docker-compose を使用
docker-compose build
```

## 使い方

### 1. 推論（ローカル）

#### 単一画像の処理

```bash
python inference.py input/document.png

# カスタムモデルを使用
python inference.py input/document.png -m models/best.pt

# 信頼度閾値を調整
python inference.py input/document.png -c 0.5

# JSON出力も保存
python inference.py input/document.png --save-json
```

#### バッチ処理

```bash
# ディレクトリ内の全画像を処理
python inference.py input/ -o output/
```

#### オプション

- `-m, --model`: モデルパス（デフォルト: yolov8n.pt）
- `-o, --output-dir`: 出力ディレクトリ（デフォルト: output）
- `-d, --device`: デバイス（cpu/cuda/mps、デフォルト: cpu）
- `-c, --conf`: 信頼度閾値（デフォルト: 0.25）
- `-i, --iou`: IOU閾値（デフォルト: 0.45）
- `--no-visualization`: 可視化画像を保存しない
- `--save-json`: 検出結果をJSON形式で保存

### 2. 推論（Docker）

```bash
# 単一画像
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  document-layout-detector:latest \
  /app/input/document.png -o /app/output

# バッチ処理
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  document-layout-detector:latest \
  /app/input -o /app/output

# docker-compose を使用
docker-compose up
```

### 3. モデルの学習

#### 学習済みモデルのダウンロード（ベースモデル）

```bash
python download_pretrained.py --model yolov8n.pt --output-dir models/
```

利用可能なモデル:
- `yolov8n.pt`: 最軽量、最速（推奨：まずはこれから）
- `yolov8s.pt`: 小型、高速
- `yolov8m.pt`: 中型、バランス
- `yolov8l.pt`: 大型、高精度
- `yolov8x.pt`: 最大、最高精度

#### DocLayNetデータセットの準備

```bash
# DocLayNetデータセットをYOLO形式に変換
python prepare_dataset.py \
  --dataset-type doclaynet \
  --dataset-root /path/to/doclaynet \
  --output-dir datasets/doclaynet_yolo
```

#### カスタムデータセットの準備

```bash
# カスタムデータセットの設定ファイルを作成
python prepare_dataset.py \
  --dataset-type custom \
  --output-dir datasets/custom \
  --class-names text table figure paragraph
```

その後、以下のように画像とアノテーションを配置:
```
datasets/custom/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### 学習の実行

```bash
# 基本的な学習
python train.py \
  --data datasets/doclaynet_yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16

# GPUを使用（推奨）
python train.py \
  --data datasets/doclaynet_yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --device 0

# 詳細設定
python train.py \
  --data datasets/doclaynet_yolo/dataset.yaml \
  --model yolov8m.pt \
  --epochs 200 \
  --batch 32 \
  --imgsz 1280 \
  --device 0 \
  --patience 50 \
  --optimizer AdamW \
  --lr0 0.001 \
  --cos-lr
```

#### 学習オプション

- `--data`: データセットYAMLファイル（必須）
- `--model`: ベースモデル（デフォルト: yolov8n.pt）
- `--epochs`: エポック数（デフォルト: 100）
- `--batch`: バッチサイズ（デフォルト: 16）
- `--imgsz`: 画像サイズ（デフォルト: 640）
- `--device`: デバイス（0: GPU, cpu: CPU）
- `--patience`: Early Stopping待機エポック数（デフォルト: 50）
- `--optimizer`: オプティマイザー（SGD/Adam/AdamW/auto）
- `--lr0`: 初期学習率（デフォルト: 0.01）
- `--lrf`: 最終学習率係数（デフォルト: 0.01）
- `--cos-lr`: コサインアニーリング学習率スケジューラー

### 4. 学習済みモデルの使用

学習完了後、ベストモデルを使用して推論:

```bash
python inference.py input/document.png \
  -m runs/train/document_layout/weights/best.pt \
  -o output/ \
  --save-json
```

## 技術詳細

### アーキテクチャ

- **検出モデル**: YOLOv8（Ultralytics）
- **フレームワーク**: PyTorch
- **推論**: CPU/GPU対応
- **入力形式**: PNG, JPEG, BMP, TIFF

### 推奨する学習済みモデル

プロジェクトではDocLayNetで学習済みのYOLOv8モデルの使用を推奨します:

1. **ppaanngggg/yolo-doclaynet**（HuggingFace）
   - `yolov8n-doclaynet`
   - `yolov8s-doclaynet`
   - `yolov8m-doclaynet`

2. **DocLayout-YOLO**（opendatalab）
   - 最新のYOLO-v10ベースモデル

使用例:
```bash
# HuggingFaceからダウンロード（手動）
# https://huggingface.co/ppaanngggg/yolo-doclaynet

# モデルを使用
python inference.py input/document.png -m models/yolov8n-doclaynet.pt
```

### データセット

#### DocLayNet
- 80,863ページ
- 11クラスラベル
- 多様なドキュメント（金融、科学、特許、法律など）
- ダウンロード: https://github.com/DS4SD/DocLayNet

#### PubLayNet
- 100万以上のページ
- 主に医学論文
- ダウンロード: https://github.com/ibm-aur-nlp/PubLayNet

## パフォーマンス

### ベンチマーク（DocLayNet）

| モデル | mAP50-95 | mAP50 | 推論速度（CPU）|
|--------|----------|-------|----------------|
| YOLOv8n | 0.75+ | 0.88+ | ~200ms/画像 |
| YOLOv8s | 0.77+ | 0.90+ | ~300ms/画像 |
| YOLOv8m | 0.78+ | 0.91+ | ~500ms/画像 |

## トラブルシューティング

### GPU学習時のメモリ不足

```bash
# バッチサイズを減らす
python train.py --data dataset.yaml --batch 8

# 画像サイズを小さくする
python train.py --data dataset.yaml --imgsz 480
```

### CPU推論が遅い

```bash
# より軽量なモデルを使用
python inference.py input.png -m yolov8n.pt

# 信頼度閾値を上げて検出数を減らす
python inference.py input.png -c 0.5
```

## 参考文献・リソース

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DocLayNet Dataset](https://github.com/DS4SD/DocLayNet)
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [ppaanngggg/yolo-doclaynet](https://github.com/ppaanngggg/yolo-doclaynet)

## ライセンス

このプロジェクトは[LICENSE](LICENSE)ファイルに記載されたライセンスの下で公開されています。

## 貢献

Issue、Pull Requestを歓迎します。
