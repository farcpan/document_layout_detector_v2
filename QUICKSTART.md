# クイックスタートガイド

最速でドキュメントレイアウト検出を始めるためのガイドです。

## 最小構成での実行（5分）

### 1. 環境構築

```bash
# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 推論の実行

```bash
# テスト用の画像を入力ディレクトリに配置
# （PDFやWordをPNG/JPEGに変換したもの）
cp /path/to/your/document.png input/

# 推論実行（初回はYOLOv8モデルが自動ダウンロードされます）
python inference.py input/document.png

# 結果は output/ ディレクトリに保存されます
```

これで完了です！output/ ディレクトリに検出結果が画像として保存されます。

## DocLayNet学習済みモデルの使用（推奨）

より高精度な検出には、DocLayNetで学習済みのモデルを使用します。

### 1. モデルのダウンロード

HuggingFaceから手動でダウンロード:
- https://huggingface.co/ppaanngggg/yolo-doclaynet

または、以下のコマンドで直接指定:

```bash
# HuggingFaceのモデルIDを直接指定
python inference.py input/document.png -m ppaanngggg/yolov8n-doclaynet
```

### 2. 推論の実行

```bash
# ダウンロードしたモデルを使用
python inference.py input/document.png -m models/yolov8n-doclaynet.pt --save-json

# 信頼度閾値を調整（デフォルト: 0.25）
python inference.py input/document.png -m models/yolov8n-doclaynet.pt -c 0.4
```

## Docker での実行

### 1. イメージのビルド

```bash
docker build -t document-layout-detector:latest .
```

### 2. 推論の実行

```bash
# 入力画像を準備
cp /path/to/document.png input/

# Dockerで実行
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  document-layout-detector:latest \
  /app/input/document.png -o /app/output
```

## カスタムモデルの学習（中級者向け）

### 1. データセットの準備

DocLayNetデータセットをダウンロード:
```bash
# DocLayNetの公式サイトからダウンロード
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
unzip DocLayNet_core.zip -d DocLayNet/

# YOLO形式に変換
python prepare_dataset.py \
  --dataset-type doclaynet \
  --dataset-root DocLayNet/ \
  --output-dir datasets/doclaynet_yolo
```

### 2. 学習の実行

```bash
# CPU環境での学習（遅い）
python train.py \
  --data datasets/doclaynet_yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 8 \
  --device cpu

# GPU環境での学習（推奨）
python train.py \
  --data datasets/doclaynet_yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --device 0
```

### 3. 学習済みモデルでの推論

```bash
# ベストモデルを使用
python inference.py input/document.png \
  -m runs/train/document_layout/weights/best.pt \
  --save-json
```

## よくある質問

### Q: どのモデルを使うべきですか？

**初心者・テスト用途:**
- `yolov8n.pt` - 最速、軽量、そこそこの精度

**本番環境:**
- `yolov8n-doclaynet.pt` - DocLayNetで学習済み、高精度
- `yolov8s-doclaynet.pt` - より高精度、少し遅い
- `yolov8m-doclaynet.pt` - 最高精度、さらに遅い

### Q: CPU で推論すると遅いです

1. より軽量なモデルを使用:
```bash
python inference.py input/document.png -m yolov8n.pt
```

2. 信頼度閾値を上げる:
```bash
python inference.py input/document.png -c 0.5
```

3. 画像サイズを小さくする（前処理で）

### Q: 特定のクラスだけ検出したい

推論後に結果をフィルタリングするスクリプトを追加できます:

```python
from src.detector import DocumentLayoutDetector

detector = DocumentLayoutDetector(model_path='models/best.pt')
results = detector.detect('input/document.png')

# 表だけフィルタリング
tables = [obj for obj in results['objects'] if obj['class'] == 'table']
print(f"検出された表の数: {len(tables)}")
```

### Q: 検出精度を上げるには？

1. より大きなモデルを使用（yolov8s, yolov8m）
2. DocLayNetで学習済みのモデルを使用
3. 自分のデータで追加学習（ファインチューニング）
4. 信頼度閾値を調整

### Q: 日本語文書でも使えますか？

はい、レイアウト検出は画像ベースなので言語に依存しません。
ただし、日本語文書特有のレイアウトが多い場合は、日本語文書でファインチューニングすることで精度が向上します。

## 次のステップ

- [README.md](README.md) - 詳細なドキュメント
- [DATASET_INFO.md](DATASET_INFO.md) - データセット情報
- [Ultralytics YOLOv8ドキュメント](https://docs.ultralytics.com/models/yolov8/)

## サポート

問題が発生した場合は、GitHubのIssueで報告してください。
