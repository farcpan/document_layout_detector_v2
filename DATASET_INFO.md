# データセット情報

ドキュメントレイアウト検出のための学習用データセットの詳細情報です。

## 推奨データセット

### 1. DocLayNet

**概要**
- 提供元: IBM Research
- ページ数: 80,863ページ
- アノテーション: 人間による高品質な注釈
- ドキュメント種別: 多様（金融、科学、特許、入札、法律、マニュアル）

**クラスラベル（11種類）**
1. Caption - キャプション
2. Footnote - 脚注
3. Formula - 数式
4. List-item - リスト項目
5. Page-footer - ページフッター
6. Page-header - ページヘッダー
7. Picture - 画像/図
8. Section-header - セクション見出し
9. Table - 表
10. Text - テキスト/段落
11. Title - タイトル

**ダウンロード**
- GitHub: https://github.com/DS4SD/DocLayNet
- 公式サイト: https://developer.ibm.com/data/doclaynet/
- ライセンス: CDLA-Permissive-2.0

**データ構造**
```
DocLayNet/
├── train/
│   ├── images/
│   └── annotations.json (COCO形式)
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

**使い方**
```bash
# ダウンロード後、YOLO形式に変換
python prepare_dataset.py \
  --dataset-type doclaynet \
  --dataset-root /path/to/DocLayNet \
  --output-dir datasets/doclaynet_yolo
```

### 2. PubLayNet

**概要**
- 提供元: IBM Research
- ページ数: 1,000,000以上
- ドキュメント種別: 医学論文（PubMed Central）
- アノテーション: 自動生成（XMLベース）

**クラスラベル（5種類）**
1. Text - テキスト
2. Title - タイトル
3. List - リスト
4. Table - 表
5. Figure - 図

**ダウンロード**
- GitHub: https://github.com/ibm-aur-nlp/PubLayNet
- ライセンス: CDLA-Permissive-1.0

**特徴**
- 大規模データセット（100万以上）
- 医学論文に特化
- 自動アノテーションのため、一部ノイズあり

**使い方**
PubLayNetはCOCO形式で提供されているため、prepare_dataset.pyで変換可能です。

### 3. PubTables-1M

**概要**
- 提供元: Microsoft
- ページ数: 1,000,000以上の表画像
- 特化領域: 表構造認識

**クラスラベル**
- Table
- Table Column
- Table Row
- Table Cell

**ダウンロード**
- HuggingFace: https://huggingface.co/datasets/bsmock/pubtables-1m

**使い方**
表の検出・認識に特化した学習に最適です。

## カスタムデータセットの作成

### 必要なツール

**アノテーションツール**
1. **LabelImg**
   - YOLO形式直接サポート
   - インストール: `pip install labelImg`
   - 起動: `labelImg`

2. **CVAT (Computer Vision Annotation Tool)**
   - Webベース
   - 複数人での協調作業可能
   - URL: https://www.cvat.ai/

3. **Roboflow**
   - Webベース
   - データ拡張機能あり
   - 形式変換が簡単
   - URL: https://roboflow.com/

### アノテーション形式

**YOLO形式（テキストファイル）**
```
<class_id> <x_center> <y_center> <width> <height>
```

すべての座標は0-1の範囲で正規化されています。

例:
```
0 0.5 0.3 0.4 0.2
1 0.5 0.7 0.6 0.3
```

### ディレクトリ構造

```
custom_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

### クラス定義例

```yaml
# dataset.yaml
path: /path/to/custom_dataset
train: images/train
val: images/val

nc: 3  # クラス数
names:
  0: text
  1: table
  2: figure
```

## データ拡張

学習時に自動的に以下のデータ拡張が適用されます：

- ランダム回転
- ランダム反転
- スケーリング
- 色調変換
- モザイク拡張
- MixUp

追加の拡張が必要な場合は、train.pyの設定を調整してください。

## ベストプラクティス

### データセットサイズ
- 最小: クラスあたり100-200画像
- 推奨: クラスあたり500-1000画像
- 理想: クラスあたり1000画像以上

### train/val/test分割
- Train: 70-80%
- Validation: 10-15%
- Test: 10-15%

### アノテーション品質
1. バウンディングボックスは対象をしっかり囲む
2. オブジェクトの端まで正確にマーク
3. 重なり合うオブジェクトも個別にアノテーション
4. 一貫したクラス分類基準

### 多様性の確保
- 異なるドキュメントスタイル
- 異なるフォント
- 異なるレイアウト
- 異なる画質/解像度
- 異なる言語（必要に応じて）

## トラブルシューティング

### データセット変換エラー
```bash
# COCOアノテーションの検証
python -c "import json; json.load(open('annotations.json'))"
```

### クラス不均衡
一部のクラスのサンプル数が極端に少ない場合：
- データ拡張を強化
- 少数クラスの重みを増加
- オーバーサンプリング

### メモリ不足
大規模データセットの場合：
- バッチサイズを減らす
- 画像サイズを小さくする
- ワーカー数を調整

## 参考資料

- [YOLO形式アノテーションガイド](https://docs.ultralytics.com/datasets/detect/)
- [DocLayNet論文](https://arxiv.org/abs/2206.01062)
- [PubLayNet論文](https://arxiv.org/abs/1908.07836)
