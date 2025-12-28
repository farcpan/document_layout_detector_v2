#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO


def download_pretrained_model(model_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"学習済みモデルをダウンロード中: {model_name}")

    model = YOLO(model_name)

    output_path = output_dir / model_name
    print(f"保存先: {output_path}")

    print(f"\nモデル情報:")
    print(f"  モデル名: {model_name}")
    print(f"  タスク: {model.task}")

    if hasattr(model, 'names'):
        print(f"  クラス数: {len(model.names)}")
        print(f"  クラス名: {list(model.names.values())}")

    print(f"\nダウンロード完了！")


def main():
    parser = argparse.ArgumentParser(
        description='学習済みYOLOモデルのダウンロード'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='モデル名（例: yolov8n.pt, yolov8s.pt, yolov8m.pt）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='出力ディレクトリ（デフォルト: models）'
    )

    args = parser.parse_args()

    download_pretrained_model(args.model, Path(args.output_dir))

    print("\n推奨モデル:")
    print("  - yolov8n.pt: 最軽量、最速")
    print("  - yolov8s.pt: 小型、高速")
    print("  - yolov8m.pt: 中型、バランス")
    print("  - yolov8l.pt: 大型、高精度")
    print("  - yolov8x.pt: 最大、最高精度")


if __name__ == '__main__':
    main()
