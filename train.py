#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8モデルの学習'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='データセットYAMLファイルのパス'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='ベースモデル（デフォルト: yolov8n.pt）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='エポック数（デフォルト: 100）'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='バッチサイズ（デフォルト: 16）'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='画像サイズ（デフォルト: 640）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='デバイス（デフォルト: 0 [GPU], cpu も指定可能）'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='プロジェクトディレクトリ（デフォルト: runs/train）'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='document_layout',
        help='実行名（デフォルト: document_layout）'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early Stoppingの待機エポック数（デフォルト: 50）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='データローダーのワーカー数（デフォルト: 8）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='前回の学習から再開'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='事前学習済み重みを使用（デフォルト: True）'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='オプティマイザー（デフォルト: auto）'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='初期学習率（デフォルト: 0.01）'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='最終学習率係数（デフォルト: 0.01）'
    )
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        help='コサインアニーリング学習率スケジューラーを使用'
    )

    args = parser.parse_args()

    print("="*50)
    print("YOLOv8 ドキュメントレイアウト検出モデル学習")
    print("="*50)
    print(f"データセット: {args.data}")
    print(f"ベースモデル: {args.model}")
    print(f"デバイス: {args.device}")
    print(f"エポック数: {args.epochs}")
    print(f"バッチサイズ: {args.batch}")
    print(f"画像サイズ: {args.imgsz}")
    print("="*50)

    if args.device != 'cpu':
        if not torch.cuda.is_available():
            print("警告: CUDAが利用できません。CPUを使用します。")
            args.device = 'cpu'
        else:
            print(f"GPU情報: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )

    print("\n" + "="*50)
    print("学習完了！")
    print("="*50)
    print(f"最良モデル: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"最終モデル: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    print("="*50)

    print("\n評価を実行中...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")


if __name__ == '__main__':
    main()
