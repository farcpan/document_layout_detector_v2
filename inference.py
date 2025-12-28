#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.detector import DocumentLayoutDetector
from src.utils import save_results_json, print_detection_summary


def main():
    parser = argparse.ArgumentParser(
        description='ドキュメントレイアウト検出ツール'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力画像パスまたはディレクトリ'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='YOLOモデルのパス（デフォルト: yolov8n.pt）'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='output',
        help='出力ディレクトリ（デフォルト: output）'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='使用デバイス（デフォルト: cpu）'
    )
    parser.add_argument(
        '-c', '--conf',
        type=float,
        default=0.25,
        help='信頼度閾値（デフォルト: 0.25）'
    )
    parser.add_argument(
        '-i', '--iou',
        type=float,
        default=0.45,
        help='IOU閾値（デフォルト: 0.45）'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='可視化画像を保存しない'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='検出結果をJSON形式で保存'
    )

    args = parser.parse_args()

    detector = DocumentLayoutDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    input_path = Path(args.input)

    if input_path.is_file():
        result = detector.detect(
            input_path,
            save_visualization=not args.no_visualization,
            output_dir=args.output_dir
        )
        print_detection_summary(result)

        if args.save_json:
            json_path = Path(args.output_dir) / f"{input_path.stem}_result.json"
            save_results_json(result, json_path)

    elif input_path.is_dir():
        results = detector.batch_detect(
            input_path,
            output_dir=args.output_dir
        )

        for result in results:
            print_detection_summary(result)

        if args.save_json:
            json_path = Path(args.output_dir) / "batch_results.json"
            save_results_json(results, json_path)

    else:
        raise ValueError(f"入力パスが無効です: {input_path}")

    print("処理が完了しました！")


if __name__ == '__main__':
    main()
