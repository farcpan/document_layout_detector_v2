#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm


def convert_coco_to_yolo(
    coco_annotation: Dict,
    image_width: int,
    image_height: int,
    category_id_to_index: Dict[int, int]
) -> str:
    category_index = category_id_to_index[coco_annotation['category_id']]

    bbox = coco_annotation['bbox']
    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    return f"{category_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def prepare_doclaynet_dataset(
    dataset_root: Path,
    output_dir: Path,
    splits: List[str] = ['train', 'val', 'test']
):
    print("DocLayNetデータセットをYOLO形式に変換中...")

    output_dir.mkdir(parents=True, exist_ok=True)

    category_names = [
        'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
        'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
    ]

    for split in splits:
        print(f"\n{split}セットを処理中...")

        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"警告: {split_dir}が見つかりません。スキップします。")
            continue

        images_output_dir = output_dir / 'images' / split
        labels_output_dir = output_dir / 'labels' / split
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)

        annotation_file = split_dir / f'{split}_annotations.json'
        if not annotation_file.exists():
            annotation_file = split_dir / 'annotations.json'

        if not annotation_file.exists():
            print(f"警告: {annotation_file}が見つかりません。スキップします。")
            continue

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        category_id_to_index = {
            cat['id']: idx for idx, cat in enumerate(coco_data['categories'])
        }

        image_id_to_info = {img['id']: img for img in coco_data['images']}

        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        for image_id, image_info in tqdm(image_id_to_info.items(), desc=split):
            image_filename = image_info['file_name']
            image_path = split_dir / image_filename

            if not image_path.exists():
                continue

            shutil.copy(image_path, images_output_dir / image_filename)

            label_filename = Path(image_filename).stem + '.txt'
            label_path = labels_output_dir / label_filename

            yolo_annotations = []
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    yolo_line = convert_coco_to_yolo(
                        ann,
                        image_info['width'],
                        image_info['height'],
                        category_id_to_index
                    )
                    yolo_annotations.append(yolo_line)

            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    dataset_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(category_names),
        'names': category_names
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\nデータセット変換完了！")
    print(f"設定ファイル: {yaml_path}")


def prepare_custom_dataset(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    class_names: List[str]
):
    print("カスタムデータセットをYOLO形式に準備中...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        images_output_dir = output_dir / 'images' / split
        labels_output_dir = output_dir / 'labels' / split
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\nカスタムデータセット準備完了！")
    print(f"設定ファイル: {yaml_path}")
    print(f"\n次のステップ:")
    print(f"1. 画像を {output_dir}/images/train と {output_dir}/images/val に配置")
    print(f"2. アノテーションを {output_dir}/labels/train と {output_dir}/labels/val に配置")


def main():
    parser = argparse.ArgumentParser(
        description='データセットをYOLO形式に変換'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['doclaynet', 'custom'],
        required=True,
        help='データセットタイプ'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        help='データセットのルートディレクトリ（DocLayNet用）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=['text', 'table', 'figure'],
        help='クラス名（カスタムデータセット用）'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset_type == 'doclaynet':
        if not args.dataset_root:
            raise ValueError("--dataset-rootを指定してください")
        dataset_root = Path(args.dataset_root)
        prepare_doclaynet_dataset(dataset_root, output_dir)
    elif args.dataset_type == 'custom':
        prepare_custom_dataset(
            images_dir=Path('images'),
            annotations_dir=Path('labels'),
            output_dir=output_dir,
            class_names=args.class_names
        )


if __name__ == '__main__':
    main()
