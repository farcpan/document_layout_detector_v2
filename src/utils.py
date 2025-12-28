import json
from pathlib import Path
from typing import Dict, List, Union


def save_results_json(
    results: Union[Dict, List[Dict]],
    output_path: Union[str, Path]
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"結果をJSON形式で保存: {output_path}")


def load_results_json(json_path: Union[str, Path]) -> Union[Dict, List[Dict]]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSONファイルが見つかりません: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_detection_summary(detections: Dict):
    print("\n" + "="*50)
    print("検出結果サマリー")
    print("="*50)
    print(f"画像サイズ: {detections['image_size']['width']}x{detections['image_size']['height']}")
    print(f"検出領域数: {len(detections['objects'])}")

    if detections['objects']:
        class_counts = {}
        for obj in detections['objects']:
            cls = obj['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\nクラス別検出数:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")

    print("="*50 + "\n")
