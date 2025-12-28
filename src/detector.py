import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch


class DocumentLayoutDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if model_path is None:
            print("学習済みモデルをダウンロード中...")
            model_path = 'yolov8n.pt'

        print(f"モデルをロード中: {model_path}")
        self.model = YOLO(model_path)

        if self.device == 'cpu':
            torch.set_num_threads(4)

        print(f"デバイス: {self.device}")
        print(f"信頼度閾値: {self.conf_threshold}")
        print(f"IOU閾値: {self.iou_threshold}")

    def detect(
        self,
        image_path: Union[str, Path],
        save_visualization: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")

        print(f"画像を処理中: {image_path.name}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = self._parse_results(results[0])

        if save_visualization:
            if output_dir is None:
                output_dir = Path('output')
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            vis_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
            self._save_visualization(image_path, detections, vis_path)
            print(f"検出結果を保存: {vis_path}")

        return detections

    def _parse_results(self, result) -> Dict:
        detections = {
            'image_size': {
                'width': int(result.orig_shape[1]),
                'height': int(result.orig_shape[0])
            },
            'objects': []
        }

        if result.boxes is None or len(result.boxes) == 0:
            print("検出された領域はありません")
            return detections

        boxes = result.boxes.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = result.names[cls] if hasattr(result, 'names') else str(cls)

            detections['objects'].append({
                'class': class_name,
                'confidence': conf,
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                }
            })

        print(f"検出された領域数: {len(detections['objects'])}")
        return detections

    def _save_visualization(
        self,
        image_path: Path,
        detections: Dict,
        output_path: Path
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗: {image_path}")

        colors = self._generate_colors(detections)

        for obj in detections['objects']:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            class_name = obj['class']
            conf = obj['confidence']

            color = colors.get(class_name, (0, 255, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {conf:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            y_label = max(y1, label_size[1] + 10)
            cv2.rectangle(
                image,
                (x1, y_label - label_size[1] - 10),
                (x1 + label_size[0], y_label + baseline - 10),
                color,
                -1
            )
            cv2.putText(
                image, label, (x1, y_label - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        cv2.imwrite(str(output_path), image)

    def _generate_colors(self, detections: Dict) -> Dict[str, Tuple[int, int, int]]:
        unique_classes = set(obj['class'] for obj in detections['objects'])

        predefined_colors = {
            'text': (255, 0, 0),
            'paragraph': (255, 0, 0),
            'title': (0, 255, 255),
            'list': (255, 255, 0),
            'table': (0, 255, 0),
            'figure': (255, 0, 255),
            'image': (255, 0, 255),
            'caption': (0, 165, 255),
        }

        colors = {}
        for cls in unique_classes:
            cls_lower = cls.lower()
            if cls_lower in predefined_colors:
                colors[cls] = predefined_colors[cls_lower]
            else:
                np.random.seed(hash(cls) % 2**32)
                colors[cls] = tuple(np.random.randint(0, 255, 3).tolist())

        return colors

    def batch_detect(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ) -> List[Dict]:
        image_dir = Path(image_dir)
        if not image_dir.exists() or not image_dir.is_dir():
            raise ValueError(f"画像ディレクトリが見つかりません: {image_dir}")

        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"画像が見つかりません: {image_dir}")
            return []

        print(f"処理する画像数: {len(image_files)}")

        all_results = []
        for image_path in image_files:
            result = self.detect(image_path, save_visualization=True, output_dir=output_dir)
            result['filename'] = image_path.name
            all_results.append(result)

        return all_results
