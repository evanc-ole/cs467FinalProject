from tqdm import tqdm
from ultralytics import YOLO
import pathlib


dataset_root = pathlib.Path('yolo_dataset').resolve()

yaml_text = f"""
path: {dataset_root}          # root dir
train: images/train   # relative to 'path'
val:   images/val
nc: 1
names: ['rect']
"""
(pathlib.Path('rect.yaml')).write_text(yaml_text)


model = YOLO('yolov8n-seg.pt')  # nano variant (2.5 M params)
model.train(
    data='rect.yaml',
    epochs=60,          # usually enough for 100 images
    imgsz=640,
    batch=8,            # keep small for CPU
    device='cpu',
    lr0=1e-3,
    optimizer='AdamW'
)