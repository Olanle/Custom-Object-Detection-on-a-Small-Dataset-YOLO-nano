# Custom Vehicle Object Detection (YOLOv8-nano)

Training YOLOv8-nano to detect and classify 6 vehicle types in real-world images using a pre-annotated YOLO-format dataset.

---

## Problem Statement

Object detection goes beyond classification — it predicts not just what is in an image but where each object is located using bounding boxes. This project trains a lightweight YOLOv8-nano detector on 6 vehicle classes, producing a model fast enough for real-world deployment and accurate enough for production use.

YOLOv8 Nano (often referred to as yolov8n.pt) is the smallest and fastest version of the Ultralytics YOLOv8 object detection model family. Released in January 2023, it is designed for edge AI applications and devices with limited computational power, such as Raspberry Pis, drones, or mobile systems.

---

## Dataset Selection Journey

Finding the right dataset was itself a significant learning experience. Three datasets were evaluated:

**Attempt 1 — Nigerian Naira Currency Dataset:** Contained only images with no annotation files. Classification dataset only — YOLO training not possible without bounding boxes. Rejected.

**Attempt 2 — Single-Class Vehicle CSV Dataset:** Had bounding box annotations in CSV format but no class column — every object was the same class. Too simple for a meaningful multi-class detector. Rejected.

**Attempt 3 — Multi-Class Vehicle Dataset (Final):** Clean YOLO-format structure with separate images and labels folders, 6 classes, and a classes.txt file. Annotation format confirmed by inspecting a sample label file. Selected.

**Key lesson:** A YOLO-ready dataset has a `labels/` folder with `.txt` files alongside the `images/` folder, plus a `classes.txt`. That is the signature to look for.

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| Train | 2,100 | 2,100 |
| Validation | 900 | 900 |

**6 Classes:** car, threewheel, bus, truck, motorbike, van

Dataset was already in YOLO format — no annotation conversion required.

---

## YAML Configuration

**Challenge:** Initial attempts used absolute paths for train and val which caused YOLO to fail locating data. Also unclear why a YAML file was needed at all.

**Solution:** Use relative paths from the base `path` key. The YAML is the configuration contract between your data and the model — it tells YOLO where data lives, how many classes exist, and what they are called.

```yaml
path: /content/vehicle/vehicle dataset
train: train/images
val:   valid/images
nc:    6
names: [car, threewheel, bus, truck, motorbike, van]
```

---

## Training Setup

| Component | Value |
|-----------|-------|
| Model | YOLOv8-nano |
| Pretrained | Yes (COCO weights) |
| Epochs | 10 |
| Image Size | 640 |
| Batch Size | 16 |
| Device | CUDA (Tesla T4) |

---

## Results

**Final mAP@50: 97.2%** across 900 validation images

| Metric | Value |
|--------|-------|
| mAP@50 | 97.2% |
| mAP@50-95 | 88.0% |
| Precision | 96.0% |
| Recall | 91.3% |
| Inference Speed | 1.9ms/image |

**Per-Class mAP@50:**

| Class | mAP@50 |
|-------|--------|
| car | 91.7% |
| threewheel | 87.0% |
| bus | 94.5% |
| truck | 86.6% |
| motorbike | 73.8% |
| van | 94.2% |

Motorbike scored lowest — smaller objects at varied angles are consistently harder to detect across all object detection tasks.

---

## Human-Readable Inference

**Challenge:** Raw YOLO output returned PyTorch tensors and verbose logs — not useful for real applications.

**Solution:** Parse class indices and confidence scores with `.item()`, map indices to class names with a dictionary, and suppress logs with `verbose=False`.

```python
class_names = {0:'car', 1:'threewheel', 2:'bus', 3:'truck', 4:'motorbike', 5:'van'}
results = best_model(img_path, verbose=False)

for i in range(len(results[0].boxes)):
    cls  = int(results[0].boxes.cls[i].item())
    conf = results[0].boxes.conf[i].item() * 100
    print(f"Detected: {class_names[cls]} (confidence: {conf:.2f}%)")
```

**Sample output:**
```
Detected: motorbike (confidence: 96.90%)
```

---

## Key Learnings

- Always verify annotation format before committing to a dataset
- YAML files are the configuration contract between your data and YOLO — not optional overhead
- Relative paths in YAML are cleaner and more portable than absolute paths
- Raw tensor output must be parsed with `.item()` for human-readable results
- `verbose=False` suppresses inference logs for clean output
- 97.2% mAP@50 in 10 epochs demonstrates YOLOv8-nano's power on clean labelled data

---

## Stack
`PyTorch` `Ultralytics YOLOv8` `OpenCV` `Kaggle`

---

*Part of a 10-project ML engineering curriculum targeting Edge/Embedded ML roles.*
