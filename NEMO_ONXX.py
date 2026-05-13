
import os
from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO("Feb_Model.pt")
cwd = os.getcwd()

model.export(format="onnx",dynamic=False)
print ("[NEMO] Training complete. Success")
