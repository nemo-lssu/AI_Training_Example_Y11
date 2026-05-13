
import os
from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO("Feb_Model.pt")

cwd = os.getcwd()
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=f"{cwd}/data.yaml", epochs=50, imgsz=640,batch=8,device="cuda",patience = 25,freeze=10)

print ("[NEMO] Training complete. Success")
