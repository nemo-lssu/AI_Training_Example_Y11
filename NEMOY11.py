
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO("Feb_Model.pt")

cwd = os.getcwd()
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=f"{cwd}/BB_Augmented/data.yaml", epochs=50, imgsz=640,batch=8,device="cuda",patience = 25,freeze=10)

print ("[NEMO] Training complete. Success")
#/media/amore/Vision_AGX1/NEMOTRAIN/YoloV9 Training Tutorial/AI_Training/RoboBoat2025_v3_Dataset
