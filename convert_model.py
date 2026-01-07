"""
Convert YOLOv8 pose model to ONNX with compatible opset version
"""
from ultralytics import YOLO

# Load the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Export to ONNX with opset 13 (compatible with ONNX Runtime Web)
model.export(
    format='onnx',
    opset=13,
    simplify=True,
    dynamic=False,
    imgsz=640
)

print("✓ Model exported successfully!")
print("✓ File: yolov8n-pose.onnx (with opset 13)")
print("✓ Ready to use in the browser!")
