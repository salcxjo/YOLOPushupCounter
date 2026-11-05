from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

# Export to OpenVINO IR format with INT8 quantization
model.export(format="openvino", imgsz = 320)