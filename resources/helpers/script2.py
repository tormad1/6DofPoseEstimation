from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(data="can-1/data.yaml", epochs=50, imgsz=640)
