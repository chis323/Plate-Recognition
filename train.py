from ultralytics import YOLO

# Load a base model (e.g., yolov8n.pt)
model = YOLO("yolov8n.pt")

# Train the model on your custom dataset
results = model.train(
    data=r"C:\Users\Chis Bogdan\Desktop\NPR\recognize-license-plate-master\dataset2\data.yaml",
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    name="license_plate_detection"  # Name of the training run
)