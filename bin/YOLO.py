from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("./models/yolov10n.pt")

# Perform object detection on an image
results = model("../data/images/Horizontal/0b7d98ea-4200080346_5543_156.jpg")

# Display the results
results[0].show()
