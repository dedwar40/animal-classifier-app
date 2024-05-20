from ultralytics import YOLO

# Load a model

model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)  


# Use the model
results = model.train(data="config16.yaml", epochs=300, batch=32)  # train the model
