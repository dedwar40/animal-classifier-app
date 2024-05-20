from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8m.yaml") 
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)  


# Use the model
results = model.train(data="config16.yaml", epochs=300, batch=64)  # train the model
