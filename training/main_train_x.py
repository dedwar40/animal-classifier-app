from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8x.yaml")  # build a new model from scratch using x version model = model 
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)  


# Use the model
results = model.train(data="config16.yaml", epochs=300, batch=32)  # train the model
