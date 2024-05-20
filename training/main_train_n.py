from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch using nano versionmodel = model 
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="config16.yaml", epochs=300, batch = 64)  # train the model
#results = model.tune(data="config16.yaml", epochs=100, iterations=300, optimizer='AdamW', plots=False, save=True, val=True)

