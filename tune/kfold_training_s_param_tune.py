import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

def setup_directories_for_fold(base_path, fold_idx):
    """ Set up directories for training and validation for each fold """
    train_dir = base_path / f"fold_{fold_idx}" / 'train'
    val_dir = base_path / f"fold_{fold_idx}" / 'val'
    for dir in [train_dir, val_dir]:
        dir.mkdir(parents=True, exist_ok=True)
        (dir / 'images').mkdir(parents=True, exist_ok=True)
        (dir / 'labels').mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir

def create_yaml_for_fold(train_dir, val_dir, yaml_path, class_dict):
    """ Create YAML file for a fold """
    yaml_content = {
        'train': str(train_dir / 'images'),
        'val': str(val_dir / 'images'),
        'nc': len(class_dict),
        'names': list(class_dict.values())
    }
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(yaml_content, file)
    print(f"YAML file created for training: {yaml_path}")

def main():
    base_path = Path('/scratch/COSC591GroupB/training_data_1000_kfold')
    class_dict = {
        0: 'Rabbit', 1: 'Brushtailed Rock Wallaby', 2: 'Spotted Tailed Quoll', 3: 'Swamp Wallaby',
        4: 'Eastern Grey Kangaroo', 5: 'Red Necked Wallaby', 6: 'Echidna', 7: 'Bandicoot',
        8: 'Brushtail Possum', 9: 'Lyrebird', 10: 'Magpie', 11: 'Kookaburra',
        12: 'Currawong', 13: 'Fox', 14: 'Cat', 15: 'Goat'
    }
    kfold = 5
    hyperparameters = {
        'batch': [16,32,64,128],
        'lr0': [0.001, 0.01, 0.1],
        'epochs': [100,200,300]
    }
    project_name = 'kfold_demo_yolov8s_bt64_ep300'

    # Handle all folds one at a time for all species
    for fold_idx in range(1, kfold + 1):
        train_dir, val_dir = setup_directories_for_fold(base_path, fold_idx)
        yaml_path = train_dir.parent / 'dataset.yaml'
        create_yaml_for_fold(train_dir, val_dir, yaml_path, class_dict)

        for batch in hyperparameters['batch']:
            for lr0 in hyperparameters['lr0']:
                for epochs in hyperparameters['epochs']:
                    # Initialize and train the YOLO model with the current hyperparameters
                    # Change model type as needed. 
                    model = YOLO('yolov8s.pt', task='train')
                    print(f"Starting training for batch={batch}, lr0={lr0}, epochs={epochs}, fold={fold_idx}...")
                    #model.train(data=yaml_path.as_posix(), epochs=epochs, batch=batch, lr0=lr0, project=project_name, name=f"train_fold_{fold_idx}")
                    model.tune(data=yaml_path.as_posix(), epochs=epochs, batch=batch, lr0=lr0, project=project_name, name=f"train_fold_{fold_idx}")
                    print(f"Training completed for batch={batch}, lr0={lr0}, epochs={epochs}, fold={fold_idx}.")

    print("All training processes completed successfully.")

if __name__ == "__main__":
    main()

