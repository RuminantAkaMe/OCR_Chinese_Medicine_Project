# train.py

from ultralytics import YOLO
import os

def train_yolo(data_yaml, model_output_dir, epochs=15, imgsz=640):
    # Create output directory
    os.makedirs(model_output_dir, exist_ok=True)

    model = YOLO('yolo10n.yaml')  

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=model_output_dir,
        name="chinese_characters_model_yolo10",
        exist_ok=True,
        pretrained=False 
    )

    print(f"[INFO] Training completed. Model saved in {model_output_dir}/chinese_characters_model_yolo10/")

if __name__ == "__main__":
    data_yaml = 'data.yaml'
    model_output_dir = 'models'
    train_yolo(data_yaml, model_output_dir)
