from ultralytics import YOLO
import os

def train_yolo(data_yaml, model_output_dir, epochs=25, imgsz=640):
    # Create output directory
    os.makedirs(model_output_dir, exist_ok=True)

    # Load YOLOv8 model (smallest, fast version)
    model = YOLO('yolov8n.pt')  # Start from a pre-trained YOLOv8n (nano) checkpoint

    # Train the model
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        project=model_output_dir,
        name="chinese_characters_model",
        exist_ok=True
    )

    print(f"[INFO] Training completed. Model saved in {model_output_dir}/chinese_characters_model/")

if __name__ == "__main__":
    data_yaml = 'data.yaml'
    model_output_dir = 'models'
    train_yolo(data_yaml, model_output_dir)
