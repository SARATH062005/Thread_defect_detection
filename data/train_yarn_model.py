from ultralytics import YOLO
import torch

def train_yarn_defect_model():
    """
    This function trains a YOLOv8 model on the yarn defect dataset.
    """
    # Check if a GPU is available and print the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load a pretrained model ---
    # We use a pretrained model to benefit from transfer learning.
    # 'yolov8n.pt' is the smallest and fastest model.
    # For higher accuracy, you can use 'yolov8s.pt' or 'yolov8m.pt'.
    model = YOLO('yolov8s.pt')
    model.to(device) # Move model to GPU if available

    # --- 2. Define the path to your dataset configuration file ---
    # !! IMPORTANT: CHANGE THIS PATH to where your 'data.yaml' file is located !!
    dataset_yaml_path = r"C:\Users\sarat\Computer_vision\yolo_v8\data\data.yaml"

    # --- 3. Train the model ---
    print("Starting model training...")
    # The 'train' method returns a results object.
    # results = model.train(
    #     data=dataset_yaml_path,
    #     epochs=100,          # Number of training epochs. Start with 50-100.
    #     imgsz=640,           # Image size for training. 640 is standard for YOLOv8.
    #     batch=16,            # Batch size. Reduce if you get 'Out of Memory' errors.
    #     name='yolov8n_yarn_defects_v1' # Name for the training run folder.
    # )
    results = model.train(
        data=dataset_yaml_path,
        epochs=200,
        imgsz=800,
        batch=8,
        name='yolov8s_yarn_defects_augmented',
        # --- Add augmentation parameters ---
        degrees=10.0,    # random rotation (-10 to +10 degrees)
        translate=0.1,   # random translation
        scale=0.1,       # random zoom (-10% to +10%)
        shear=5.0,       # random shear
        perspective=0.0, # random perspective
        flipud=0.5,      # random vertical flip (50% chance)
        fliplr=0.5,      # random horizontal flip (50% chance)
        mosaic=1.0,      # mosaic augmentation (powerful for different object scales)
        mixup=0.1,       # mixup augmentation
        hsv_h=0.015,     # change hue
        hsv_s=0.7,       # change saturation
        hsv_v=0.4        # change brightness/value
    )

    print("Training finished!")
    print("Best model weights saved at:", results.save_dir)

if __name__ == '__main__':
    train_yarn_defect_model()