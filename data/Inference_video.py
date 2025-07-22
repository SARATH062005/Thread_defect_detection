from ultralytics import YOLO
import os

def run_video_inference():
    """
    Run YOLOv8 inference on an mp4 video and save the output with detected defects.
    """
    # --- 1. Load the custom-trained model ---
    model_path = r"C:\Users\sarat\Computer_vision\yolo_v8\runs\detect\yolov8n_yarn_defects_v13\weights\best.pt"
    model = YOLO(model_path)

    # --- 2. Define the video path you want to test ---
    input_video_path = r"C:\Users\sarat\Downloads\Untitled design (1).mp4"

    # --- 3. Run prediction on video ---
    print(f"Running inference on video: {input_video_path}")
    results = model.predict(
        source=input_video_path,
        conf=0.5,
        save=True,       # Save output video with boxes
        save_txt=False,  # Set to True if you want detection labels in txt format
        save_crop=False  # Set to True if you want cropped defects
    )

    # --- 4. Show where output is saved ---
    output_dir = results[0].save_dir  # Get the directory where video is saved
    output_video_path = os.path.join(output_dir, os.path.basename(input_video_path))
    print(f"\n Output saved at: {output_video_path}")

if __name__ == '__main__':
    run_video_inference()
