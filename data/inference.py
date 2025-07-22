from ultralytics import YOLO
from PIL import Image
import os

def run_inference():
    """
    This function runs inference using the trained yarn defect model.
    """
    # --- 1. Load your custom-trained model ---
    # !! IMPORTANT: Update this path to your 'best.pt' file !!
    model_path = r"C:\Users\sarat\Computer_vision\yolo_v8\runs\detect\yolov8s_yarn_defects_augmented\weights\best.pt"
    model = YOLO(model_path)

    # --- 2. Define the path to the image you want to test ---
    # !! IMPORTANT: Update this path to an image of yarn !!
    image_to_predict = r"C:\Users\sarat\Computer_vision\yolo_v8\data\test\images\33_bmp.rf.8e175e88a088ebc8ebaeeb5a192862b7.jpg"
    
    # --- 3. Run prediction ---
    print(f"Running inference on: {image_to_predict}")
    results = model.predict(source=image_to_predict, save=True, conf=0.5) # save=True saves the image with boxes
    
    # --- 4. Process and display results ---
    # The results object is a list of Results objects.
    for result in results:
        # Get the annotated image as a NumPy array
        annotated_image_array = result.plot()
        
        # Convert to a PIL Image and display it
        annotated_image = Image.fromarray(annotated_image_array[..., ::-1]) # Convert BGR to RGB
        annotated_image.show()

        # Print detected boxes information
        boxes = result.boxes
        print(f"Found {len(boxes)} defects.")
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            print(f"  - Defect: {class_name}, Confidence: {confidence:.2f}")


if __name__ == '__main__':
    # You can comment out the training function if you only want to run inference
    # train_yarn_defect_model()
    
    # Run the inference function
    run_inference()