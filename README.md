# 🧵 Thread Defect Detection using YOLOv8

This project is focused on detecting defects in yarn/thread during textile manufacturing using a custom-trained YOLOv8 object detection model. It is capable of identifying critical thread anomalies such as:

  - 🔁 Loop Fiber
  - 🪡 Protruding Fiber

The system helps textile industries automate quality control and improve consistency in thread-based products.

-----

## 🚀 Features

  - ✅ Real-time defect detection on yarn/thread
  - 🎯 High-precision custom YOLOv8 model
  - 📈 Visualized output for both image and video inference
  - 💾 Easy integration into production lines for automated inspection

-----

## 🧠 Model

  - **Model**: YOLOv8 (from [Ultralytics](https://github.com/ultralytics/ultralytics))
  - **Training Dataset**: Custom annotated dataset of yarn/thread with multiple defect classes
  - **Format**: `.pt` trained model weights

-----

## 📂 Project Structure

```
Thread_Defect_Detection/
├── dataset/
│   ├── images/
│   └── labels/
├── weights/
│   └── best.pt
├── inference/
│   ├── images/
│   └── videos/
├── detect.py
├── train.py
└── README.md
```

-----

## ⚙️ Installation

To set up the project, follow these steps:

```bash
git clone https://github.com/SARATH062005/Thread_defect_detection.git
cd Thread_defect_detection
pip install -r requirements.txt
```

-----

## 🧪 How to Run Inference

### 🔍 For Images:

To detect defects in images, use the following command:

```bash
yolo task=detect mode=predict model=weights/best.pt source=inference/images
```

### 🎥 For Videos:

For video inference, run:

```bash
yolo task=detect mode=predict model=weights/best.pt source=inference/videos save=True
```

📌 **Output**: The processed video with detected defects will be saved in `runs/detect/` in MP4 format.

-----

## 🏋️‍♂️ Training Your Own Model

To train your own custom model:

1.  Place your images and YOLO-format labels inside the `dataset/` folder.

2.  Create a `data.yaml` file with your class names.

3.  Run the training command:

    ```bash
    yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640
    ```

-----

## 📊 Results

The model demonstrates strong performance in detecting thread defects.

| Metric     | Value   |
| :--------- | :------ |
| mAP50      | 90%+    |
| mAP50-95   | \~75%    |
| Precision  | High    |
| Recall     | High    |

The model achieves high precision in image inference and is being fine-tuned for better motion inference on video streams.

-----

## 📸 Example Output

![Thread Defect Detection Example](https://github.com/SARATH062005/Thread_defect_detection/blob/main/runs/detect/predict4/Thread2.jpg)

---

-----

## 🤖 Industrial Impact

This solution provides a robust and scalable way to detect micro defects in yarns, significantly reducing manual inspection time and improving defect traceability in textile manufacturing processes.

-----

## 🧑‍💻 Author

**Sarath Chandiran**

Computer Vision & Robotics Engineer

[LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/sarathchandiran/) | [GitHub](https://www.google.com/search?q=https://github.com/SARATH062005)

-----

## 🙌 Contributions

Contributions, suggestions, and improvements are welcome\! Feel free to open an issue or a pull request.
