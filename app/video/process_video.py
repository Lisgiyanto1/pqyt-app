import os

import cv2
import numpy as np


# Load model ResNet50 ONNX
def load_model(model_path="Resnet50.onnx"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    print(f"✅ Loading model: {model_path}")
    net = cv2.dnn.readNetFromONNX(model_path)
    return net

# Preprocess frame sesuai input ResNet50
def preprocess_frame(frame, size=224):
    # Resize -> BGR to RGB -> scale
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0/255.0,
        size=(size, size),
        mean=(0.485*255, 0.456*255, 0.406*255),  # ImageNet mean
        swapRB=True,
        crop=False
    )
    # Normalize dengan std (trik: bagi setelah blob dibuat)
    blob[0][0] = (blob[0][0] - 0.485) / 0.229
    blob[0][1] = (blob[0][1] - 0.456) / 0.224
    blob[0][2] = (blob[0][2] - 0.406) / 0.225
    return blob

def process_video(video_path, model_path="Resnet50.onnx"):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return

    # Load model
    net = load_model(model_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    print(f"🎥 Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished reading video.")
            break

        # Preprocess frame
        blob = preprocess_frame(frame)
        net.setInput(blob)

        # Forward pass
        preds = net.forward()  # shape biasanya (1, 1000) untuk ImageNet
        class_id = np.argmax(preds)
        confidence = preds[0][class_id]

        # Tampilkan di frame
        text = f"Class ID: {class_id}, Conf: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ResNet50 Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("vid1.MOV", "Resnet50.onnx")
