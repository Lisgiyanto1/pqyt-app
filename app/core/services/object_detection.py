import onnxruntime as ort
import numpy as np
import cv2

def get_combined_prediction(image_bytes, model_path):
    try:
        # Load model
        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Decode image
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Gagal membaca gambar"}

        # TODO: lakukan pre-processing sesuai model kamu
        input_data = cv2.resize(img, (224, 224)).astype(np.float32)
        input_data = np.transpose(input_data, (2, 0, 1)) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Inference
        outputs = session.run(None, {"input": input_data})

        # TODO: parsing output agar konsisten dengan kebutuhan
        # Misal outputs[0] berisi bounding box, label, dll.
        return {"detections": []}  # placeholder
    except Exception as e:
        return {"error": str(e)}
