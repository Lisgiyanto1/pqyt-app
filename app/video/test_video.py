import argparse
import os
from app.video.process_video import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis Video dengan Model ONNX dan Deteksi Objek di Jalan.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path ke file video input.")
    parser.add_argument('-o', '--output', type=str, default="output/output_video.mp4", help="Path untuk menyimpan file video hasil.")
    parser.add_argument('-m', '--model', type=str, default=os.path.abspath(os.path.join(os.getcwd(), "app", "core", "models", "EfficientNetb0100_baseline.onnx")), help="Path ke file model .onnx.")
    parser.add_argument('--no-show', action='store_true', help="Jalankan tanpa menampilkan jendela pratinjau real-time.")
    parser.add_argument('--debug', action='store_true', help="Aktifkan mode debug")

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        show_window=not args.no_show,
        debug_mode=args.debug
    )
