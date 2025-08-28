import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QProgressBar
from app.gui.video_worker import VideoWorker

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Video ONNX - Jetson Orin NX")
        self.resize(400, 200)

        self.label = QLabel("Pilih video untuk diproses")
        self.progress = QProgressBar()
        self.btn_select = QPushButton("Pilih Video")
        self.btn_run = QPushButton("Jalankan Inference")

        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_inference)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def select_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi)")
        if fname:
            self.video_path = fname
            self.label.setText(f"Video: {fname}")

    def run_inference(self):
        output_path = os.path.join("output", "output_gui.mp4")
        model_path = "app/core/models/EfficientNetb0100_baseline.onnx"

        self.worker = VideoWorker(self.video_path, output_path, model_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, output_path):
        self.label.setText(f"Selesai! Hasil disimpan di: {output_path}")
        self.progress.setValue(100)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
