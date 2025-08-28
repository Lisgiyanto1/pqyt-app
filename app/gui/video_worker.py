from PyQt5.QtCore import QThread, pyqtSignal
from app.video.process_video import process_video

class VideoWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_path, model_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

    def run(self):
        process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            model_path=self.model_path,
            show_window=False,
            debug_mode=False,
            progress_callback=self.progress.emit
        )
        self.finished.emit(self.output_path)
