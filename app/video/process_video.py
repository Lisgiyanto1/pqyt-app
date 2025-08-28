import cv2
import numpy as np
from shapely.geometry import Polygon, box
from app.core.services.object_detection import get_combined_prediction

def process_video(input_path: str, output_path: str, model_path: str,
                  show_window: bool = True, debug_mode: bool = False, progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka video di '{input_path}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    source_points = np.float32([
        (-20, 430.0),
        (330.0, 250.0),
        (350.0, 250.0),
        (630.0, 430.0)
    ])
    road_polygon = Polygon(source_points)

    bev_width, bev_height = 400, 600
    destination_points = np.float32([[0, bev_height], [0, 0], [bev_width, 0], [bev_width, bev_height]])
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    PIXELS_PER_METER = 20

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
        image_bytes = buffer.tobytes()

        analysis_results = get_combined_prediction(image_bytes, model_path)
        if "error" in analysis_results:
            video_writer.write(frame)
            continue

        objects_in_roi = []

        if "detections" in analysis_results:
            for pred in analysis_results["detections"]:
                x1, y1, x2, y2 = map(int, pred['bounding_box'])
                bbox_poly = box(x1, y1, x2, y2)
                is_inside = road_polygon.intersects(bbox_poly)

                if is_inside:
                    ref_point = (int((x1 + x2) / 2), y2)
                    ref_point_np = np.array([[ref_point]], dtype=np.float32)
                    transformed_point = cv2.perspectiveTransform(ref_point_np, M)
                    y_bev = transformed_point[0][0][1]
                    distance_pixels = bev_height - y_bev
                    distance_meters = distance_pixels / PIXELS_PER_METER

                    objects_in_roi.append({
                        'class': pred['label'],
                        'distance_m': distance_meters,
                        'box': (x1, y1, x2, y2)
                    })

        # TODO: tambahkan overlay (opsional)

        video_writer.write(frame)

        if progress_callback:
            progress_callback(int((frame_num / total_frames) * 100))

    cap.release()
    video_writer.release()
