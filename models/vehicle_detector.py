import cv2
import numpy as np
import os
import supervision as sv
import csv
from ultralytics import YOLO


from speed.speed_estimator import ViewTransformer, SpeedEstimator


VEHICLE_CLASS_IDS = [2, 5]  # 2 = car, 5 = bus

SPEED_THRESHOLD_KMH = 60.0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SOURCE = np.array([
    [500, 120],
    [800, 120],
    [1500, 900],
    [-250, 900]
])

TARGET_WIDTH = 20
TARGET_HEIGHT = 90

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
], dtype=np.float32)

LINE_START = sv.Point(960, 300)
LINE_END = sv.Point(330, 300)


def draw_speed_boxes(
    frame,
    detections,
    labels,
    speeds,
    speed_threshold=SPEED_THRESHOLD_KMH
):
    if detections.tracker_id is None:
        return frame

    for i, tracker_id in enumerate(detections.tracker_id):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])

        speed_kmh = speeds.get(int(tracker_id))

        if speed_kmh is None:
            color = (200, 200, 200)
            label = f"ID {int(tracker_id)} | measuring..."
        else:
            color = (
                (0, 0, 255)
                if speed_kmh >= speed_threshold
                else (0, 255, 0)
            )

            label = (
                f"ID {int(tracker_id)} | "
                f"{speed_kmh:.1f} km/h"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        text_width, text_height = text_size

        label_y1 = max(0, y1 - text_height - 10)

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + text_width + 8, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame
def draw_stats_panel(frame, cars, buses, in_count, out_count, avg_speed):
    overlay = frame.copy()

    panel_x1, panel_y1 = 18, 18
    panel_x2, panel_y2 = 300, 118

    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    scale = 0.55
    thickness = 1

    cv2.putText(frame, f"Cars: {cars}   Buses: {buses}", (30, 45), font, scale, color, thickness)
    cv2.putText(frame, f"Vehicles In: {in_count}   Out: {out_count}", (30, 72), font, scale, color, thickness)
    cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (30, 99), font, scale, color, thickness)

    return frame

def draw_bird_eye_view(transformed_points, detections, frame_size=(400, 500)):
    bird_view = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    scale_x = frame_size[0] / TARGET_WIDTH
    scale_y = frame_size[1] / TARGET_HEIGHT

    if detections.tracker_id is not None and len(transformed_points) > 0:
        for i, point in enumerate(transformed_points):
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)

            x = max(0, min(frame_size[0] - 1, x))
            y = max(0, min(frame_size[1] - 1, y))

            cv2.circle(bird_view, (x, y), 6, (0, 255, 0), -1)

            if i < len(detections.tracker_id):
                cv2.putText(
                    bird_view,
                    str(int(detections.tracker_id[i])),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

    cv2.rectangle(
        bird_view,
        (0, 0),
        (frame_size[0] - 1, frame_size[1] - 1),
        (255, 255, 255),
        1
    )

    cv2.putText(
        bird_view,
        "Bird's Eye View",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return bird_view

def run_vehicle_detection(video_path: str, model_path: str) -> None:
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    os.makedirs("outputs", exist_ok=True)


    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "outputs/output.mp4",
        fourcc,
        fps,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    bird_out = cv2.VideoWriter(
        "outputs/bird_eye_view.mp4",
        fourcc,
        fps,
        (400, 500)
    )

    frame_delay = max(1, int(1000 / fps))

    tracker = sv.ByteTrack(frame_rate=fps)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    speed_estimator = SpeedEstimator(fps=fps, smoothing_window=12)

    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=int(fps * 2),
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=(FRAME_WIDTH, FRAME_HEIGHT)
    )

    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    number_of_cars = 0
    number_of_buses = 0
    line_counted_ids = set()

    os.makedirs("outputs", exist_ok=True)

    csv_file_path = "outputs/vehicles_log.csv"
    csv_rows = []

    vehicle_last_y = {}
    manual_in_count = 0
    manual_out_count = 0

    vehicle_counter = 1
    vehicle_mapping = {}
    paused = False
    frame_index = 0
    while True:
        ret, frame = cap.read()
        frame_index += 1
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))


        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
       
        # Filter confidence
        confidence_mask = detections.confidence > 0.40
        detections = detections[confidence_mask]

        if detections.class_id is not None:
            class_mask = np.array([class_id in VEHICLE_CLASS_IDS for class_id in detections.class_id])
            detections = detections[class_mask]

            # Remove tiny detections
            if len(detections) > 0:
                widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
                heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]

                size_mask = (widths > 35) & (heights > 35)
        detections = detections[polygon_zone.trigger(detections)]
        # Ignore vehicles too close to edges
        if len(detections) > 0:
            x1 = detections.xyxy[:, 0]
            x2 = detections.xyxy[:, 2]

            edge_mask = (x1 > 15) & (x2 < FRAME_WIDTH - 15)
            detections = detections[edge_mask]
            # Pastreaza rezultatul YOLO inainte de ByteTrack
            yolo_detections = detections
        detections = tracker.update_with_detections(detections)

        line_zone.trigger(detections)

        anchor_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_points = view_transformer.transform_points(anchor_points)

        bird_view = draw_bird_eye_view(
            transformed_points=transformed_points,
            detections=detections
        )

        current_speeds = {}
        labels = []

        if detections.tracker_id is not None and detections.class_id is not None:
            for i, (tracker_id, class_id) in enumerate(zip(detections.tracker_id, detections.class_id)):
                tracker_id = int(tracker_id)
                class_name = model.names[int(class_id)]

                if i < len(transformed_points):
                    transformed_y = transformed_points[i][1]
                    speed_kmh = speed_estimator.update(tracker_id, transformed_y)
                    if speed_kmh is not None:
                        current_speeds[tracker_id] = speed_kmh

                    label = class_name

                    if speed_kmh is not None:
                        label += f" {int(speed_kmh)} km/h"

                    labels.append(label)
                else:
                    labels.append(class_name)

        # count only when vehicle reaches the line
        line_y = LINE_START.y

        for i in range(len(detections)):
            if detections.tracker_id is None or detections.class_id is None:
                break

            tracker_id = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])

            _, _, _, y2 = detections.xyxy[i]
            current_y = float(y2)

            if tracker_id in vehicle_last_y and tracker_id not in line_counted_ids:
                previous_y = vehicle_last_y[tracker_id]

                crossed_down = previous_y < line_y <= current_y
                crossed_up = previous_y > line_y >= current_y

                if crossed_down or crossed_up:
                    speed_value = speed_estimator.latest_speeds.get(tracker_id)

                    if speed_value is None or speed_value <= 5:
                        vehicle_last_y[tracker_id] = current_y
                        continue
                    if crossed_down:
                        manual_in_count += 1
                    else:
                        manual_out_count += 1

                    if class_id == 2:
                        number_of_cars += 1
                    elif class_id == 5:
                        number_of_buses += 1

                    line_counted_ids.add(tracker_id)

                    speed_estimator.confirm_speed(tracker_id)

                    direction = "IN" if crossed_down else "OUT"
                    vehicle_type = "car" if class_id == 2 else "bus"
                    speed_value = speed_estimator.latest_speeds.get(tracker_id, 0.0)

                    if tracker_id not in vehicle_mapping:
                        vehicle_mapping[tracker_id] = vehicle_counter
                        vehicle_counter += 1

                    csv_rows.append({
                        "tracker_id": vehicle_mapping[tracker_id],
                        "vehicle_type": vehicle_type,
                        "direction": direction,
                        "speed_kmh": round(speed_value, 2)
                    })

            vehicle_last_y[tracker_id] = current_y

        annotated_frame = frame.copy()

        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = draw_speed_boxes(
            frame=annotated_frame,
            detections=detections,
            labels=labels,
            speeds=current_speeds
        )
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=SOURCE,
            color=sv.Color.RED
        )

        annotated_frame = cv2.line(
            annotated_frame,
            (int(LINE_START.x), int(LINE_START.y)),
            (int(LINE_END.x), int(LINE_END.y)),
            (255, 255, 255),
            2
        )

        for point in anchor_points:
            x, y = point.astype(int)
            cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)

        annotated_frame = draw_speed_boxes(
            frame=annotated_frame,
            detections=detections,
            labels=labels,
            speeds=current_speeds
        )

        avg_speed = speed_estimator.get_average_speed()
        in_count = manual_in_count
        out_count = manual_out_count

        annotated_frame = draw_stats_panel(
            annotated_frame,
            cars=number_of_cars,
            buses=number_of_buses,
            in_count=in_count,
            out_count=out_count,
            avg_speed=avg_speed
        )
        out.write(annotated_frame)
        bird_out.write(bird_view)
        cv2.imshow("Vehicle Detection + Tracking + Speed + Counting", annotated_frame)
        cv2.imshow("Bird's Eye View", bird_view)
        key = cv2.waitKey(0 if paused else frame_delay) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            paused = not paused

        if paused:
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord("p"):
                    paused = False
                    break

                if key == ord("q"):
                    cap.release()
                    out.release()
                    bird_out.release()
                    cv2.destroyAllWindows()
                    return

                if key == ord("n"):
                    break

    cap.release()
    out.release()
    bird_out.release()
    with open("outputs/statistics.txt", "w") as file:
        file.write(f"cars={number_of_cars}\n")
        file.write(f"buses={number_of_buses}\n")
        file.write(f"vehicles_in={manual_in_count}\n")
        file.write(f"vehicles_out={manual_out_count}\n")
        file.write(f"average_speed={speed_estimator.get_average_speed():.2f} km/h\n")

    with open(csv_file_path, "w", newline="") as csv_file:
        fieldnames = ["tracker_id", "vehicle_type", "direction", "speed_kmh"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_rows)

    cv2.destroyAllWindows()