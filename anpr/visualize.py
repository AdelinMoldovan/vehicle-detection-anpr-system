import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from anpr.plate_utils import (
    draw_border,
    json_to_box
)


VIDEO_PATH = "videos/test_video.mp4"

INTERPOLATED_CSV_PATH = (
    "outputs/anpr/anpr_interpolated.csv"
)

OUTPUT_VIDEO_PATH = (
    "outputs/anpr/anpr_final.mp4"
)


PLATE_DISPLAY_HEIGHT = 120
PLATE_TEXT_AREA_HEIGHT = 75

VEHICLE_BORDER_COLOR = (0, 255, 0)
PLATE_BORDER_COLOR = (0, 0, 255)

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def read_csv_rows(csv_path: str) -> List[dict]:
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def select_best_plate_per_vehicle(
    rows: List[dict]
) -> Dict[int, dict]:

    candidates_by_vehicle = defaultdict(
        list
    )

    for row in rows:
        license_number = row[
            "license_number"
        ].strip()

        crop_path = row[
            "plate_crop_path"
        ].strip()

        if not license_number or not crop_path:
            continue

        try:
            score = float(
                row["license_number_score"]
            )
        except (TypeError, ValueError):
            score = 0.0

        car_id = int(
            float(row["car_id"])
        )

        candidates_by_vehicle[
            car_id
        ].append({
            "text": license_number,
            "score": score,
            "crop_path": crop_path
        })

    best_results = {}

    for car_id, candidates in (
        candidates_by_vehicle.items()
    ):
        grouped_by_text = defaultdict(
            list
        )

        for candidate in candidates:
            grouped_by_text[
                candidate["text"]
            ].append(candidate)

        ranked_texts = []

        for text, text_candidates in (
            grouped_by_text.items()
        ):
            vote_count = len(
                text_candidates
            )

            average_score = sum(
                candidate["score"]
                for candidate in text_candidates
            ) / vote_count

            maximum_score = max(
                candidate["score"]
                for candidate in text_candidates
            )

            ranked_texts.append({
                "text": text,
                "vote_count": vote_count,
                "average_score": average_score,
                "maximum_score": maximum_score,
                "candidates": text_candidates
            })

        # Prioritate:
        # 1. numărul de cadre în care apare textul;
        # 2. confidence-ul mediu;
        # 3. cel mai mare confidence individual.
        ranked_texts.sort(
            key=lambda item: (
                item["vote_count"],
                item["average_score"],
                item["maximum_score"]
            ),
            reverse=True
        )

        winner = ranked_texts[0]

        best_crop_candidate = max(
            winner["candidates"],
            key=lambda item: item["score"]
        )

        crop = cv2.imread(
            best_crop_candidate[
                "crop_path"
            ]
        )

        if crop is None or crop.size == 0:
            continue

        best_results[car_id] = {
            "license_number": winner[
                "text"
            ],
            "score": winner[
                "average_score"
            ],
            "vote_count": winner[
                "vote_count"
            ],
            "crop": crop
        }

    return best_results
def resize_plate_for_overlay(
    plate_crop,
    target_height: int
):
    crop_height, crop_width = (
        plate_crop.shape[:2]
    )

    if crop_height <= 0 or crop_width <= 0:
        return None

    scale = target_height / crop_height

    target_width = max(
        1,
        int(crop_width * scale)
    )

    return cv2.resize(
        plate_crop,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC
    )


def overlay_plate_and_text(
    frame,
    vehicle_box,
    plate_crop,
    plate_text
):
    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, _ = map(
        int,
        vehicle_box
    )

    enlarged_plate = resize_plate_for_overlay(
        plate_crop,
        PLATE_DISPLAY_HEIGHT
    )

    if enlarged_plate is None:
        return

    plate_height, plate_width = (
        enlarged_plate.shape[:2]
    )

    vehicle_center_x = int(
        (x1 + x2) / 2
    )

    overlay_x1 = int(
        vehicle_center_x
        - plate_width / 2
    )

    overlay_y2 = y1 - 25
    text_y2 = overlay_y2
    text_y1 = text_y2 - PLATE_TEXT_AREA_HEIGHT
    overlay_y1 = text_y1 - plate_height

    # Dacă nu există loc deasupra, mutăm blocul sub începutul imaginii.
    if overlay_y1 < 0:
        overlay_y1 = 10
        text_y1 = overlay_y1 + plate_height
        text_y2 = text_y1 + PLATE_TEXT_AREA_HEIGHT

    overlay_x1 = max(
        0,
        min(
            frame_width - plate_width,
            overlay_x1
        )
    )

    overlay_x2 = overlay_x1 + plate_width
    overlay_y2_for_crop = overlay_y1 + plate_height

    if (
        overlay_x2 > frame_width
        or overlay_y2_for_crop > frame_height
        or text_y2 > frame_height
    ):
        return

    frame[
        overlay_y1:overlay_y2_for_crop,
        overlay_x1:overlay_x2
    ] = enlarged_plate

    cv2.rectangle(
        frame,
        (overlay_x1, text_y1),
        (overlay_x2, text_y2),
        (255, 255, 255),
        -1
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            plate_text,
            font,
            font_scale,
            thickness
        )
    )

    while (
        text_width > plate_width - 20
        and font_scale > 0.5
    ):
        font_scale -= 0.1

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                plate_text,
                font,
                font_scale,
                thickness
            )
        )

    text_x = int(
        overlay_x1
        + (plate_width - text_width) / 2
    )

    text_y = int(
        text_y1
        + (
            PLATE_TEXT_AREA_HEIGHT
            + text_height
        ) / 2
        - baseline
    )

    cv2.putText(
        frame,
        plate_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def run_visualization(
    video_path: str = VIDEO_PATH,
    csv_path: str = INTERPOLATED_CSV_PATH,
    output_video_path: str = OUTPUT_VIDEO_PATH
) -> None:
    rows = read_csv_rows(csv_path)

    rows_by_frame = defaultdict(list)

    for row in rows:
        frame_number = int(
            float(row["frame_nmr"])
        )

        rows_by_frame[
            frame_number
        ].append(row)

    best_plates = (
        select_best_plate_per_vehicle(
            rows
        )
    )

    output_parent = Path(
        output_video_path
    ).parent

    output_parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Nu s-a putut deschide videoclipul: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = -1
    frame_delay = max(
        1,
        int(1000 / fps)
    )

    paused = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        current_rows = rows_by_frame.get(
            frame_number,
            []
        )

        for row in current_rows:
            car_id = int(
                float(row["car_id"])
            )

            car_box = json_to_box(
                row["car_bbox"]
            )

            plate_box = json_to_box(
                row["license_plate_bbox"]
            )

            if car_box is None:
                continue

            car_x1, car_y1, car_x2, car_y2 = map(
                int,
                car_box
            )

            draw_border(
                frame,
                (car_x1, car_y1),
                (car_x2, car_y2),
                color=VEHICLE_BORDER_COLOR,
                thickness=8,
                line_length_x=100,
                line_length_y=100
            )

            if plate_box is not None:
                px1, py1, px2, py2 = map(
                    int,
                    plate_box
                )

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    PLATE_BORDER_COLOR,
                    4
                )

            best_plate = best_plates.get(
                car_id
            )

            if best_plate is not None:
                overlay_plate_and_text(
                    frame=frame,
                    vehicle_box=car_box,
                    plate_crop=best_plate["crop"],
                    plate_text=best_plate[
                        "license_number"
                    ]
                )

        writer.write(frame)

        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        cv2.putText(
            display,
            "P = Pause/Resume | N = Next | Q = Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "ANPR Final Visualization",
            display
        )

        key = cv2.waitKey(
            0 if paused else frame_delay
        ) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            paused = not paused

        if paused:
            while True:
                pause_key = cv2.waitKey(0) & 0xFF

                if pause_key == ord("p"):
                    paused = False
                    break

                if pause_key == ord("n"):
                    break

                if pause_key == ord("q"):
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    print(
                        f"Videoclipul a fost salvat în: "
                        f"{output_video_path}"
                    )
                    return

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(
        f"Videoclipul ANPR final a fost salvat în: "
        f"{output_video_path}"
    )


if __name__ == "__main__":
    run_visualization()