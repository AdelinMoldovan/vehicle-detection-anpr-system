import csv
from collections import defaultdict
from typing import Dict, List

import numpy as np

from anpr.plate_utils import (
    box_to_json,
    json_to_box
)


RAW_CSV_PATH = "outputs/anpr/anpr_raw.csv"
INTERPOLATED_CSV_PATH = (
    "outputs/anpr/anpr_interpolated.csv"
)


def interpolate_box(
    target_frame: int,
    frame_numbers: np.ndarray,
    boxes: np.ndarray
) -> List[float]:
    return [
        float(
            np.interp(
                target_frame,
                frame_numbers,
                boxes[:, coordinate_index]
            )
        )
        for coordinate_index in range(4)
    ]


def run_interpolation(
    input_csv_path: str = RAW_CSV_PATH,
    output_csv_path: str = INTERPOLATED_CSV_PATH
) -> None:
    with open(
        input_csv_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    grouped_rows: Dict[int, List[dict]] = defaultdict(list)

    for row in rows:
        car_id = int(float(row["car_id"]))
        row["frame_nmr"] = int(float(row["frame_nmr"]))
        grouped_rows[car_id].append(row)

    interpolated_rows = []

    for car_id, car_rows in grouped_rows.items():
        car_rows.sort(
            key=lambda item: item["frame_nmr"]
        )

        original_by_frame = {
            row["frame_nmr"]: row
            for row in car_rows
        }

        valid_rows = []

        for row in car_rows:
            car_box = json_to_box(
                row["car_bbox"]
            )
            plate_box = json_to_box(
                row["license_plate_bbox"]
            )

            if car_box is None or plate_box is None:
                continue

            valid_rows.append(
                (
                    row["frame_nmr"],
                    car_box,
                    plate_box
                )
            )

        if not valid_rows:
            continue

        frame_numbers = np.array(
            [item[0] for item in valid_rows],
            dtype=float
        )

        car_boxes = np.array(
            [item[1] for item in valid_rows],
            dtype=float
        )

        plate_boxes = np.array(
            [item[2] for item in valid_rows],
            dtype=float
        )

        first_frame = int(frame_numbers.min())
        last_frame = int(frame_numbers.max())

        for frame_number in range(
            first_frame,
            last_frame + 1
        ):
            interpolated_car_box = interpolate_box(
                frame_number,
                frame_numbers,
                car_boxes
            )

            interpolated_plate_box = interpolate_box(
                frame_number,
                frame_numbers,
                plate_boxes
            )

            original_row = original_by_frame.get(
                frame_number
            )

            interpolated_rows.append({
                "frame_nmr": frame_number,
                "car_id": car_id,
                "car_bbox": box_to_json(
                    interpolated_car_box
                ),
                "license_plate_bbox": box_to_json(
                    interpolated_plate_box
                ),
                "license_plate_bbox_score": (
                    original_row[
                        "license_plate_bbox_score"
                    ]
                    if original_row
                    else "0"
                ),
                "license_number": (
                    original_row["license_number"]
                    if original_row
                    else ""
                ),
                "license_number_score": (
                    original_row[
                        "license_number_score"
                    ]
                    if original_row
                    else "0"
                ),
                "plate_crop_path": (
                    original_row["plate_crop_path"]
                    if original_row
                    else ""
                )
            })

    interpolated_rows.sort(
        key=lambda row: (
            int(row["frame_nmr"]),
            int(row["car_id"])
        )
    )

    fieldnames = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number",
        "license_number_score",
        "plate_crop_path"
    ]

    with open(
        output_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(interpolated_rows)

    print(
        f"Datele interpolate au fost salvate în: "
        f"{output_csv_path}"
    )


if __name__ == "__main__":
    run_interpolation()