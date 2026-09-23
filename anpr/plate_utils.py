import json
from typing import Iterable, Optional, Sequence, Tuple

import cv2


Box = Tuple[float, float, float, float]


def box_to_json(box: Sequence[float]) -> str:
    return json.dumps(
        [round(float(value), 2) for value in box]
    )


def json_to_box(value: str) -> Optional[Box]:
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        parsed = json.loads(value)

        if len(parsed) != 4:
            return None

        return tuple(float(item) for item in parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def clamp_box(
    box: Sequence[float],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(0, min(frame_width, int(x2)))
    y2 = max(0, min(frame_height, int(y2)))

    return x1, y1, x2, y2


def assign_plate_to_vehicle(
    plate_box: Sequence[float],
    tracked_vehicles: Iterable[
        Tuple[int, float, float, float, float]
    ]
) -> Optional[int]:
    """
    Returnează ID-ul vehiculului al cărui box conține
    centrul plăcuței.
    """
    px1, py1, px2, py2 = plate_box

    plate_center_x = (px1 + px2) / 2.0
    plate_center_y = (py1 + py2) / 2.0

    matches = []

    for tracker_id, vx1, vy1, vx2, vy2 in tracked_vehicles:
        contains_plate = (
            vx1 <= plate_center_x <= vx2
            and vy1 <= plate_center_y <= vy2
        )

        if not contains_plate:
            continue

        vehicle_area = max(
            1.0,
            (vx2 - vx1) * (vy2 - vy1)
        )

        matches.append(
            (vehicle_area, int(tracker_id))
        )

    if not matches:
        return None

    # Dacă apar două box-uri suprapuse, îl alegem pe cel mai mic.
    matches.sort(key=lambda item: item[0])

    return matches[0][1]


def draw_border(
    frame,
    top_left,
    bottom_right,
    color=(0, 255, 0),
    thickness=8,
    line_length_x=120,
    line_length_y=120
):
    """
    Desenează doar colțurile box-ului, similar vizualizării
    din repository-ul de referință.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(
        frame,
        (x1, y1),
        (x1 + line_length_x, y1),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + line_length_y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2 - line_length_x, y1),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + line_length_y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1 + line_length_x, y2),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - line_length_y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2 - line_length_x, y2),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - line_length_y),
        color,
        thickness
    )

    return frame