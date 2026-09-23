import cv2


# ========================== CONFIGURARE ==========================

DEFAULT_SPEED_LIMIT = 80.0

# Dimensiunea crop-ului mărit al plăcuței
PLATE_DISPLAY_HEIGHT = 120

# Zona albă în care apare textul plăcuței
PLATE_TEXT_AREA_HEIGHT = 75

# Culori OpenCV = BGR
SAFE_COLOR = (0, 255, 0)       # verde
SPEEDING_COLOR = (0, 0, 255)   # roșu
PLATE_BOX_COLOR = (0, 255, 255)  # galben

TEXT_COLOR = (255, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# ================================================================


def draw_corner_border(
    frame,
    box,
    color,
    thickness=6,
    line_length=70
):
    """
    Desenează doar colțurile box-ului vehiculului,
    similar vizualizării ANPR.
    """

    x1, y1, x2, y2 = map(
        int,
        box
    )

    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )

    # Adaptăm lungimea colțurilor la dimensiunea vehiculului
    horizontal_length = min(
        line_length,
        max(
            15,
            int(box_width * 0.25)
        )
    )

    vertical_length = min(
        line_length,
        max(
            15,
            int(box_height * 0.25)
        )
    )

    # Stânga sus
    cv2.line(
        frame,
        (x1, y1),
        (x1 + horizontal_length, y1),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + vertical_length),
        color,
        thickness
    )

    # Dreapta sus
    cv2.line(
        frame,
        (x2, y1),
        (x2 - horizontal_length, y1),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + vertical_length),
        color,
        thickness
    )

    # Stânga jos
    cv2.line(
        frame,
        (x1, y2),
        (x1 + horizontal_length, y2),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - vertical_length),
        color,
        thickness
    )

    # Dreapta jos
    cv2.line(
        frame,
        (x2, y2),
        (x2 - horizontal_length, y2),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - vertical_length),
        color,
        thickness
    )


def draw_vehicle_label(
    frame,
    box,
    tracker_id,
    vehicle_type,
    speed_kmh,
    speed_limit=DEFAULT_SPEED_LIMIT
):
    """
    Desenează eticheta vehiculului.

    Exemplu:
        ID 7 | car | 72.4 km/h
    """

    x1, y1, _, _ = map(
        int,
        box
    )

    if speed_kmh is None:
        label = (
            f"ID {tracker_id} | "
            f"{vehicle_type} | "
            f"measuring..."
        )

        color = SAFE_COLOR

    else:
        speed_kmh = float(
            speed_kmh
        )

        is_speeding = (
            speed_kmh >= speed_limit
        )

        color = (
            SPEEDING_COLOR
            if is_speeding
            else SAFE_COLOR
        )

        label = (
            f"ID {tracker_id} | "
            f"{vehicle_type} | "
            f"{speed_kmh:.1f} km/h"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    padding_x = 8
    padding_y = 7

    label_height = (
        text_height
        + baseline
        + padding_y * 2
    )

    # Deasupra vehiculului dacă există spațiu
    if y1 - label_height >= 0:
        label_top = (
            y1 - label_height
        )

        label_bottom = y1

    else:
        label_top = y1

        label_bottom = (
            y1 + label_height
        )

    label_right = min(
        frame.shape[1] - 1,
        x1
        + text_width
        + padding_x * 2
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_top
        ),
        (
            label_right,
            label_bottom
        ),
        color,
        -1
    )

    text_y = (
        label_bottom
        - padding_y
        - baseline
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + padding_x,
            text_y
        ),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA
    )

    return color


def draw_vehicle(
    frame,
    box,
    tracker_id,
    vehicle_type,
    speed_kmh,
    speed_limit=DEFAULT_SPEED_LIMIT
):
    """
    Desenează vehiculul complet:
    - colțuri roșii/verzi;
    - eticheta cu ID, tip și viteză.
    """

    if speed_kmh is not None:
        is_speeding = (
            float(speed_kmh)
            >= float(speed_limit)
        )
    else:
        is_speeding = False

    color = (
        SPEEDING_COLOR
        if is_speeding
        else SAFE_COLOR
    )

    draw_corner_border(
        frame=frame,
        box=box,
        color=color
    )

    draw_vehicle_label(
        frame=frame,
        box=box,
        tracker_id=tracker_id,
        vehicle_type=vehicle_type,
        speed_kmh=speed_kmh,
        speed_limit=speed_limit
    )


def draw_plate_box(
    frame,
    plate_box
):
    """
    Desenează box-ul plăcuței.
    """

    if plate_box is None:
        return

    x1, y1, x2, y2 = map(
        int,
        plate_box
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        PLATE_BOX_COLOR,
        3
    )


def resize_plate_crop(
    plate_crop,
    target_height=PLATE_DISPLAY_HEIGHT
):
    """
    Redimensionează crop-ul plăcuței păstrând
    raportul de aspect.
    """

    if plate_crop is None:
        return None

    if plate_crop.size == 0:
        return None

    crop_height, crop_width = (
        plate_crop.shape[:2]
    )

    if (
        crop_height <= 0
        or crop_width <= 0
    ):
        return None

    scale = (
        target_height
        / crop_height
    )

    target_width = max(
        1,
        int(
            crop_width
            * scale
        )
    )

    enlarged_plate = cv2.resize(
        plate_crop,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return enlarged_plate


def draw_plate_overlay(
    frame,
    vehicle_box,
    plate_crop,
    plate_text
):
    """
    Desenează deasupra vehiculului:

        [ crop plăcuță mărit ]
        [     CJ20SLI       ]

    Dacă nu există suficient spațiu deasupra,
    încearcă să poziționeze overlay-ul sub vehicul.
    """

    if not plate_text:
        return

    enlarged_plate = resize_plate_crop(
        plate_crop
    )

    if enlarged_plate is None:
        return

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = map(
        int,
        vehicle_box
    )

    plate_height, plate_width = (
        enlarged_plate.shape[:2]
    )

    total_overlay_height = (
        plate_height
        + PLATE_TEXT_AREA_HEIGHT
    )

    vehicle_center_x = int(
        (x1 + x2) / 2
    )

    overlay_x1 = int(
        vehicle_center_x
        - plate_width / 2
    )

    # Poziția preferată: deasupra vehiculului
    overlay_y2 = y1 - 20

    overlay_y1 = (
        overlay_y2
        - total_overlay_height
    )

    # Dacă nu avem loc deasupra,
    # încercăm sub vehicul.
    if overlay_y1 < 0:
        overlay_y1 = y2 + 20

    # Limităm orizontal
    overlay_x1 = max(
        0,
        min(
            frame_width - plate_width,
            overlay_x1
        )
    )

    overlay_x2 = (
        overlay_x1
        + plate_width
    )

    plate_y1 = overlay_y1

    plate_y2 = (
        plate_y1
        + plate_height
    )

    text_y1 = plate_y2

    text_y2 = (
        text_y1
        + PLATE_TEXT_AREA_HEIGHT
    )

    # Dacă nici sub vehicul nu încape,
    # nu desenăm overlay-ul.
    if (
        plate_y1 < 0
        or text_y2 > frame_height
        or overlay_x1 < 0
        or overlay_x2 > frame_width
    ):
        return

    # Crop mărit
    frame[
        plate_y1:plate_y2,
        overlay_x1:overlay_x2
    ] = enlarged_plate

    # Fundal alb pentru text
    cv2.rectangle(
        frame,
        (
            overlay_x1,
            text_y1
        ),
        (
            overlay_x2,
            text_y2
        ),
        WHITE,
        -1
    )

    # Textul plăcuței
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        plate_text,
        font,
        font_scale,
        thickness
    )

    # Micșorăm textul dacă este prea lat
    while (
        text_width
        > plate_width - 20
        and font_scale > 0.5
    ):
        font_scale -= 0.1

        (
            text_width,
            text_height
        ), baseline = (
            cv2.getTextSize(
                plate_text,
                font,
                font_scale,
                thickness
            )
        )

    text_x = int(
        overlay_x1
        + (
            plate_width
            - text_width
        ) / 2
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
        (
            text_x,
            text_y
        ),
        font,
        font_scale,
        BLACK,
        thickness,
        cv2.LINE_AA
    )


def draw_statistics_panel(
    frame,
    cars_count,
    buses_count,
    vehicles_in,
    vehicles_out,
    average_speed,
    speeding_count,
    speed_limit
):
    """
    Panoul statistic principal.
    """

    panel_x1 = 20
    panel_y1 = 20

    panel_width = 420
    panel_height = 245

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x1,
            panel_y1
        ),
        (
            panel_x1 + panel_width,
            panel_y1 + panel_height
        ),
        (0, 0, 0),
        -1
    )

    # Transparanță
    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    lines = [
        f"Cars: {cars_count}",
        f"Buses: {buses_count}",
        f"Vehicles IN: {vehicles_in}",
        f"Vehicles OUT: {vehicles_out}",
        f"Average speed: {average_speed:.1f} km/h",
        (
            f"Speeding (>={speed_limit:.0f} km/h): "
            f"{speeding_count}"
        )
    ]

    start_y = 55
    line_spacing = 34

    for index, line in enumerate(
        lines
    ):
        cv2.putText(
            frame,
            line,
            (
                40,
                start_y
                + index
                * line_spacing
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WHITE,
            2,
            cv2.LINE_AA
        )


def draw_controls(
    frame,
    paused=False
):
    """
    Afișează comenzile disponibile.
    """

    if paused:
        text = (
            "PAUSED | "
            "P = Resume | "
            "N = Next frame | "
            "Q = Save & Quit"
        )
    else:
        text = (
            "P = Pause | "
            "Q = Save & Quit"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x = 20

    y = (
        frame.shape[0]
        - 25
    )

    cv2.rectangle(
        frame,
        (
            x - 8,
            y - text_height - 10
        ),
        (
            x + text_width + 8,
            y + baseline + 5
        ),
        BLACK,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA
    )


def draw_combined_frame(
    frame,
    tracked_vehicles,
    result_manager,
    anpr_manager,
    statistics,
    speed_limit=DEFAULT_SPEED_LIMIT,
    paused=False
):
    """
    Funcția principală de vizualizare.

    tracked_vehicles:

    [
        (
            tracker_id,
            vehicle_type,
            x1,
            y1,
            x2,
            y2
        ),
        ...
    ]

    statistics:

    {
        "cars": 10,
        "buses": 1,
        "in": 8,
        "out": 3,
        "average_speed": 61.5,
        "speeding": 4
    }
    """

    # ------------------------------------------------------
    # VEHICULE + ANPR
    # ------------------------------------------------------

    for (
        tracker_id,
        vehicle_type,
        x1,
        y1,
        x2,
        y2
    ) in tracked_vehicles:

        vehicle_box = (
            x1,
            y1,
            x2,
            y2
        )

        vehicle_result = (
            result_manager.get_vehicle(
                tracker_id
            )
        )

        speed_kmh = None

        if vehicle_result is not None:
            speed_kmh = (
                vehicle_result.get(
                    "speed_kmh"
                )
            )

        # Vehicul roșu / verde
        draw_vehicle(
            frame=frame,
            box=vehicle_box,
            tracker_id=tracker_id,
            vehicle_type=vehicle_type,
            speed_kmh=speed_kmh,
            speed_limit=speed_limit
        )

        # Rezultatul ANPR pentru același tracker_id
        anpr_result = (
            anpr_manager.get_result(
                tracker_id
            )
        )

        if anpr_result is None:
            continue

        # Box-ul plăcuței
        draw_plate_box(
            frame,
            anpr_result.get(
                "plate_box"
            )
        )

        # Crop + text mărit
        draw_plate_overlay(
            frame=frame,
            vehicle_box=vehicle_box,
            plate_crop=anpr_result.get(
                "crop"
            ),
            plate_text=anpr_result.get(
                "license_plate"
            )
        )

    # ------------------------------------------------------
    # STATISTICI
    # ------------------------------------------------------

    draw_statistics_panel(
        frame=frame,
        cars_count=statistics.get(
            "cars",
            0
        ),
        buses_count=statistics.get(
            "buses",
            0
        ),
        vehicles_in=statistics.get(
            "in",
            0
        ),
        vehicles_out=statistics.get(
            "out",
            0
        ),
        average_speed=statistics.get(
            "average_speed",
            0.0
        ),
        speeding_count=statistics.get(
            "speeding",
            0
        ),
        speed_limit=speed_limit
    )

    # ------------------------------------------------------
    # CONTROALE
    # ------------------------------------------------------

    draw_controls(
        frame,
        paused=paused
    )

    return frame