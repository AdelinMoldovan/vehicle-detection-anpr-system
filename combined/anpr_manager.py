from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from anpr.ocr import PlateOCR
from anpr.plate_utils import (
    assign_plate_to_vehicle,
    clamp_box
)


class ANPRManager:
    """
    Modul ANPR pentru pipeline-ul combinat.

    IMPORTANT:
    Acest modul NU detectează și NU urmărește vehicule.

    Primește tracker_id-urile deja generate de trackerul
    principal al sistemului.
    """

    def __init__(
        self,
        plate_model_path: str,
        output_crop_dir: str = (
            "outputs/combined/crops"
        ),
        plate_confidence: float = 0.20,
        inference_size: int = 1280,
        frame_interval: int = 3,
        ocr_zone_ratio: float = 0.35,
        padding_x: int = 10,
        padding_y: int = 6
    ) -> None:

        # Model YOLO pentru plăcuțe
        self.plate_model = YOLO(
            plate_model_path
        )

        # OCR-ul românesc deja existent
        self.ocr = PlateOCR(
            use_gpu=True
        )

        self.plate_confidence = float(
            plate_confidence
        )

        self.inference_size = int(
            inference_size
        )

        self.frame_interval = max(
            1,
            int(frame_interval)
        )

        self.ocr_zone_ratio = float(
            ocr_zone_ratio
        )

        self.padding_x = int(
            padding_x
        )

        self.padding_y = int(
            padding_y
        )

        # Folder pentru cele mai bune crop-uri
        self.output_crop_dir = Path(
            output_crop_dir
        )

        self.output_crop_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Curățăm crop-urile vechi la o rulare nouă
        for crop_file in (
            self.output_crop_dir.glob("*")
        ):
            if crop_file.is_file():
                crop_file.unlink()

        # tracker_id -> [(text, confidence), ...]
        self.plate_votes = defaultdict(
            list
        )

        # tracker_id -> datele celui mai bun crop
        self.best_crops: Dict[int, dict] = {}

        # tracker_id -> ultimul box al plăcuței
        self.last_plate_boxes: Dict[
            int,
            Tuple[int, int, int, int]
        ] = {}

    def process_frame(
        self,
        original_frame,
        frame_index: int,
        tracked_vehicles: List[
            Tuple[
                int,
                str,
                float,
                float,
                float,
                float
            ]
        ]
    ) -> Dict[int, dict]:
        """
        Procesează ANPR pentru cadrul curent.

        tracked_vehicles trebuie să fie:

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

        Returnează rezultatele ANPR curente:
            tracker_id -> date ANPR
        """

        # Nu rulăm plate YOLO pe fiecare frame.
        if (
            frame_index
            % self.frame_interval
            != 0
        ):
            return self.get_all_results()

        if original_frame is None:
            return self.get_all_results()

        if original_frame.size == 0:
            return self.get_all_results()

        frame_height, frame_width = (
            original_frame.shape[:2]
        )

        minimum_plate_y = int(
            frame_height
            * self.ocr_zone_ratio
        )

        # Convertim formatul pentru funcția existentă
        # assign_plate_to_vehicle().
        #
        # Format:
        # (
        #     tracker_id,
        #     x1,
        #     y1,
        #     x2,
        #     y2
        # )

        vehicles_for_assignment = []

        vehicle_types = {}

        for (
            tracker_id,
            vehicle_type,
            x1,
            y1,
            x2,
            y2
        ) in tracked_vehicles:

            tracker_id = int(
                tracker_id
            )

            vehicles_for_assignment.append(
                (
                    tracker_id,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                )
            )

            vehicle_types[
                tracker_id
            ] = vehicle_type

        if not vehicles_for_assignment:
            return self.get_all_results()

        # --------------------------------------------------
        # Detectarea plăcuțelor pe frame-ul ORIGINAL
        # --------------------------------------------------

        plate_result = self.plate_model(
            original_frame,
            conf=self.plate_confidence,
            imgsz=self.inference_size,
            verbose=False
        )[0]

        if plate_result.boxes is None:
            return self.get_all_results()

        for detected_plate in (
            plate_result.boxes
        ):

            detection_confidence = float(
                detected_plate.conf[0]
            )

            plate_box = (
                detected_plate.xyxy[0]
                .cpu()
                .tolist()
            )

            px1, py1, px2, py2 = (
                plate_box
            )

            # Procesăm doar când plăcuța este
            # suficient de aproape de cameră.
            if py2 < minimum_plate_y:
                continue

            tracker_id = (
                assign_plate_to_vehicle(
                    plate_box,
                    vehicles_for_assignment
                )
            )

            if tracker_id is None:
                continue

            tracker_id = int(
                tracker_id
            )

            self.last_plate_boxes[
                tracker_id
            ] = (
                int(px1),
                int(py1),
                int(px2),
                int(py2)
            )

            crop_box = clamp_box(
                (
                    px1 - self.padding_x,
                    py1 - self.padding_y,
                    px2 + self.padding_x,
                    py2 + self.padding_y
                ),
                frame_width,
                frame_height
            )

            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2
            ) = crop_box

            if (
                crop_x2 <= crop_x1
                or crop_y2 <= crop_y1
            ):
                continue

            plate_crop = original_frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if plate_crop.size == 0:
                continue

            # --------------------------------------------------
            # OCR
            # --------------------------------------------------

            plate_text, ocr_confidence = (
                self.ocr.read_plate(
                    plate_crop
                )
            )

            if not plate_text:
                continue

            ocr_confidence = float(
                ocr_confidence
            )

            # Salvăm votul
            self.plate_votes[
                tracker_id
            ].append(
                (
                    plate_text,
                    ocr_confidence
                )
            )

            # Actualizăm cel mai bun crop pentru
            # ACEASTĂ CITIRE.
            self._update_best_crop(
                tracker_id=tracker_id,
                plate_text=plate_text,
                plate_crop=plate_crop,
                ocr_confidence=ocr_confidence,
                detection_confidence=(
                    detection_confidence
                ),
                frame_index=frame_index
            )

        return self.get_all_results()

    def _get_winning_text(
        self,
        tracker_id: int
    ) -> Optional[str]:
        """
        Alege textul care apare cel mai des.

        La egalitate, câștigă candidatul cu
        confidence-ul mediu mai mare.
        """

        votes = self.plate_votes.get(
            int(tracker_id),
            []
        )

        if not votes:
            return None

        counts = Counter(
            text
            for text, _ in votes
        )

        highest_vote_count = max(
            counts.values()
        )

        tied_texts = [
            text
            for text, count
            in counts.items()
            if count == highest_vote_count
        ]

        if len(tied_texts) == 1:
            return tied_texts[0]

        # Egalitate la numărul de voturi:
        # alegem media confidence mai mare.
        best_text = None
        best_average_confidence = -1.0

        for text in tied_texts:

            confidences = [
                confidence
                for vote_text, confidence
                in votes
                if vote_text == text
            ]

            average_confidence = (
                sum(confidences)
                / len(confidences)
            )

            if (
                average_confidence
                > best_average_confidence
            ):
                best_text = text
                best_average_confidence = (
                    average_confidence
                )

        return best_text

    def _update_best_crop(
        self,
        tracker_id: int,
        plate_text: str,
        plate_crop,
        ocr_confidence: float,
        detection_confidence: float,
        frame_index: int
    ) -> None:
        """
        Păstrează cel mai bun crop separat pentru
        fiecare text candidat.

        Astfel, dacă voting-ul final schimbă câștigătorul,
        avem cel mai bun crop pentru textul câștigător.
        """

        tracker_id = int(
            tracker_id
        )

        if tracker_id not in self.best_crops:
            self.best_crops[
                tracker_id
            ] = {}

        current = self.best_crops[
            tracker_id
        ].get(
            plate_text
        )

        if (
            current is not None
            and current[
                "ocr_confidence"
            ] >= ocr_confidence
        ):
            return

        crop_filename = (
            f"tracker_{tracker_id}_"
            f"{plate_text}.jpg"
        )

        crop_path = (
            self.output_crop_dir
            / crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            plate_crop
        )

        self.best_crops[
            tracker_id
        ][
            plate_text
        ] = {
            "crop": plate_crop.copy(),
            "crop_path": str(
                crop_path
            ).replace("\\", "/"),

            "ocr_confidence": (
                ocr_confidence
            ),

            "detection_confidence": (
                detection_confidence
            ),

            "frame_index": int(
                frame_index
            )
        }

    def get_result(
        self,
        tracker_id: int
    ) -> Optional[dict]:
        """
        Returnează rezultatul ANPR actual pentru
        un tracker.
        """

        tracker_id = int(
            tracker_id
        )

        winning_text = (
            self._get_winning_text(
                tracker_id
            )
        )

        if not winning_text:
            return None

        votes = self.plate_votes.get(
            tracker_id,
            []
        )

        winning_confidences = [
            confidence
            for text, confidence in votes
            if text == winning_text
        ]

        vote_count = len(
            winning_confidences
        )

        average_confidence = (
            sum(winning_confidences)
            / vote_count
        )

        crop_data = (
            self.best_crops
            .get(
                tracker_id,
                {}
            )
            .get(
                winning_text
            )
        )

        result = {
            "license_plate": winning_text,
            "plate_confidence": (
                average_confidence
            ),
            "vote_count": vote_count,

            "plate_box": (
                self.last_plate_boxes.get(
                    tracker_id
                )
            ),

            "crop": None,
            "crop_path": "",
            "best_frame": None
        }

        if crop_data is not None:
            result["crop"] = (
                crop_data["crop"]
            )

            result["crop_path"] = (
                crop_data["crop_path"]
            )

            result["best_frame"] = (
                crop_data["frame_index"]
            )

        return result

    def get_all_results(
        self
    ) -> Dict[int, dict]:
        """
        Returnează toate rezultatele ANPR curente.
        """

        results = {}

        for tracker_id in (
            self.plate_votes.keys()
        ):
            result = self.get_result(
                tracker_id
            )

            if result is not None:
                results[
                    int(tracker_id)
                ] = result

        return results