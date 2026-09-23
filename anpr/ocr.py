import re
from typing import Optional, Tuple

import cv2
import easyocr


ROMANIAN_PLATE_PATTERNS = [
    # București: B + 2/3 cifre + 3 litere
    re.compile(r"^B\d{2,3}[A-Z]{3}$"),

    # Județe: 2 litere + 2/3 cifre + 3 litere
    re.compile(r"^[A-Z]{2}\d{2,3}[A-Z]{3}$"),
]


def normalize_plate_text(text: str) -> str:
    """
    Elimină spații, cratime și simboluri, fără să înlocuiască
    litere cu cifre sau invers.
    """
    normalized = text.upper()
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return normalized


def is_valid_romanian_plate(text: str) -> bool:
    """
    Verifică formatele românești civile uzuale.

    Nu include numere diplomatice, militare, provizorii sau speciale.
    """
    return any(
        pattern.fullmatch(text)
        for pattern in ROMANIAN_PLATE_PATTERNS
    )


class PlateOCR:
    def __init__(
        self,
        use_gpu: bool = True,
        strict_romanian_format: bool = True
    ) -> None:
        self.strict_romanian_format = strict_romanian_format

        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu
        )

    @staticmethod
    def enhance_plate(plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return None

        enlarged = cv2.resize(
            plate_crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        enhanced = cv2.bilateralFilter(
            enhanced,
            7,
            50,
            50
        )

        return enhanced

    def read_plate(
        self,
        plate_crop
    ) -> Tuple[Optional[str], float]:
        enhanced = self.enhance_plate(plate_crop)

        if enhanced is None:
            return None, 0.0

        results = self.reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        best_text = None
        best_confidence = 0.0

        for _, raw_text, confidence in results:
            text = normalize_plate_text(raw_text)
            confidence = float(confidence)

            if not 5 <= len(text) <= 8:
                continue

            if confidence < 0.10:
                continue

            if (
                self.strict_romanian_format
                and not is_valid_romanian_plate(text)
            ):
                continue

            if confidence > best_confidence:
                best_text = text
                best_confidence = confidence

        return best_text, best_confidence