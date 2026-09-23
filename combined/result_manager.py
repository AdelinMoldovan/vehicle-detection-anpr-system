import csv
from pathlib import Path
from typing import Dict, Optional


class ResultManager:
    """
    Păstrează rezultatul final pentru fiecare vehicul urmărit.

    tracker_id:
        ID-ul intern primit de la ByteTrack.

    vehicle_id:
        ID consecutiv pentru rezultatul final:
        1, 2, 3, 4...
    """

    def __init__(
        self,
        speed_limit: float = 80.0
    ) -> None:
        self.speed_limit = float(speed_limit)

        # tracker_id -> date vehicul
        self.vehicles: Dict[int, dict] = {}

        # Numerotare consecutivă pentru CSV
        self.next_vehicle_id = 1

    def ensure_vehicle(
        self,
        tracker_id: int,
        vehicle_type: str
    ) -> dict:
        """
        Creează vehiculul dacă nu există deja.
        """

        tracker_id = int(tracker_id)

        if tracker_id not in self.vehicles:
            self.vehicles[tracker_id] = {
                "vehicle_id": self.next_vehicle_id,
                "tracker_id": tracker_id,
                "vehicle_type": vehicle_type,

                "direction": "",

                "speed_kmh": None,
                "speed_limit": self.speed_limit,
                "speeding": False,

                "license_plate": "",
                "plate_confidence": 0.0,
                "plate_votes": 0
            }

            self.next_vehicle_id += 1

        else:
            # Actualizăm tipul dacă este disponibil.
            if vehicle_type:
                self.vehicles[
                    tracker_id
                ]["vehicle_type"] = vehicle_type

        return self.vehicles[tracker_id]

    def update_speed(
        self,
        tracker_id: int,
        vehicle_type: str,
        speed_kmh: Optional[float]
    ) -> None:
        """
        Actualizează viteza vehiculului.

        Păstrăm ultima viteză validă furnizată de sistemul
        principal de speed estimation.
        """

        vehicle = self.ensure_vehicle(
            tracker_id,
            vehicle_type
        )

        if speed_kmh is None:
            return

        speed_kmh = float(speed_kmh)

        if speed_kmh < 0:
            return

        vehicle["speed_kmh"] = speed_kmh

        vehicle["speeding"] = (
            speed_kmh >= self.speed_limit
        )

    def update_direction(
        self,
        tracker_id: int,
        vehicle_type: str,
        direction: str
    ) -> None:
        """
        Salvează direcția IN sau OUT.
        """

        if not direction:
            return

        vehicle = self.ensure_vehicle(
            tracker_id,
            vehicle_type
        )

        vehicle["direction"] = str(
            direction
        ).upper()

    def update_plate(
        self,
        tracker_id: int,
        vehicle_type: str,
        license_plate: Optional[str],
        confidence: float = 0.0,
        vote_count: int = 0
    ) -> None:
        """
        Actualizează rezultatul ANPR final pentru vehicul.
        """

        if not license_plate:
            return

        vehicle = self.ensure_vehicle(
            tracker_id,
            vehicle_type
        )

        vehicle["license_plate"] = str(
            license_plate
        )

        vehicle["plate_confidence"] = float(
            confidence
        )

        vehicle["plate_votes"] = int(
            vote_count
        )

    def get_vehicle(
        self,
        tracker_id: int
    ) -> Optional[dict]:
        return self.vehicles.get(
            int(tracker_id)
        )

    def get_all_vehicles(self) -> Dict[int, dict]:
        return self.vehicles

    def save_csv(
        self,
        output_path: str
    ) -> None:
        """
        Scrie un singur rând pentru fiecare vehicul.
        """

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fieldnames = [
            "vehicle_id",
            "tracker_id",
            "vehicle_type",
            "direction",
            "speed_kmh",
            "speed_limit",
            "speeding",
            "license_plate",
            "plate_confidence",
            "plate_votes"
        ]

        vehicles_sorted = sorted(
            self.vehicles.values(),
            key=lambda vehicle: vehicle[
                "vehicle_id"
            ]
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for vehicle in vehicles_sorted:

                speed_kmh = vehicle[
                    "speed_kmh"
                ]

                if speed_kmh is None:
                    speed_value = ""
                else:
                    speed_value = round(
                        speed_kmh,
                        2
                    )

                writer.writerow({
                    "vehicle_id": vehicle[
                        "vehicle_id"
                    ],

                    "tracker_id": vehicle[
                        "tracker_id"
                    ],

                    "vehicle_type": vehicle[
                        "vehicle_type"
                    ],

                    "direction": vehicle[
                        "direction"
                    ],

                    "speed_kmh": speed_value,

                    "speed_limit": round(
                        vehicle[
                            "speed_limit"
                        ],
                        2
                    ),

                    "speeding": vehicle[
                        "speeding"
                    ],

                    "license_plate": vehicle[
                        "license_plate"
                    ],

                    "plate_confidence": round(
                        vehicle[
                            "plate_confidence"
                        ],
                        4
                    ),

                    "plate_votes": vehicle[
                        "plate_votes"
                    ]
                })

        print(
            f"CSV final salvat în: {output_file}"
        )