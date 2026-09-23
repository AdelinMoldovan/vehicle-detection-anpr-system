from models.vehicle_detector import run_vehicle_detection


"""if __name__ == "__main__":
    run_vehicle_detection(
        video_path="videos/test_video.mp4",
        model_path="yolov8n.pt",
        #model_path="weights/plate_model.pt"
        #plate_model_path="weights/plate_model.pt"
    )

from anpr.plate_detector import run_anpr

if __name__ == "__main__":
    run_anpr(
        video_path="videos/test_video.mp4",
        vehicle_model_path="yolov8n.pt",
        plate_model_path="weights/plate_model.pt"
    )
    """
"""
import argparse

from anpr.detect import run_detection
from anpr.interpolate import run_interpolation
from anpr.visualize import run_visualization


VIDEO_PATH = "videos/test_video.mp4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "weights/plate_model.pt"

RAW_CSV_PATH = "outputs/anpr/anpr_raw.csv"

INTERPOLATED_CSV_PATH = (
    "outputs/anpr/anpr_interpolated.csv"
)

FINAL_VIDEO_PATH = (
    "outputs/anpr/anpr_final.mp4"
)


def run_all_steps():
    run_detection(
        video_path=VIDEO_PATH,
        vehicle_model_path=VEHICLE_MODEL_PATH,
        plate_model_path=PLATE_MODEL_PATH,
        output_csv_path=RAW_CSV_PATH
    )

    run_interpolation(
        input_csv_path=RAW_CSV_PATH,
        output_csv_path=INTERPOLATED_CSV_PATH
    )

    run_visualization(
        video_path=VIDEO_PATH,
        csv_path=INTERPOLATED_CSV_PATH,
        output_video_path=FINAL_VIDEO_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Romanian ANPR pipeline"
    )

    parser.add_argument(
        "--step",
        choices=[
            "detect",
            "interpolate",
            "visualize",
            "all"
        ],
        default="all"
    )

    args = parser.parse_args()

    if args.step == "detect":
        run_detection(
            video_path=VIDEO_PATH,
            vehicle_model_path=VEHICLE_MODEL_PATH,
            plate_model_path=PLATE_MODEL_PATH,
            output_csv_path=RAW_CSV_PATH
        )

    elif args.step == "interpolate":
        run_interpolation(
            input_csv_path=RAW_CSV_PATH,
            output_csv_path=INTERPOLATED_CSV_PATH
        )

    elif args.step == "visualize":
        run_visualization(
            video_path=VIDEO_PATH,
            csv_path=INTERPOLATED_CSV_PATH,
            output_video_path=FINAL_VIDEO_PATH
        )

    else:
        run_all_steps()

from models.vehicle_detector import run_vehicle_detection


if __name__ == "__main__":
    run_vehicle_detection(
        video_path="videos/test_video.mp4",
        model_path="yolov8n.pt"
    )
   """
from combined.combined_detector import run_combined_detection


if __name__ == "__main__":
    run_combined_detection(
        video_path="videos/test_video.mp4",
        vehicle_model_path="yolov8n.pt",
        plate_model_path="weights/plate_model.pt"
    )

