"""
main.py
"""

# import cProfile
import argparse
import csv
import logging
import os
from datetime import datetime

from .dead_pixels import correct_dead_pixels
from .georef import align, prepare_layout
from .utils import load_geotiff, save_geojson, save_geotiff


def setup_logging(task_id):
    """
    Configures logging for the given task ID.
    """
    task_dir = os.path.join("tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)
    log_file = os.path.join(task_dir, "process.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def process(layout_crop_paths, crop_path):
    """
    Process image
    """
    logging.info("Loading crop.")
    crop = load_geotiff(crop_path, layout="hwc")

    logging.info("Correcting dead pixels.")
    corrected_img, bug_report = correct_dead_pixels(crop["data"])

    logging.info("Starting georeference procedure. Fixing start time.")
    start_time = datetime.now()
    aligned = align(layout_crop_paths, corrected_img, crop_path)

    end_time = datetime.now()
    logging.info("Fixing end time.")

    corners = aligned["corners"]

    return {
        "corners": corners,
        "start": start_time,
        "end": end_time,
        "corrected_img": corrected_img,
        "corrected_meta": crop["meta"],
        "bug_report": bug_report,
        "aligned": aligned,
    }


def main(layout_path, crop_path, task_id, add_suffix=False):
    """
    Main function
    """
    setup_logging(task_id)
    logging.info(
        "Starting main processing with layout %s and crop %s", layout_path, crop_path
    )

    crop_name = os.path.basename(crop_path)

    task_dir = os.path.join("tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)

    if add_suffix:
        coords_csv_file_path = os.path.join(
            task_dir, f"coords_{crop_name.replace('.tif', '.csv')}"
        )
        coords_txt_file_path = os.path.join(
            task_dir, f"coords_{crop_name.replace('.tif', '.txt')}"
        )
        geojson_file_path = os.path.join(
            task_dir, f"coords_{crop_name.replace('.tif', '.geojson')}"
        )
        bug_report_file_path = os.path.join(
            task_dir, f"bug_report_{crop_name.replace('.tif', '.csv')}"
        )
        corrected_file_path = os.path.join(task_dir, f"corrected_{crop_name}")
        aligned_file_path = os.path.join(task_dir, f"aligned_{crop_name}")
    else:
        coords_csv_file_path = os.path.join(task_dir, "coords.csv")
        coords_txt_file_path = os.path.join(task_dir, "coords.txt")
        geojson_file_path = os.path.join(task_dir, "coords.geojson")
        bug_report_file_path = os.path.join(task_dir, "bug_report.csv")
        corrected_file_path = os.path.join(task_dir, "corrected.tif")
        aligned_file_path = os.path.join(task_dir, "aligned.tif")

    logging.info("Loading layout model")
    layout_crop_paths = prepare_layout(layout_path)

    logging.info("Starting main processing")
    result = process(layout_crop_paths, crop_path)

    logging.info("Saving results")
    coords = result["corners"]
    bug_report = result["bug_report"]

    save_geojson(geojson_file_path, coords)

    save_geotiff(
        corrected_file_path,
        result["corrected_img"],
        result["corrected_meta"],
        layout="hwc",
    )

    save_geotiff(
        aligned_file_path,
        result["corrected_img"],
        result["aligned"]["meta"],
        layout="hwc",
    )

    with open(bug_report_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "row_number",
            "column_number",
            "channel_number",
            "dead_value",
            "corrected_value",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for line in bug_report:
            writer.writerow(line)

    with open(coords_csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "layout_name",
            "crop_name",
            "ul",
            "ur",
            "br",
            "bl",
            "crs",
            "start",
            "end",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerow(
            {
                "layout_name": os.path.basename(layout_path),
                "crop_name": crop_name,
                "ul": f"{coords[0][0]:.1f}; {coords[0][1]:.1f}",
                "ur": f"{coords[1][0]:.1f}; {coords[1][1]:.1f}",
                "br": f"{coords[2][0]:.1f}; {coords[2][1]:.1f}",
                "bl": f"{coords[3][0]:.1f}; {coords[3][1]:.1f}",
                "crs": "EPSG:32637",
                "start": result["start"].strftime("%Y-%m-%dT%H:%M:%S"),
                "end": result["end"].strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    with open(coords_txt_file_path, "w", encoding="utf-8") as txtfile:
        for coord in coords:
            txtfile.write(f"{coord[0]:.1f}; {coord[1]:.1f}\n")


def process_all_crops(layout_name, input_folder):
    """
    Process all crop files in a folder
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif"):
            print(file_name)
            crop_path = os.path.join(input_folder, file_name)
            task_id = "process_folder"
            main(layout_name, crop_path, task_id, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process satellite images")
    parser.add_argument("--crop_name", type=str, help="Path to the crop image")
    parser.add_argument(
        "--layout_name", type=str, required=True, help="Path to the layout image"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the input folder containing crop images",
    )
    args = parser.parse_args()

    # profiler = cProfile.Profile()
    # profiler.enable()

    if args.input_folder:
        process_all_crops(args.layout_name, args.input_folder)
    else:
        main(args.layout_name, args.crop_name, "process_crop", False)

    # profiler.disable()
    # profiler.dump_stats("profile_results.prof")
