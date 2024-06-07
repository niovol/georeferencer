"""
main.py
"""

import os
import argparse
import csv
from datetime import datetime

from dead_pixels import correct_dead_pixels
from georef import align, detect_keypoints_and_descriptors
from utils import load_geotiff, save_geotiff


def load_layout(layout_name):
    """
    Loads the layout file
    """
    loaded = load_geotiff(layout_name, layout="hwc")
    image, meta = loaded["data"], loaded["meta"]
    keypoints_list, descriptors_list = detect_keypoints_and_descriptors(image, "SIFT")

    return {
        "image": image,
        "meta": meta,
        "keypoints": keypoints_list,
        "descriptors": descriptors_list,
    }


def process(layout, crop_name):
    """
    Process image
    """
    crop = load_geotiff(crop_name, layout="hwc")
    corrected_img, bug_report = correct_dead_pixels(crop["data"])

    start_time = datetime.now()
    aligned = align(
        layout["keypoints"], layout["descriptors"], layout["meta"], corrected_img
    )
    end_time = datetime.now()

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


def process_folder(layout_name, input_folder, output_folder):
    """
    Process all crops in a folder
    """
    output_folder_corrected = os.path.join(output_folder, "corrected")
    output_folder_aligned = os.path.join(output_folder, "aligned")
    os.makedirs(output_folder_corrected, exist_ok=True)
    os.makedirs(output_folder_aligned, exist_ok=True)

    layout = load_layout(layout_name)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif"):
            print(file_name)

            crop_path = os.path.join(input_folder, file_name)
            output_path_corrected = os.path.join(output_folder_corrected, file_name)
            output_path_aligned = os.path.join(output_folder_aligned, file_name)

            result = process(layout, crop_path)

            save_geotiff(
                output_path_corrected,
                result["corrected_img"],
                result["corrected_meta"],
                layout="hwc",
            )

            bug_report_path = os.path.join(
                output_folder, file_name.replace(".tif", "_bug_report.txt")
            )
            with open(bug_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result["bug_report"]))

            save_geotiff(
                output_path_aligned,
                result["corrected_img"],
                result["aligned"]["meta"],
                layout="hwc",
            )


def main(layout_name, crop_name):
    """
    Main function
    """
    layout = load_layout(layout_name)
    result = process(layout, crop_name)
    coordinates = result["corners"]

    with open("coords.csv", "w", newline="", encoding="utf-8") as csvfile:
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
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "layout_name": layout_name,
                "crop_name": crop_name,
                "ul": f"{coordinates[0][0]}; {coordinates[0][1]}",
                "ur": f"{coordinates[1][0]}; {coordinates[1][1]}",
                "br": f"{coordinates[2][0]}; {coordinates[2][1]}",
                "bl": f"{coordinates[3][0]}; {coordinates[3][1]}",
                "crs": "EPSG:32637",
                "start": result["start"].strftime("%Y-%m-%dT%H:%M:%S"),
                "end": result["end"].strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process satellite images")
    parser.add_argument(
        "--crop_name", type=str, required=True, help="Path to the crop image"
    )
    parser.add_argument(
        "--layout_name", type=str, required=True, help="Path to the layout image"
    )
    args = parser.parse_args()
    main(args.layout_name, args.crop_name)
