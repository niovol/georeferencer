"""
main.py
"""

import argparse
import csv
import os
import pickle
import cv2
from datetime import datetime

from .dead_pixels import correct_dead_pixels
from .georef import align, detect_keypoints_and_descriptors
from .utils import load_geotiff, save_geotiff


def save_model(filename, keypoints, descriptors, meta):
    """
    Saves the keypoints, descriptors, and meta information to a file.
    """
    with open(filename, "wb") as f:
        # Convert keypoints to a serializable format
        keypoints_serializable = [
            [
                (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in kp_list
            ]
            for kp_list in keypoints
        ]
        pickle.dump((keypoints_serializable, descriptors, meta), f)


def load_model(filename):
    """
    Loads the keypoints, descriptors, and meta information from a file.
    """
    with open(filename, "rb") as f:
        keypoints_serializable, descriptors, meta = pickle.load(f)
        # Convert keypoints back to cv2.KeyPoint objects
        keypoints = [
            [
                cv2.KeyPoint(
                    x=kp[0][0],
                    y=kp[0][1],
                    size=kp[1],
                    angle=kp[2],
                    response=kp[3],
                    octave=kp[4],
                    class_id=kp[5],
                )
                for kp in kp_list
            ]
            for kp_list in keypoints_serializable
        ]
        return keypoints, descriptors, meta


def load_layout_info(layout_path):
    """
    Loads the layout info
    """

    filename = f"models/sift/{os.path.basename(layout_path)}.pkl"
    os.makedirs("models/sift", exist_ok=True)

    if os.path.exists(filename):
        keypoints, descriptors, meta = load_model(filename)
    else:
        loaded = load_geotiff(layout_path, layout="hwc")
        image, meta = loaded["data"], loaded["meta"]
        keypoints, descriptors = detect_keypoints_and_descriptors(image, "SIFT")
        save_model(filename, keypoints, descriptors, meta)

    return {
        "meta": meta,
        "keypoints": keypoints,
        "descriptors": descriptors,
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

    layout = load_layout_info(layout_name)

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


def main(layout_path, crop_path, task_id="."):
    """
    Main function
    """

    task_dir = os.path.join("tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)

    coords_file_path = os.path.join(task_dir, "coords.csv")
    bug_report_file_path = os.path.join(task_dir, "bug_report.csv")

    layout = load_layout_info(layout_path)
    result = process(layout, crop_path)
    coords = result["corners"]
    bug_report = result["bug_report"]

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

    with open(coords_file_path, "w", newline="", encoding="utf-8") as csvfile:
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
                "crop_name": os.path.basename(crop_path),
                "ul": f"{coords[0][0]:.3f}; {coords[0][1]:.3f}",
                "ur": f"{coords[1][0]:.3f}; {coords[1][1]:.3f}",
                "br": f"{coords[2][0]:.3f}; {coords[2][1]:.3f}",
                "bl": f"{coords[3][0]:.3f}; {coords[3][1]:.3f}",
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
