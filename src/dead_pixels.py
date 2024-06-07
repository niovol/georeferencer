"""
dead_pixels.py
"""

import os

import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from sklearn.linear_model import LinearRegression

from .utils import load_geotiff, save_geotiff


def compute_local_min_max_exclude_center(image):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    local_mins = np.zeros_like(image)
    local_maxs = np.zeros_like(image)
    for c in range(image.shape[2]):
        local_mins[:, :, c] = minimum_filter(
            image[:, :, c], footprint=kernel, mode="reflect"
        )
        local_maxs[:, :, c] = maximum_filter(
            image[:, :, c], footprint=kernel, mode="reflect"
        )
    return local_mins, local_maxs


def identify_dead_pixels(image, local_mins, local_maxs):
    image_mean = image.mean(axis=(0, 1))

    zero_pixels = image == 0
    underlit_pixels = image < (local_mins - 0.5 * image_mean)
    overlit_pixels = image > (local_maxs + 1.0 * image_mean)

    dead_mask = zero_pixels | underlit_pixels | overlit_pixels
    dead_pixels = np.argwhere(dead_mask)

    return dead_pixels, dead_mask


def interpolate_dead_pixel(image, row, col, channel):
    original_value = image[row, col, channel]

    neighborhood = image[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2, channel]
    mask = np.ones_like(neighborhood, dtype=bool)
    local_row, local_col = row - max(0, row - 1), col - max(0, col - 1)
    mask[local_row, local_col] = False

    neighborhood = neighborhood[mask]
    neighborhood = neighborhood[neighborhood != 0]

    if neighborhood.size > 0:
        return np.uint16(neighborhood.mean())
    return original_value


def local_regression(image, row, col, channel, dead_mask, window_size=9):
    half_window = window_size // 2
    row_start = max(0, row - half_window)
    row_end = min(image.shape[0], row + half_window + 1)
    col_start = max(0, col - half_window)
    col_end = min(image.shape[1], col + half_window + 1)

    x, y = [], []

    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            if r == row and c == col:
                continue
            if image[r, c, channel] != 0:
                x.append(image[r, c, ~dead_mask[row, col]])
                y.append(image[r, c, channel])

    if len(y) > 0:
        x = np.array(x)
        y = np.array(y)
        reg = LinearRegression().fit(x, y)
        x_pred = image[row, col, ~dead_mask[row, col]].reshape(1, -1)
        return np.uint16(np.clip(reg.predict(x_pred)[0], 1, 65535))

    return None


def correct_dead_pixels(image, window_size=9):
    local_mins, local_maxs = compute_local_min_max_exclude_center(image)
    dead_pixels, dead_mask = identify_dead_pixels(image, local_mins, local_maxs)

    corrected_image = image.copy()
    bug_report = []

    for pixel in dead_pixels:
        row, col, channel = pixel
        original_value = corrected_image[row, col, channel]

        corrected_value = local_regression(
            corrected_image, row, col, channel, dead_mask, window_size
        )
        if corrected_value is None:
            corrected_value = interpolate_dead_pixel(corrected_image, row, col, channel)

        corrected_image[row, col, channel] = corrected_value
        bug_report.append(
            f"{row}; {col}; {channel+1}; {original_value}; {corrected_value}"
        )

    return corrected_image, bug_report


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif"):
            file_path = os.path.join(input_folder, file_name)
            loaded_crop = load_geotiff(file_path, layout="hwc")

            corrected_img, bug_report = correct_dead_pixels(loaded_crop["data"])
            output_path = os.path.join(output_folder, file_name)
            save_geotiff(output_path, corrected_img, loaded_crop["meta"], layout="hwc")

            bug_report_path = os.path.join(
                output_folder, file_name.replace(".tif", "_bug_report.txt")
            )
            with open(bug_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(bug_report))
