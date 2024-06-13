"""
georef.py
"""

import logging
import os
import pickle

import cv2
import numpy as np
import torch
from affine import Affine

from .superglue.models.superglue import SuperGlue
from .superglue.models.superpoint import SuperPoint
from .superglue.models.utils import frame2tensor
from .utils import downscale, final_uint8, load_geotiff, slice_geotiff


def save_keypoints(keypoints, keypoints_path):
    """
    Saves the keypoints, descriptors, and scores to a file.
    """
    with open(keypoints_path, "wb") as f:
        pickle.dump(keypoints, f)


def load_keypoints(filename):
    """
    Loads the keypoints, descriptors, and scores from a file.
    """
    with open(filename, "rb") as f:
        keypoints = pickle.load(f)
        return keypoints


def prepare_layout(layout_path):
    """
    Loads the layout info
    """
    os.makedirs("cache/layout_downscale", exist_ok=True)

    filename_downscale = f"cache/layout_downscale/{os.path.basename(layout_path)}"
    if not os.path.exists(filename_downscale):
        downscale(layout_path, filename_downscale, 1600, 1600)
        slice_geotiff(
            filename_downscale, "cache/layout_downscale_crops", (8, 5), (160, 100)
        )

    layout_crop_paths = []
    for i in range(5):
        for j in range(8):
            filename = os.path.basename(layout_path).replace(".tif", f"_{i}_{j}.tif")
            layout_crop_path = f"cache/layout_downscale_crops/{filename}"
            layout_crop_paths.append(layout_crop_path)

    return layout_crop_paths


def convert_affine_to_numpy(affine: Affine):
    """
    Converts an Affine object to a NumPy array.

    Args:
        affine (Affine): An Affine transformation object.

    Returns:
        numpy.ndarray: A 2x3 NumPy array representing the affine transformation.
    """
    return np.array([[affine.a, affine.b, affine.c], [affine.d, affine.e, affine.f]])


def convert_numpy_to_affine(array):
    """
    Converts a NumPy array to an Affine object.

    Args:
        array (numpy.ndarray): A 2x3 NumPy array representing the affine transformation.

    Returns:
        Affine: An Affine transformation object.
    """
    return Affine(
        array[0, 0], array[0, 1], array[0, 2], array[1, 0], array[1, 1], array[1, 2]
    )


def multiply_affine_arrays(array1, array2):
    """
    Multiplies two affine transformation arrays.

    Args:
        array1 (numpy.ndarray): The first affine transformation array.
        array2 (numpy.ndarray): The second affine transformation array.

    Returns:
        numpy.ndarray: The result of multiplying the two affine transformation arrays.
    """
    array1_3x3 = np.vstack([array1, [0, 0, 1]])
    array2_3x3 = np.vstack([array2, [0, 0, 1]])
    mult = np.dot(array2_3x3, array1_3x3)
    return mult[:2, :]


def align(layout_crop_paths, crop_image, crop_path):
    """
    Aligns a cropped image to a layout using keypoints and descriptors.

    Args:
        layout_keypoints (list): List of keypoints from the layout.
        layout_descriptors (list): List of descriptors from the layout.
        layout_meta (dict): Metadata of the layout, including the affine transformation.
        crop_image (numpy.ndarray): The cropped image to be aligned.

    Returns:
        dict: A dictionary containing the new corners and updated metadata.
    """

    torch.set_grad_enabled(False)

    keypoints_dir = "cache/keypoints"
    os.makedirs(keypoints_dir, exist_ok=True)

    superpoint_config = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048,
    }
    superglue_config = {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
        "descriptor_dim": 256,
    }
    superpoint = SuperPoint(superpoint_config).eval().to("cpu")
    superglue = SuperGlue(superglue_config).eval().to("cpu")

    all_matches = []
    for layout_crop_path in layout_crop_paths:
        loaded = load_geotiff(layout_crop_path, layout="hwc")
        inp0 = frame2tensor(
            np.squeeze(final_uint8(loaded["data"])).astype("float32"), "cpu"
        )
        inp1 = frame2tensor(
            np.squeeze(final_uint8(crop_image)).astype("float32"), "cpu"
        )

        keypoint0_path = os.path.join(
            keypoints_dir, os.path.basename(layout_crop_path).replace(".tif", ".pkl")
        )
        keypoint1_path = os.path.join(
            keypoints_dir, os.path.basename(crop_path).replace(".tif", ".pkl")
        )

        if os.path.exists(keypoint0_path):
            pred0 = load_keypoints(keypoint0_path)
        else:
            # pylint: disable=not-callable
            pred0 = {k + "0": v for k, v in superpoint({"image": inp0}).items()}
            save_keypoints(pred0, keypoint0_path)

        if os.path.exists(keypoint1_path):
            pred1 = load_keypoints(keypoint1_path)
        else:
            pred1 = {k + "1": v for k, v in superpoint({"image": inp1}).items()}
            save_keypoints(pred1, keypoint1_path)

        pred = {**pred0, **pred1}
        data = {"image0": inp0, "image1": inp1, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = {**pred, **superglue(data)}
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]

        valid_matches = matches > -1
        filtered_matches = matches[valid_matches]
        filtered_conf = conf[valid_matches]
        n_matches = np.sum(valid_matches)
        score = np.sum(filtered_conf)
        layout_points = kpts0[valid_matches]
        crop_points = kpts1[filtered_matches]

        # high_conf_matches = filtered_conf > conf_threshold
        # pts0 = pts0[high_conf_matches]
        # pts1 = pts1[high_conf_matches]

        logging.info("%s matches found, score=%s", n_matches, score)

        all_matches.append(
            {
                "n_matches": len(crop_points),
                "score": score,
                "layout": layout_points,
                "crop": crop_points,
                "meta": loaded["meta"],
            }
        )

    item_chosen = max(all_matches, key=lambda x: x["n_matches"])
    layout_meta = item_chosen["meta"]
    transf_matrix, _ = cv2.estimateAffine2D(
        np.array(item_chosen["crop"]),
        np.array(item_chosen["layout"]),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
    )

    h, w = crop_image.shape[:2]
    crop_corners_pixels = [[0, 0], [w, 0], [w, h], [0, h]]

    new_affine_array = multiply_affine_arrays(
        transf_matrix, convert_affine_to_numpy(layout_meta["transform"])
    )
    new_affine = convert_numpy_to_affine(new_affine_array)
    new_corners = [new_affine * corner_pixels for corner_pixels in crop_corners_pixels]
    print(f"Corners: {new_corners}")

    new_meta = layout_meta.copy()
    new_meta["transform"] = new_affine
    new_meta["width"] = w
    new_meta["height"] = h

    return {
        "corners": new_corners,
        "meta": new_meta,
        "layout_points": layout_points,
        "crop_points": crop_points,
    }
