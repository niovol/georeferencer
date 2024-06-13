"""
georef.py
"""

import logging
import os
import pickle

import cv2
import numpy as np
from affine import Affine

from .superglue.models.matching import Matching
from .superglue.models.utils import frame2tensor
from .utils import (
    downscale,
    final_uint8,
    load_geotiff,
    slice_geotiff,
)


def save_model(filename, keypoints, descriptors, metas, layout_crop_filenames):
    """
    Saves the keypoints, descriptors, and meta information to a file.
    """
    with open(filename, "wb") as f:
        all_keypoints_serializable = []
        for keypoints_for_one_image in keypoints:
            keypoints_serializable = [
                [
                    (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    for kp in kp_list
                ]
                for kp_list in keypoints_for_one_image
            ]
            all_keypoints_serializable.append(keypoints_serializable)
        pickle.dump(
            (all_keypoints_serializable, descriptors, metas, layout_crop_filenames), f
        )


def load_model(filename):
    """
    Loads the keypoints, descriptors, and meta information from a file.
    """
    with open(filename, "rb") as f:
        all_keypoints_serializable, descriptors, metas, layout_crop_filenames = (
            pickle.load(f)
        )
        # Convert keypoints back to cv2.KeyPoint objects
        keypoints = [
            [
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
            for keypoints_serializable in all_keypoints_serializable
        ]
        return keypoints, descriptors, metas, layout_crop_filenames


def prepare_layout(layout_path):
    """
    Loads the layout info
    """

    # filename_sift = f"models/sift/{os.path.basename(layout_path)}.pkl"
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

    # save_model(filename_sift, keypoints, descriptors, metas, layout_crop_filenames)

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


def align(layout_crop_paths, crop_image):
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
    matching_config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 2048,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
            "descriptor_dim": 256,
        },
    }
    matching = Matching(matching_config).eval().to("cpu")

    all_matches = []
    for layout_crop_path in layout_crop_paths:
        loaded = load_geotiff(layout_crop_path, layout="hwc")
        inp0 = frame2tensor(
            np.squeeze(final_uint8(loaded["data"])).astype("float32"), "cpu"
        )
        inp1 = frame2tensor(
            np.squeeze(final_uint8(crop_image)).astype("float32"), "cpu"
        )

        pred = matching({"image0": inp0, "image1": inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
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
