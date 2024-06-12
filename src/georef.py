"""
georef.py
"""

import logging
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from torch import layout

from .superglue.models.matching import Matching
from .superglue.models.utils import frame2tensor
from .utils import (
    downscale,
    equalize_hist,
    final_uint8,
    load_geotiff,
    megagray,
    save_geotiff,
    scale_image_percentile,
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


def load_layout_info(layout_path):
    """
    Loads the layout info
    """

    filename_sift = f"models/sift/{os.path.basename(layout_path)}.pkl"
    os.makedirs("models/sift", exist_ok=True)
    os.makedirs("models/downscale", exist_ok=True)
    os.makedirs("models/hist_equalize", exist_ok=True)
    os.makedirs("models/sobel", exist_ok=True)

    if os.path.exists(filename_sift):
        keypoints, descriptors, metas, layout_crop_filenames = load_model(filename_sift)
    else:
        filename_downscale = f"models/downscale/{os.path.basename(layout_path)}"
        if not os.path.exists(filename_downscale):
            downscale(layout_path, filename_downscale, 1600, 1600)

        # loaded = load_geotiff(filename_downscale, layout="hwc")
        # image, meta = loaded["data"], loaded["meta"]
        # final = final_uint8(image)

        slice_geotiff(filename_downscale, "models/downscale_crops", (8, 5), (160, 100))

        keypoints, descriptors, metas, layout_crop_filenames = [], [], [], []
        for i in range(5):
            for j in range(8):
                filename = os.path.basename(layout_path).replace(
                    ".tif", f"_{i}_{j}.tif"
                )
                layout_crop_pathname = f"models/downscale_crops/{filename}"
                loaded_downscale_crop = load_geotiff(layout_crop_pathname, layout="hwc")
                keypoints_detected, descriptors_detected, keypoints_image = (
                    detect_keypoints_and_descriptors(
                        final_uint8(loaded_downscale_crop["data"]), "SIFT"
                    )
                )
                keypoints.append(keypoints_detected)
                descriptors.append(descriptors_detected)
                metas.append(loaded_downscale_crop["meta"])
                layout_crop_filenames.append(layout_crop_pathname)

        # filename_final = f"models/final/{os.path.basename(layout_path)}"
        # new_meta = meta.copy()
        # new_meta["dtype"] = "uint8"
        # new_meta["count"] = 1
        # save_geotiff(filename_final, final, new_meta, layout="hwc")
        # slice_geotiff(
        #     filename_final, "models/final_crops", (8, 5), (160, 100), (0, 0, 0, 0)
        # )

        # final0 = load_geotiff(
        #     f"models/final_crops/{os.path.basename(layout_path).replace('.tif', '_0_0.tif')}",
        #     layout="hwc",
        # )["data"]

        save_model(filename_sift, keypoints, descriptors, metas, layout_crop_filenames)

        # new_meta = meta.copy()
        # new_meta["count"] = 3
        # save_geotiff(
        #     "tasks/keypoints_image.tif", keypoints_image, new_meta, layout="hwc"
        # )

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "metas": metas,
        "crop_filenames": layout_crop_filenames,
    }


def match_features(descriptors1, descriptors2, method="SIFT"):
    """
    Matches features between two sets of descriptors.

    Args:
        descriptors1 (numpy.ndarray): Descriptors from the first image.
        descriptors2 (numpy.ndarray): Descriptors from the second image.
        method (str): The feature extraction method used ('SIFT' or 'SuperGlue').

    Returns:
        matches: List of matched features.
    """
    if method == "SIFT":
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        return good_matches

    raise ValueError("`method` should be either 'SIFT' or 'SuperGlue'")


def detect_keypoints_and_descriptors(image, detector_type="SIFT", draw=False):
    """
    Detects keypoints and computes descriptors for the given image.

    Args:
        image (numpy.ndarray): Input image.
        detector_type (str): The type of feature detector to use ('SIFT' or 'SuperGlue').
        draw (bool): Whether to draw and display the keypoints on the image.

    Returns:
        tuple: Lists of keypoints and descriptors for each channel.
    """
    if detector_type == "SIFT":
        feature_detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    image_equalized = equalize_hist(image)

    keypoints_list = []
    descriptors_list = []
    for i in [20]:  # range(1):
        logging.info("Detecting keypoints and computing descriptors on channel %s", i)

        if i < 4:
            gray = image_equalized[:, :, i]
            # gray = scale_image_percentile(image[:, :, i])
        elif i == 10:
            gray = scale_image_percentile(megagray(image))
        elif i == 20:
            gray = image

        keypoints, descriptors = feature_detector.detectAndCompute(gray, None)
        logging.info("Found %s keypoints", len(keypoints))

        keypoints_image = cv2.drawKeypoints(
            gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        keypoints_image_rgb = cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB)

        if draw:
            plt.imshow(keypoints_image_rgb)
            plt.show()

        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list, keypoints_image_rgb


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


def align(layout, crop_image):
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

    keypoints_list_crop, descriptors_list_crop, _ = detect_keypoints_and_descriptors(
        final_uint8(crop_image), "SIFT"
    )

    layout_keypoints_all = layout["keypoints"]
    layout_descriptors_all = layout["descriptors"]
    layout_metas = layout["metas"]
    layout_crop_filenames = layout["crop_filenames"]
    all_matches = []
    for layout_keypoints, layout_descriptors, meta, layout_crop_filename in zip(
        layout_keypoints_all,
        layout_descriptors_all,
        layout_metas,
        layout_crop_filenames,
    ):
        crop_points = []
        layout_points = []
        for j in range(1):
            logging.info("%s channel", j)

            # matches = match_features(
            #    descriptors_list_crop[j], layout_descriptors[j], method="SIFT"
            # )
            loaded = load_geotiff(layout_crop_filename, layout="hwc")
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
            pts0 = kpts0[valid_matches]
            pts1 = kpts1[filtered_matches]

            # high_conf_matches = filtered_conf > conf_threshold
            # pts0 = pts0[high_conf_matches]
            # pts1 = pts1[high_conf_matches]

            logging.info("%s matches found.", n_matches)
            # for match in matches:
            #    crop_points.append(keypoints_list_crop[j][match.queryIdx].pt)
            #    layout_points.append(layout_keypoints[j][match.trainIdx].pt)
            crop_points = pts1
            layout_points = pts0

        all_matches.append(
            {
                "n_matches": len(crop_points),
                "layout": layout_points,
                "crop": crop_points,
                "meta": meta,
            }
        )

    item_max_matches = max(all_matches, key=lambda x: x["n_matches"])
    layout_meta = item_max_matches["meta"]
    transf_matrix, _ = cv2.estimateAffine2D(
        np.array(item_max_matches["crop"]),
        np.array(item_max_matches["layout"]),
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
