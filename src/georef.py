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

from .utils import (
    downscale,
    equalize_hist,
    load_geotiff,
    megagray,
    save_geotiff,
    scale_image_percentile,
    slice_geotiff,
)


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

    filename_sift = f"models/sift/{os.path.basename(layout_path)}.pkl"
    os.makedirs("models/sift", exist_ok=True)
    os.makedirs("models/downscale", exist_ok=True)
    os.makedirs("models/hist_equalize", exist_ok=True)
    os.makedirs("models/sobel", exist_ok=True)

    if os.path.exists(filename_sift):
        keypoints, descriptors, meta = load_model(filename_sift)
    else:
        filename_downscale = f"models/downscale/{os.path.basename(layout_path)}"
        if not os.path.exists(filename_downscale):
            downscale(layout_path, filename_downscale, 1600, 1600)
            slice_geotiff(
                filename_downscale, "models/downscale", (8, 5), (160, 100), (0, 0, 0, 0)
            )

        loaded = load_geotiff(filename_downscale, layout="hwc")
        image, meta = loaded["data"], loaded["meta"]

        filename_hist_equalize = f"models/hist_equalize/{os.path.basename(layout_path)}"
        if os.path.exists(filename_hist_equalize):
            loaded_equalized = load_geotiff(filename_hist_equalize, layout="hwc")
            image_equalized = loaded_equalized["data"]
        else:
            image_equalized = equalize_hist(image)
            new_meta = meta.copy()
            new_meta["dtype"] = "uint8"
            save_geotiff(
                filename_hist_equalize, image_equalized, new_meta, layout="hwc"
            )

        filename_megagray = f"models/megagray/{os.path.basename(layout_path)}"
        if os.path.exists(filename_megagray):
            loaded_megagray = load_geotiff(filename_megagray, layout="hwc")
            image_megagray = loaded_megagray["data"]
        else:
            image_megagray = megagray(image).reshape(
                (image.shape[0], image.shape[1], 1)
            )
            image_megagray = scale_image_percentile(image_megagray)
            new_meta = meta.copy()
            new_meta["dtype"] = "uint8"
            new_meta["count"] = 1
            save_geotiff(filename_megagray, image_megagray, new_meta, layout="hwc")

        filename_sobel = f"models/sobel/{os.path.basename(layout_path)}"
        if os.path.exists(filename_sobel):
            loaded_sobel = load_geotiff(filename_sobel, layout="hwc")
            sobel_combined = loaded_sobel["data"]
        else:
            sobelx = cv2.Sobel(image_equalized[:, :, 3], cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image_equalized[:, :, 3], cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sobel_combined = cv2.normalize(
                sobel_combined, None, 0, 255, cv2.NORM_MINMAX
            )

            new_meta = meta.copy()
            new_meta["dtype"] = "uint8"
            new_meta["count"] = 1
            save_geotiff(
                filename_sobel,
                sobel_combined.reshape(
                    (sobel_combined.shape[0], sobel_combined.shape[1], 1)
                ),
                new_meta,
                layout="hwc",
            )

        filename_canny = f"models/canny/{os.path.basename(layout_path)}"
        if os.path.exists(filename_canny):
            loaded_canny = load_geotiff(filename_canny, layout="hwc")
            canny_edges = loaded_canny["data"]
        else:
            canny_edges = cv2.Canny(image_equalized[:, :, 3], 100, 200)

            new_meta = meta.copy()
            new_meta["dtype"] = "uint8"
            new_meta["count"] = 1
            save_geotiff(
                filename_canny,
                canny_edges.reshape((canny_edges.shape[0], canny_edges.shape[1], 1)),
                new_meta,
                layout="hwc",
            )

        keypoints, descriptors, keypoints_image = detect_keypoints_and_descriptors(
            image, "SIFT"
        )
        save_model(filename_sift, keypoints, descriptors, meta)

        new_meta = meta.copy()
        new_meta["count"] = 3
        save_geotiff(
            "tasks/keypoints_image.tif", keypoints_image, new_meta, layout="hwc"
        )

    return {
        "meta": meta,
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def match_features(descriptors1, descriptors2, method="SIFT"):
    """
    Matches features between two sets of descriptors using FLANN-based matcher
    for SIFT or brute-force for ORB.

    Args:
        descriptors1 (numpy.ndarray): Descriptors from the first image.
        descriptors2 (numpy.ndarray): Descriptors from the second image.
        method (str): The feature extraction method used ('SIFT' or 'ORB').

    Returns:
        matches: List of matched features.
    """
    if method == "SIFT":
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        return good_matches

    elif method == "ORB":
        # Brute-force matcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    raise ValueError("`method` should be either 'SIFT' or 'ORB'")


def detect_keypoints_and_descriptors(image, detector_type="SIFT", draw=False):
    """
    Detects keypoints and computes descriptors for the given image.

    Args:
        image (numpy.ndarray): Input image.
        detector_type (str): The type of feature detector to use ('SIFT' or 'ORB').
        draw (bool): Whether to draw and display the keypoints on the image.

    Returns:
        tuple: Lists of keypoints and descriptors for each channel.
    """
    if detector_type == "SIFT":
        feature_detector = cv2.SIFT_create()
    elif detector_type == "ORB":
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    image_equalized = equalize_hist(image)

    keypoints_list = []
    descriptors_list = []
    for i in [10]:  # range(1):
        logging.info("Detecting keypoints and computing descriptors on channel %s", i)

        if i < 4:
            gray = image_equalized[:, :, i]
            # gray = scale_image_percentile(image[:, :, i])
        elif i == 10:
            gray = scale_image_percentile(megagray(image))
        elif i == 20:
            pass

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


def compute_transformation(
    scene_image,
    keypoints_list_layout,
    descriptors_list_layout,
    transformation_type="homography",
    method="SIFT",
):
    """
    Computes the transformation matrix to align the scene.

    Args:
        scene_image (numpy.ndarray): The image to be transformed.
        keypoints_list_layout (list): List of keypoints from the layout image.
        descriptors_list_layout (list): List of descriptors from the layout image.
        transformation_type (str): The type of transformation
            ('homography', 'affine', or 'affine_partial').
        method (str): The feature extraction method used ('SIFT' or 'ORB').

    Returns:
        numpy.ndarray: The transformation matrix.
    """
    keypoints_list_crop, descriptors_list_crop, _ = detect_keypoints_and_descriptors(
        scene_image, method
    )

    crop_points = []
    layout_points = []

    for i in range(1):
        logging.info("%s channel", i)
        matches = match_features(
            descriptors_list_crop[i], descriptors_list_layout[i], method=method
        )
        logging.info("%s matches found.", len(matches))
        for match in matches:
            crop_points.append(keypoints_list_crop[i][match.queryIdx].pt)
            layout_points.append(keypoints_list_layout[i][match.trainIdx].pt)

    if transformation_type == "homography":
        transf_matrix, _ = cv2.findHomography(
            np.array(crop_points), np.array(layout_points), cv2.RANSAC, 5.0
        )
    elif transformation_type == "affine":
        transf_matrix, _ = cv2.estimateAffine2D(
            np.array(crop_points),
            np.array(layout_points),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    elif transformation_type == "affine_partial":
        transf_matrix, _ = cv2.estimateAffinePartial2D(
            np.array(crop_points),
            np.array(layout_points),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    else:
        raise ValueError(
            "transformation_type must be either "
            "'homography', 'affine' or 'affine_partial'"
        )

    return transf_matrix, layout_points, crop_points


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


def align(layout_keypoints, layout_descriptors, layout_meta, crop_image):
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
    transf_matrix, layout_points, crop_points = compute_transformation(
        crop_image,
        layout_keypoints,
        layout_descriptors,
        transformation_type="affine",
        method="SIFT",
    )

    h, w = crop_image.shape[:2]
    scene_corners_pixels = [[0, 0], [w, 0], [w, h], [0, h]]

    new_affine_array = multiply_affine_arrays(
        transf_matrix, convert_affine_to_numpy(layout_meta["transform"])
    )
    new_affine = convert_numpy_to_affine(new_affine_array)
    new_corners = [new_affine * corner_pixels for corner_pixels in scene_corners_pixels]
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
