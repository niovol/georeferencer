"""
georef.py
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine

from .utils import scale_image_percentile


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
        index_params = dict(algorithm=flann_index_kdtree, trees=3)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
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

    keypoints_list = []
    descriptors_list = []
    for i in range(4):
        gray = scale_image_percentile(image[:, :, i])
        keypoints, descriptors = feature_detector.detectAndCompute(gray, None)

        if draw:
            keypoints_image = cv2.drawKeypoints(
                gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            keypoints_image_rgb = cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB)
            plt.imshow(keypoints_image_rgb)
            plt.show()

        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def compute_transformation(
    scene_image,
    keypoints_list_substrate,
    descriptors_list_substrate,
    transformation_type="homography",
    method="SIFT",
):
    """
    Computes the transformation matrix to align the scene.

    Args:
        scene_image (numpy.ndarray): The image to be transformed.
        keypoints_list_substrate (list): List of keypoints from the substrate image.
        descriptors_list_substrate (list): List of descriptors from the substrate image.
        transformation_type (str): The type of transformation
            ('homography', 'affine', or 'affine_partial').
        method (str): The feature extraction method used ('SIFT' or 'ORB').

    Returns:
        numpy.ndarray: The transformation matrix.
    """
    keypoints_list_scene, descriptors_list_scene = detect_keypoints_and_descriptors(
        scene_image, method
    )

    scene_points = []
    substrate_points = []

    for i in range(4):
        matches = match_features(
            descriptors_list_scene[i], descriptors_list_substrate[i], method=method
        )
        print(f"Number of matches in {i} channel: ", len(matches))
        for match in matches:
            scene_points.append(keypoints_list_scene[i][match.queryIdx].pt)
            substrate_points.append(keypoints_list_substrate[i][match.trainIdx].pt)

    if transformation_type == "homography":
        transf_matrix, _ = cv2.findHomography(
            np.array(scene_points), np.array(substrate_points), cv2.RANSAC, 5.0
        )
    elif transformation_type == "affine":
        transf_matrix, _ = cv2.estimateAffine2D(
            np.array(scene_points),
            np.array(substrate_points),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    elif transformation_type == "affine_partial":
        transf_matrix, _ = cv2.estimateAffinePartial2D(
            np.array(scene_points),
            np.array(substrate_points),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    else:
        raise ValueError(
            "transformation_type must be either "
            "'homography', 'affine' or 'affine_partial'"
        )

    return transf_matrix


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
    transf_matrix = compute_transformation(
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

    return {"corners": new_corners, "meta": new_meta}
