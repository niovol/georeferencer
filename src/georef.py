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
from .utils import downscale, load_geotiff, slice_geotiff
from .edges_infer import infer_edges, load_edges_model


def save_keypoints(keypoints: dict, keypoints_path: str) -> None:
    """
    Saves the keypoints, descriptors, and scores to a file.

    Args:
        keypoints (dict): Keypoints, descriptors, and scores to save.
        keypoints_path (str): Path to the file where the keypoints will be saved.
    """
    with open(keypoints_path, "wb") as f:
        pickle.dump(keypoints, f)


def load_keypoints(filename: str) -> dict:
    """
    Loads the keypoints, descriptors, and scores from a file.

    Args:
        filename (str): Path to the file containing the keypoints.

    Returns:
        dict: Loaded keypoints, descriptors, and scores.
    """
    with open(filename, "rb") as f:
        keypoints = pickle.load(f)
        return keypoints


def prepare_layout(layout_path: str) -> list:
    """
    Loads and prepares the layout information.

    Args:
        layout_path (str): Path to the layout file.

    Returns:
        list: Paths to the cropped layout images.
    """
    os.makedirs("cache/layout_downscale", exist_ok=True)

    grid_type = "mod"
    grid = {
        "base": {"nx": 8, "ny": 5, "tile_size_x": 340, "tile_size_y": 400},
        "mod": {"nx": 4, "ny": 5, "tile_size_x": 480, "tile_size_y": 384},
    }

    filename_downscale = f"cache/layout_downscale/{os.path.basename(layout_path)}"
    if not os.path.exists(filename_downscale):
        downscale(layout_path, filename_downscale, 1600, 1600, (0, 1, 2, 3))
        slice_geotiff(
            filename_downscale,
            "cache/layout_downscale_crops",
            (grid[grid_type]["tile_size_x"], grid[grid_type]["tile_size_y"]),
            (grid[grid_type]["nx"], grid[grid_type]["ny"]),
        )

    layout_crop_paths = []
    for i in range(grid[grid_type]["ny"]):
        for j in range(grid[grid_type]["nx"]):
            filename = os.path.basename(layout_path).replace(".tif", f"_{i}_{j}.tif")
            layout_crop_path = f"cache/layout_downscale_crops/{filename}"
            layout_crop_paths.append(layout_crop_path)

    return layout_crop_paths


def convert_affine_to_numpy(affine: Affine) -> np.ndarray:
    """
    Converts an Affine object to a NumPy array.

    Args:
        affine (Affine): An Affine transformation object.

    Returns:
        numpy.ndarray: A 2x3 NumPy array representing the affine transformation.
    """
    return np.array([[affine.a, affine.b, affine.c], [affine.d, affine.e, affine.f]])


def convert_numpy_to_affine(array: np.ndarray) -> Affine:
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


def multiply_affine_arrays(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
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


def align(layout_crop_paths: list, crop_image: np.ndarray, crop_path: str) -> dict:
    """
    Aligns a cropped image to a layout using keypoints and descriptors.

    Args:
        layout_crop_paths (list): List of paths to the cropped layout images.
        crop_image (numpy.ndarray): The cropped image to be aligned.
        crop_path (str): Path to the cropped image file.

    Returns:
        dict: A dictionary containing the new corners and updated metadata.
    """
    # pylint: disable=not-callable

    torch.set_grad_enabled(False)

    keypoints_dir = "cache/keypoints"
    os.makedirs(keypoints_dir, exist_ok=True)

    superpoint_config = {
        "nms_radius": 4,
        "keypoint_threshold": 0.015,
        "max_keypoints": -1,
    }
    superglue_config = {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.7,
        "descriptor_dim": 256,
    }
    superpoint = SuperPoint(superpoint_config).eval().to("cpu")
    superglue = SuperGlue(superglue_config).eval().to("cpu")

    edges_model = load_edges_model("resnet18", "models/edges.pth")

    keypoint1_path = os.path.join(
        keypoints_dir, os.path.basename(crop_path).replace(".tif", ".pkl")
    )
    if os.path.exists(keypoint1_path):
        pred1 = load_keypoints(keypoint1_path)
    else:
        # im1 = final_uint8(crop_image, "crop").astype("float32")
        im1 = infer_edges(edges_model, crop_image).astype("float32")
        inp1 = frame2tensor(np.squeeze(im1), "cpu")
        pred1 = {k + "1": v for k, v in superpoint({"image": inp1}).items()}
        pred1["image1"] = inp1
        save_keypoints(pred1, keypoint1_path)

    all_matches = []
    for layout_crop_path in layout_crop_paths:
        loaded_layout = load_geotiff(layout_crop_path, layout="hwc")

        keypoint0_path = os.path.join(
            keypoints_dir, os.path.basename(layout_crop_path).replace(".tif", ".pkl")
        )
        if os.path.exists(keypoint0_path):
            pred0 = load_keypoints(keypoint0_path)
        else:
            # im0 = final_uint8(loaded_layout["data"], "layout")).astype("float32")
            im0 = infer_edges(edges_model, loaded_layout["data"]).astype("float32")
            inp0 = frame2tensor(np.squeeze(im0), "cpu")
            pred0 = {k + "0": v for k, v in superpoint({"image": inp0}).items()}
            pred0["image0"] = inp0
            save_keypoints(pred0, keypoint0_path)

        pred = {**pred0, **pred1}

        for k in pred:
            if isinstance(pred[k], (list, tuple)):
                pred[k] = torch.stack(pred[k])

        pred = {**pred, **superglue(pred)}
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

        logging.info("%s matches found, score=%s", n_matches, score)

        all_matches.append(
            {
                "n_matches": len(crop_points),
                "score": score,
                "layout": layout_points,
                "crop": crop_points,
                "meta": loaded_layout["meta"],
            }
        )

    item_chosen = max(all_matches, key=lambda x: x["score"])
    layout_meta = item_chosen["meta"]
    transf_matrix, _ = cv2.estimateAffine2D(
        np.array(item_chosen["crop"]),
        np.array(item_chosen["layout"]),
        method=cv2.USAC_MAGSAC,
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
    new_meta["count"] = 4

    return {
        "corners": new_corners,
        "meta": new_meta,
        "layout_points": layout_points,
        "crop_points": crop_points,
    }
