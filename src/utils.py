"""
utils.py
"""

import os
from typing import Any, Dict

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows


def scale_image_percentile(image, low_percentile=2, high_percentile=98):
    """
    Scale the image based on the 2nd and 98th percentiles, excluding zero values.

    Args:
        image (numpy.ndarray): Input image.
        low_percentile (int): Lower percentile for scaling.
        high_percentile (int): Upper percentile for scaling.

    Returns:
        numpy.ndarray: Scaled image.
    """
    non_zero_values = image[image > 0]
    low, high = np.percentile(non_zero_values, [low_percentile, high_percentile])
    scaled_image = np.clip((image - low) / (high - low) * 254 + 1, 1, 255).astype(
        "uint8"
    )
    return scaled_image


def _validate_input_file(input_file: str) -> None:
    """
    Validate the input file path and extension.

    Args:
        input_file (str): The path to the file to be validated.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file extension is not .tif or .tiff.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"The specified file does not exist: " f"{input_file}")
    if not input_file.lower().endswith((".tif", ".tiff")):
        raise ValueError(
            f"Unsupported file format: {input_file}. "
            f"Expected a .tif or .tiff file, "
            f"but got {os.path.splitext(input_file)[-1]}."
        )


def _validate_arguments(**kwargs) -> None:
    """
    Validate keyword arguments passed to the load and save functions.

    Args:
        **kwargs: Keyword arguments

    Raises:
        ValueError: If an unsupported layout is specified.
    """
    layout = kwargs.get("layout", "chw")
    if layout not in ["chw", "hwc"]:
        raise ValueError(
            f"Invalid layout specified: {layout}. " f"Choose 'chw' or 'hwc'"
        )


def _load_with_rasterio(input_file: str, **kwargs) -> Dict[str, Any]:
    """
    Load a geospatial image file using the Rasterio library.

    Args:
        input_file (str): The path to the image file to be loaded.
        **kwargs: Additional keyword arguments:
            - layout (str, optional): Format of the output array
                                      'hwc' for height-width-channel
                                      'chw' for channel-height-width.
            - dtype (str, optional): Data type of the output (e.g., 'float32').
            - window (tuple, optional): A subset of the image specified as
                                        (x_offset, y_offset, width, height).
            - only_meta (bool, optional): If True, returns only the metadata
                                          without loading the image data.

    Returns:
        dict: A dictionary containing:
              - 'data' (np.ndarray): Image data array.
              - 'meta' (dict): Metadata like geotransform, projection, etc.
    """
    layout = kwargs.get("layout", "chw")
    dtype = kwargs.get("dtype", None)
    window = kwargs.get("window", None)
    only_meta = kwargs.get("only_meta", False)

    with rasterio.open(input_file) as img:
        meta = img.meta.copy()
        if only_meta:
            return {"meta": meta}

        if window is None:
            image = img.read()
        else:
            rio_window = rasterio.windows.Window(*window)
            image = img.read(window=rio_window)
            meta.update(
                {
                    "width": window[2],
                    "height": window[3],
                    "transform": img.window_transform(rio_window),
                }
            )

        if dtype is not None:
            image = image.astype(dtype)
        if layout == "hwc":
            image = np.transpose(image, (1, 2, 0))

    return {"data": image, "meta": meta}


def load_geotiff(input_file: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Load a geospatial image file.

    This function supports loading RGB or multi-band images, offering options
    for data layout, data type, and a specified window of the image. It
    utilizes Rasterio library based on provided arguments.
    The function can also return only the metadata of the image if  'only_meta'
    is set to True.

    Args:
        input_file (str): The path to the image file to be loaded.
        **kwargs: Additional keyword arguments:
            - layout (str, optional): Format of the output array
                                      'hwc' for height-width-channel
                                      'chw' for channel-height-width.
            - dtype (str, optional): Data type of the output (e.g., 'float32').
            - window (tuple, optional): A subset of the image specified as
              (x_offset, y_offset, width, height).
            - only_meta (bool, optional): If True, returns only the metadata
                                          without loading the image data.

    Returns:
        dict: A dictionary containing:
              - 'data' (np.ndarray): Image data array.
              - 'meta' (dict): Metadata like geotransform, projection, etc.

    Raises:
        ValueError: If an invalid layout is specified.
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file extension is not .tif or .tiff.
    """
    _validate_arguments(**kwargs)
    _validate_input_file(input_file)

    loaded = _load_with_rasterio(input_file, **kwargs)
    return loaded


def _save_with_rasterio(
    output_file: str, image: np.ndarray, meta: Dict[str, Any], **kwargs
) -> None:
    """
    Save a NumPy array as a geospatial image file using Rasterio.

    Args:
        output_file (str): Path where the image will be saved.
        image (np.ndarray): The image data to be saved.
        meta (dict): Metadata including geotransform and projection.
        **kwargs: Additional keyword arguments:
            - layout (str, optional): Format of the input array
                                      'hwc' for height-width-channel
                                      'chw' for channel-height-width.
            - nodata (int, optional): No-data value.

    Returns:
        None: The function saves the image and does not return a value.
    """
    layout = kwargs.get("layout", "chw")
    nodata = kwargs.get("nodata", None)

    if layout == "hwc":
        image = np.transpose(image, (2, 0, 1))

    if nodata is not None:
        meta.update({"nodata": nodata})

    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(image)


def save_geotiff(
    output_file: str, image: np.ndarray, meta: Dict[str, Any], **kwargs
) -> None:
    """
    Save a NumPy array as a geospatial image file.

    This function writes a NumPy array to a file in a geospatial image format,
    while retaining important metadata such as geotransform and projection.
    It supports various configurations for the array's format and data type.

    Args:
        output_file (str): Path where the image will be saved.
        image (np.ndarray): The image data to be saved.
        meta (dict): Metadata including geotransform and projection.
        **kwargs: Additional keyword arguments:
            - layout (str, optional): Format of the output array
                                      'hwc' for height-width-channel
                                      'chw' for channel-height-width.
            - dtype (str, optional): Data type for saving the array.

    Returns:
        None: The function saves the image and does not return a value.

    Raises:
        ValueError: If an invalid layout is specified.
    """
    _validate_arguments(**kwargs)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        _save_with_rasterio(output_file, image, meta, **kwargs)
    except Exception as e:
        raise IOError(
            f"Failed to save file {output_file} using Rasterio. " f"Error: {str(e)}"
        ) from e
