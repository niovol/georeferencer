"""
utils.py
"""

import json
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window


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
    non_zero_values = image[image != 0]
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


def save_geojson(file_path, coords):
    """
    Saves the coordinates of the corners of an image as a GeoJSON file.

    Args:
        file_path (str): The path where the GeoJSON file will be saved.
        coords (list of tuple): A list of tuples representing the coordinates
            of the four corners of the image. The list should contain four tuples,
            each with two float values (longitude, latitude).
    """
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32637"}},
        "features": [],
    }
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [coords[0][0], coords[0][1]],
                    [coords[1][0], coords[1][1]],
                    [coords[2][0], coords[2][1]],
                    [coords[3][0], coords[3][1]],
                    [coords[0][0], coords[0][1]],
                ]
            ],
        },
    }
    geojson_data["features"].append(feature)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=4)


def downscale(input_path: str, output_path: str, height: int, width: int) -> None:
    """
    Downscale a raster image to the specified height and width.

    Args:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the downscaled raster file.
        height (int): Desired height of the downscaled raster.
        width (int): Desired width of the downscaled raster.

    Returns:
        None
    """
    with rasterio.open(input_path) as src:
        transform = src.transform * src.transform.scale(
            (src.width / width), (src.height / height)
        )

        new_meta = src.meta.copy()
        new_meta.update({"transform": transform, "width": width, "height": height})

        with rasterio.open(output_path, "w", **new_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )


def equalize_hist_16bit_exclude_zero(img: np.ndarray) -> np.ndarray:
    """
    Equalizes the histogram of a 16-bit image, excluding zero values.

    Args:
        img (numpy.ndarray): Input 16-bit image.

    Returns:
        numpy.ndarray: Histogram-equalized image.
    """
    img_hist_eq = np.zeros_like(img)
    for i in range(img.shape[2]):
        channel = img[:, :, i]
        mask = channel > 0  # Create a mask to exclude zero values
        hist, _ = np.histogram(channel[mask].flatten(), 65535, [1, 65536])
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 65535 / (cdf.max() - cdf.min())  # Normalize to 16-bit
        cdf = np.insert(cdf, 0, 0)  # Insert 0 for the zero value
        cdf = cdf.astype(np.uint16)
        img_hist_eq[:, :, i] = cdf[channel]
    return img_hist_eq


def equalize_hist(img: np.ndarray) -> np.ndarray:
    """
    Equalizes the histogram of an image and applies CLAHE.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: CLAHE applied histogram-equalized image.
    """
    img_hist_eq = equalize_hist_16bit_exclude_zero(img) / 65535
    img_hist_eq_8bit = (img_hist_eq * 254 + 1).astype("uint8")

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    img_clahe = np.zeros_like(img_hist_eq_8bit)
    for i in range(img.shape[2]):
        img_clahe[:, :, i] = clahe.apply(img_hist_eq_8bit[:, :, i])

    return img_clahe


def slice_geotiff(
    input_path: str,
    output_dir: str,
    grid_size: Tuple[int, int],
    overlap: Tuple[int, int] = (0, 0),
    margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> None:
    """
    Slices a GeoTIFF image into smaller tiles.

    Args:
        input_path (str): Path to the input GeoTIFF file.
        output_dir (str): Directory to save the sliced tiles.
        grid_size (tuple): Number of tiles in the form (columns, rows).
        overlap (tuple, optional): Overlap between tiles in the form
            (x_overlap, y_overlap).
        margins (tuple, optional): Margins to exclude from slicing in the form
            (left, right, top, bottom).

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    left_margin, right_margin, top_margin, bottom_margin = margins
    x_overlap, y_overlap = overlap

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height

        cropped_width = width - left_margin - right_margin
        cropped_height = height - top_margin - bottom_margin

        tile_width = (cropped_width + (grid_size[0] - 1) * x_overlap) // grid_size[0]
        tile_height = (cropped_height + (grid_size[1] - 1) * y_overlap) // grid_size[1]

        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                window = Window(
                    j * (tile_width - x_overlap) + left_margin,
                    i * (tile_height - y_overlap) + top_margin,
                    tile_width,
                    tile_height,
                )
                transform = src.window_transform(window)

                output_path = os.path.join(
                    output_dir,
                    os.path.basename(input_path).replace(".tif", f"_{i}_{j}.tif"),
                )
                meta = src.meta.copy()
                meta.update(
                    {"height": tile_height, "width": tile_width, "transform": transform}
                )

                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(src.read(window=window))


def sobel(gray: np.ndarray) -> np.ndarray:
    """
    Apply the Sobel operator to detect edges in the image.

    Args:
        gray (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Image with edges detected by the Sobel operator.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobelx**2 + sobely**2)


def final_uint8(image: np.ndarray, channel: int) -> np.ndarray:
    """
    Generate a final uint8 image by applying Sobel operator and scaling.

    Args:
        image (numpy.ndarray): Input image.
        channel (int): Channel.

    Returns:
        numpy.ndarray: Final uint8 image.
    """
    red = image[:, :, 0]
    nir = image[:, :, 3]
    mask = 1 - 0.57 * (np.clip(red / 4000, 1, 2) - 1)
    gray = nir * mask

    nodata = np.where(gray > 64000, True, np.where(gray == 0, True, False))
    gray[nodata] = np.mean(gray[~nodata])
    # nir_equalized = np.squeeze(
    #    equalize_hist(nir.reshape(nir.shape[0], nir.shape[1], 1))
    # )
    # good2 = scale_image_percentile(sobel(nir_equalized), 30, 99.5).reshape(
    #    image.shape[0], image.shape[1], 1
    # )
    # megasobel = sobel(gray) + sobel(red) + sobel(gray - red)
    # megacontours = scale_image_percentile(megasobel, 30, 99.5).reshape(
    #     image.shape[0], image.shape[1], 1
    # )

    contours = scale_image_percentile(
        sobel(gray if channel == 3 else gray - red), 35, 99.5
    ).reshape(image.shape[0], image.shape[1], 1)

    return contours
