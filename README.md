# Georeferencer

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)

## Introduction

This project is developed as part of the hackathon "Leaders of Digital Transformation" task "Efficient Processing Algorithm for Satellite Images of the Russian Orbital Group". The service processes satellite images to determine their geographical location accurately, corrects dead pixels, and provides the output in various formats. The service is fully automated, ensuring minimal latency and high efficiency.

## Features

- High-speed data processing algorithm
- Automatic geolocation determination
- Dead pixel detection and correction
- Supports output in multiple formats (CSV, GeoJSON, GeoTIFF)
- RESTful API for easy integration

## Installation

### Prerequisites

- Docker
- Docker Compose

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/niovol/georeferencer
    cd <repository_directory>
    ```

2. Build and start the service using Docker Compose:

    ```bash
    docker-compose up --build
    ```

The service will be available at `http://localhost:8000`.

## Usage

### Using the Script

To process images using the script, you can either run the script inside the Docker container or set up a local Python environment.

#### Using the Script in Docker Container

Run the following command:

```bash
docker run --rm -v .:/app -v /layouts:<layouts_dir> nikolove18 python -m src.main --layout_name <layout_name> --crop_name <path_to_crop_image_inside_project_dir>
```

#### Using the Script Locally

1. Set up a Python environment and install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the script:

    ```bash
    python main.py --crop_name <path_to_crop_image> --layout_name <path_to_layout_image>
    ```

### Using the API

To process images via API, you can use tools like `curl` or Postman.

#### Example

1. Upload an image for processing:

    ```bash
    curl -X POST "http://localhost:8000/process" -F "layout_name=<layout_filename>" -F "file=@<path_to_image>"
    ```

2. Check the results:

    ```bash
    curl -X GET "http://localhost:8000/coords?task_id=<task_id>"
    ```

## API Endpoints

### POST `/process`

Starts the image processing task.

- **Parameters:**
  - `layout_name` (string): The name of the layout file.
  - `file` (file): The image file to process.
- **Response:**
  - `task_id` (string): The ID of the processing task.

### GET `/coords`

Gets the georeferenced coordinates of the processed image.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - JSON object with the coordinates and other processing details.

### GET `/bug_report`

Gets the bug report for dead pixel correction.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - JSON object with the bug report details.

### GET `/download/geojson`

Downloads the result as a GeoJSON file.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - GeoJSON file.

### GET `/download/geotiff`

Downloads the result as a GeoTIFF file with georeferencing.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - GeoTIFF file.

### GET `/download/corrected_pixels`

Downloads the corrected pixels in the original reference system.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - GeoTIFF file.

### GET `/download/bug_report`

Downloads the bug report as a CSV file.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - CSV file.
