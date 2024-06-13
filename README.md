# Georeferencer

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)
6. [Logs and Results](#logs-and-results)

## Introduction

This project was developed as part of the hackathon "Leaders of Digital Transformation". The service is designed for determining the geographical location of satellite image scenes and correcting dead pixels. The service is fully automated and requires no intermediate configuration during operation.

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

2. Configure the `docker-compose.yml` file, specifying the path to the layouts directory on your local machine. Replace `/layouts` on the right side of the colon with the path to your layouts directory. Example:

    ```yaml
    services:
      fastapi-app:
        container_name: nikolove18_fastapi
        image: nikolove18
        build:
          context: ./
          dockerfile: Dockerfile
        command: uvicorn src.api.server:app --reload --host 0.0.0.0
        volumes:
          - .:/app
          - /layouts:/path/to/local/layouts
        shm_size: '2gb'
        ports:
          - 8000:8000
    ```

3. Build and start the service using Docker Compose:

    ```bash
    docker-compose up --build
    ```

The service will be available at `http://localhost:8000`.

## Usage

You can use the service either via API or with a script.

### Using the API

To process images via API, you can use tools like `curl` or Postman.

#### Example

```python
import requests

file_path = 'path_to_your_image.tif'
layout_name = 'your_layout_image.tif'
url = "http://localhost:8000/process"
with open(file_path, 'rb') as f:
    files = {'file': (file_path, f)}
    data = {'layout_name': layout_name}
    response = requests.post(url, data=data, files=files)
    print(response.json())
```

Response:

```python
{'task_id': '2823d72a-0760-4219-a75b-e50e176a1287'}
```

To get the processing results, use the received `task_id` as a parameter in your requests.

### Using the Script

To process images using the script, you can either run it inside the Docker container or set up a local Python environment.

#### Using the Script in Docker Container

Run the following command:

```bash
docker run --rm -v .:/app -v /layouts:<layouts_dir> nikolove18 python -m src.main \ 
    --layout_name <layout_name> --crop_name <path_to_crop_image_inside_project_dir>
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
  - JSON object with the coordinates and other processing details:

    ```json
    {
      "layout_name": "layout filename",
      "crop_name": "crop filename",
      "ul": "upper left coordinates",
      "ur": "upper right coordinates",
      "br": "bottom right coordinates",
      "bl": "bottom left coordinates",
      "crs": "coordinate reference system",
      "start": "processing start time",
      "end": "processing end time"
    }
    ```

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

Downloads the bug report for dead pixel correction as a CSV file.

- **Parameters:**
  - `task_id` (string): The ID of the processing task.
- **Response:**
  - CSV file.

## Logs and Results

Logs, processing results, and reports are stored in the `tasks/{task_id}` directory. This directory contains:

- `process.log` - log file of the processing task.
- `coords.csv` - file with the coordinates of the scene corners and additional processing parameters.
- `coords.txt` - file with the coordinates of the scene corners.
- `coords.geojson` - GeoJSON with the coordinates of the scene corners in the layout's coordinate system.
- `bug_report.csv` - report on restored dead pixels.
- `corrected.tif` - GeoTIFF file with restored dead pixels.
- `aligned.tif` - GeoTIFF scene file with geographical referencing to the layout.
