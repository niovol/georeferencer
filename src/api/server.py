"""
api.py
"""

import csv
import os
import uuid
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.main import main

from .schemas import BugReportResponse, ProcessResponse, ResultResponse

app = FastAPI()


@app.post("/process", response_model=ProcessResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    layout_name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Starts image processing task
    """

    task_id = str(uuid.uuid4())
    crop_name = file.filename
    crop_path = f"uploads/{crop_name}"
    layout_path = f"/layouts/{layout_name}"

    os.makedirs("uploads", exist_ok=True)

    with open(crop_path, "wb") as crop_file:
        crop_file.write(await file.read())

    background_tasks.add_task(main, layout_path, crop_path, task_id, False)

    return {"task_id": task_id}


@app.get("/coords", response_model=ResultResponse)
async def get_coords(task_id: str):
    """
    Gets result of image processing with georeferenced coords
    """

    task_dir = f"tasks/{task_id}"
    coords_file_path = os.path.join(task_dir, "coords.csv")

    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail="Task not found")

    if not os.path.exists(coords_file_path):
        raise HTTPException(status_code=404, detail="Task results not found")

    with open(coords_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        result = next(reader)

    return result


@app.get("/bug_report", response_model=List[BugReportResponse])
async def get_bug_report(task_id: str):
    """
    Gets bug report
    """
    task_dir = f"tasks/{task_id}"
    bug_report_file_path = os.path.join(task_dir, "bug_report.csv")

    if not os.path.exists(bug_report_file_path):
        raise HTTPException(status_code=404, detail="Bug report not found")

    bug_report_data = []
    with open(bug_report_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            bug_report_data.append(
                BugReportResponse(
                    row_number=int(row["row_number"]),
                    column_number=int(row["column_number"]),
                    channel_number=int(row["channel_number"]),
                    dead_value=int(row["dead_value"]),
                    corrected_value=int(row["corrected_value"]),
                )
            )

    return bug_report_data


@app.get("/download/geojson")
async def download_geojson(task_id: str):
    """
    Downloads result as GeoJSON file
    """
    task_dir = f"tasks/{task_id}"
    file_path = os.path.join(task_dir, "coords.geojson")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="GeoJSON file not found")

    return FileResponse(
        file_path, media_type="application/json", filename="coords.geojson"
    )


@app.get("/download/geotiff")
async def download_geotiff(task_id: str):
    """
    Downloads result as GeoTIFF file with georeferencing.
    """
    task_dir = f"tasks/{task_id}"
    file_path = os.path.join(task_dir, "aligned.tif")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="GeoTIFF file not found")

    return FileResponse(file_path, media_type="image/tiff", filename="aligned.tif")


@app.get("/download/corrected_pixels")
async def download_corrected(task_id: str):
    """
    Downloads Geotiff with corrected pixels in original reference system.
    """
    task_dir = f"tasks/{task_id}"
    file_path = os.path.join(task_dir, "corrected.tif")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Corrected file not found")

    return FileResponse(file_path, media_type="image/tiff", filename="corrected.tif")


@app.get("/download/bug_report")
async def download_bug_report(task_id: str):
    """
    Downloads CSV file with bug report
    """
    task_dir = f"tasks/{task_id}"
    bug_report_file_path = os.path.join(task_dir, "bug_report.csv")

    if not os.path.exists(bug_report_file_path):
        raise HTTPException(status_code=404, detail="Bug report not found")

    return FileResponse(
        bug_report_file_path, media_type="text/csv", filename="bug_report.csv"
    )
