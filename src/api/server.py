"""
api.py
"""

import csv
import os
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from src.main import main

from .schemas import ProcessResponse, ResultResponse

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

    background_tasks.add_task(main, layout_path, crop_path, task_id)

    return {"task_id": task_id}


@app.get("/result", response_model=ResultResponse)
async def get_result(task_id: str):
    """
    Gets result of image processing
    """

    task_dir = f"tasks/{task_id}"
    coords_file_path = os.path.join(task_dir, "coords.csv")

    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail="Task not found")

    if not os.path.exists(coords_file_path):
        raise HTTPException(status_code=400, detail="Task results not found")

    with open(coords_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        result = next(reader)

    return result
