"""
api.py
"""

import os
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from src.main import load_layout, process

from .schemas import ProcessResponse, ResultResponse

app = FastAPI()
tasks = {}


def process_task(task_id: str, layout_path: str, crop_path: str):
    """
    Background task to process the image
    """

    layout = load_layout(layout_path)
    result = process(layout, crop_path)
    tasks[task_id].update({"result": result, "status": "processed"})


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
    layout_path = f"/layouts/{layout_name}"
    crop_path = f"uploads/{crop_name}"

    os.makedirs("uploads", exist_ok=True)
    with open(crop_path, "wb") as crop_file:
        crop_file.write(await file.read())

    background_tasks.add_task(process_task, task_id, layout_path, crop_name)

    tasks[task_id] = {
        "layout_name": layout_name,
        "crop_name": crop_name,
        "status": "processing",
    }

    return {"task_id": task_id}


@app.get("/result", response_model=ResultResponse)
async def get_result(task_id: str):
    """
    Gets result of image processing
    """
    task = tasks.get(task_id, None)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != "processed":
        raise HTTPException(status_code=400, detail="Task is still processing")

    layout_name = task["layout_name"]
    crop_name = task["crop_name"]
    result = task["result"]

    coords = result["corners"]
    start_time = result["start"].strftime("%Y-%m-%dT%H:%M:%S")
    end_time = result["end"].strftime("%Y-%m-%dT%H:%M:%S")

    response_data = {
        "layout_name": layout_name,
        "crop_name": crop_name,
        "ul": f"{coords[0][0]}; {coords[0][1]}",
        "ur": f"{coords[1][0]}; {coords[1][1]}",
        "br": f"{coords[2][0]}; {coords[2][1]}",
        "bl": f"{coords[3][0]}; {coords[3][1]}",
        "crs": "EPSG:32637",
        "start": start_time,
        "end": end_time,
    }

    return response_data
