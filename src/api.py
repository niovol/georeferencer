from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import asyncio
import uuid
from datetime import datetime

app = FastAPI()


class TaskResponse(BaseModel):
    task_id: str


class ResultResponse(BaseModel):
    layout_name: str
    crop_name: str
    ul: str
    ur: str
    br: str
    bl: str
    crs: str
    start: str
    end: str
