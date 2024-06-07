"""
api.py
"""

from fastapi import FastAPI
from .schemas import TaskResponse, ResultResponse

app = FastAPI()
