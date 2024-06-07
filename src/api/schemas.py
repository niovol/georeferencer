"""
schemas.py
"""

from pydantic import BaseModel  # pylint: disable=E0611


class ProcessResponse(BaseModel):
    """
    Task response schema
    """

    task_id: str


class ResultResponse(BaseModel):
    """
    Result response schema
    """

    layout_name: str
    crop_name: str
    ul: str
    ur: str
    br: str
    bl: str
    crs: str
    start: str
    end: str
