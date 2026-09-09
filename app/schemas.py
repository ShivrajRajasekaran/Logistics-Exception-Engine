"""Pydantic contracts for both endpoints."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                              description="[x1, y1, x2, y2] in absolute pixels")


class DetectResponse(BaseModel):
    status: Literal["success"] = "success"
    filename: Optional[str] = None
    image_size: List[int] = Field(..., description="[width, height]")
    inference_time_ms: float
    count: int
    detections: List[Detection]


class ReasonRequest(BaseModel):
    package_id: str = Field(..., examples=["PKG-8821"])
    query: str = Field(..., examples=["Is there seal damage requiring a carrier claim?"])
    image_path: Optional[str] = Field(
        None, description="Server-side path to the parcel image. Omit for non-visual queries."
    )


class ReasonResponse(BaseModel):
    package_id: str
    status: Literal[
        "EXCEPTION_FLAGGED",
        "CLEAR",
        "INSUFFICIENT_INFORMATION",
        "ANSWERED_WITHOUT_VISION",
        "UNSUPPORTED_CAPABILITY",
    ]
    requires_vision_model: bool
    guardrail_passed: bool
    max_critical_confidence: Optional[float] = None
    decision_summary: str
    detections: List[Detection] = []
    ledger_record: Optional[dict] = None
