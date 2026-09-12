"""Pydantic contracts for both endpoints."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(..., description="Predicted class name",
                       examples=["damaged-package"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.87])
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                              description="[x1, y1, x2, y2] in absolute pixels",
                              examples=[[120.5, 45.3, 540.2, 410.8]])


class DetectResponse(BaseModel):
    status: Literal["success"] = "success"
    filename: Optional[str] = Field(None, examples=["damaged_parcel.jpg"])
    image_size: List[int] = Field(..., description="[width, height]",
                                  examples=[[640, 640]])
    inference_time_ms: float = Field(..., examples=[152.34])
    count: int = Field(..., examples=[2])
    detections: List[Detection]


class ReasonRequest(BaseModel):
    package_id: str = Field(
        default="PKG-8821",
        examples=["PKG-8821"],
        description="Package ID to look up in the transit ledger.",
    )
    query: str = Field(
        default="Is this package damaged?",
        examples=["Is there damage requiring a carrier claim?"],
        description="Operator's question about the parcel.",
    )
    image_path: Optional[str] = Field(
        default="/app/sample_images/damaged_parcel.jpg",
        description="Server-side path to the parcel image. Omit for non-visual queries.",
        examples=["/app/sample_images/damaged_parcel.jpg"],
    )


class TransitEvent(BaseModel):
    hub: str = Field(..., examples=["HUB-01 Chennai"])
    event: str = Field(..., examples=["DISPATCH_SCAN"])
    condition: str = Field(..., examples=["INTACT"])


class LedgerRecord(BaseModel):
    """Typed transit ledger record for clean Swagger docs."""
    package_id: str = Field(..., examples=["PKG-8821"])
    carrier: str = Field(..., examples=["Apex Logistics"])
    origin_hub: str = Field(..., examples=["HUB-01 Chennai Sorting Center"])
    origin_label_status: str = Field(..., examples=["INTACT"])
    origin_seal_status: str = Field(..., examples=["INTACT"])
    dispatched_at: str = Field(..., examples=["2026-09-04T06:12:00Z"])
    sku_manifest: str = Field(..., examples=["SKU-9901"])
    declared_value_inr: Optional[float] = Field(None, examples=[48500])
    transit_history: List[TransitEvent] = []

    model_config = {"extra": "allow"}


class ReasonResponse(BaseModel):
    package_id: str = Field(..., examples=["PKG-8821"])
    status: Literal[
        "EXCEPTION_FLAGGED",
        "CLEAR",
        "INSUFFICIENT_INFORMATION",
        "ANSWERED_WITHOUT_VISION",
        "UNSUPPORTED_CAPABILITY",
    ] = Field(..., examples=["EXCEPTION_FLAGGED"])
    requires_vision_model: bool = Field(..., examples=[True])
    guardrail_passed: bool = Field(..., examples=[True])
    max_critical_confidence: Optional[float] = Field(None, examples=[0.87])
    decision_summary: str = Field(
        ...,
        examples=[
            "Detected damaged-package at 0.87 confidence. Ledger records origin "
            "status INTACT under Apex Logistics, so this damage was not present "
            "at dispatch and is attributable to the carrier."
        ],
    )
    detections: List[Detection] = []
    ledger_record: Optional[LedgerRecord] = None
