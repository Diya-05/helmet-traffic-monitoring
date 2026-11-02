import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import engine, get_db
from app.models import Violation
from app.services.pipeline_service import analyze_traffic_image

load_dotenv()

EVIDENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evidence"
)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

app = FastAPI(
    title=os.getenv("APP_NAME", "Helmet Detection API"),
    description="Backend for helmet detection and traffic compliance monitoring.",
    version="0.2.0",
)

# Enable CORS for local development and dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence directory for static image serving
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")

# Mount frontend dashboard directory
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
)
if os.path.exists(FRONTEND_DIR):
    app.mount(
        "/dashboard",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="dashboard",
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive monitoring dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------

class DetectionResult(BaseModel):
    number_plate: Optional[str] = Field(default=None, max_length=20)
    helmet_detected: bool
    detection_source: str = Field(default="mock", max_length=30)
    evidence_path: Optional[str] = None


class ViolationResponse(BaseModel):
    id: int
    number_plate: Optional[str]
    helmet_detected: bool
    detection_source: str
    review_status: str
    evidence_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StatusUpdate(BaseModel):
    review_status: str = Field(
        ...,
        pattern="^(pending|approved|rejected|dismissed)$",
        description="Updated review status: pending, approved, rejected, or dismissed",
    )


class ViolationsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ViolationResponse]


class StatsResponse(BaseModel):
    total_detections: int
    helmet_compliant: int
    violations: int
    pending_review: int
    approved_violations: int
    rejected_violations: int


class AnalyzeResultResponse(BaseModel):
    violation: ViolationResponse
    helmet_classification: list[dict]
    number_plate_detection: list[dict]
    ocr_output: dict
    evidence_path: Optional[str]
    is_mock_inference: bool
    model_status: str
    summary: str
    processing_time_ms: Optional[float] = None


# ---------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "helmet-detection-api",
    }


@app.get("/api/health/database", tags=["Health"])
def database_health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected",
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "unavailable",
            },
        )


# ---------------------------------------------------------
# Detection & Violation Endpoints
# ---------------------------------------------------------

@app.post("/api/detections", response_model=ViolationResponse, tags=["Detections"])
def create_detection(
    detection: DetectionResult,
    db: Session = Depends(get_db),
):
    """Insert a new detection or violation record."""
    violation = Violation(
        number_plate=detection.number_plate,
        helmet_detected=detection.helmet_detected,
        detection_source=detection.detection_source,
        review_status="pending",
        evidence_path=detection.evidence_path,
    )

    db.add(violation)
    db.commit()
    db.refresh(violation)

    return violation


@app.post(
    "/api/detections/analyze",
    response_model=AnalyzeResultResponse,
    tags=["Detections"],
)
async def analyze_traffic_frame(
    file: UploadFile = File(..., description="Traffic image or CCTV frame"),
    detection_source: str = Form(
        default="camera_upload", description="Source camera or stream tag"
    ),
    conf_threshold: float = Form(
        default=0.2, ge=0.05, le=1.0, description="Detection confidence threshold"
    ),
    use_mock: bool = Form(
        default=False,
        description="Explicit simulated mock inference mode (for testing upload, validation, and storage without claiming real AI output)",
    ),
    db: Session = Depends(get_db),
):
    """Execute model inference workflow: detects helmet compliance, recognizes license

    plate with EasyOCR, annotates evidence image, and logs record to PostgreSQL.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image (JPEG, PNG, etc.).",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image file received.",
            )

        result = analyze_traffic_image(
            image_bytes=image_bytes,
            original_filename=file.filename or "frame.jpg",
            db=db,
            detection_source=detection_source,
            conf_threshold=conf_threshold,
            force_mock=use_mock,
        )
        return result

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing computer vision pipeline: {err}",
        )


@app.get("/api/violations", response_model=ViolationsListResponse, tags=["Violations"])
def list_violations(
    review_status: Optional[str] = Query(
        None,
        description="Filter by review status: pending, approved, rejected",
    ),
    helmet_detected: Optional[bool] = Query(
        None,
        description="Filter by helmet compliance status",
    ),
    search: Optional[str] = Query(
        None,
        description="Search substring in number plate",
    ),
    page: int = Query(default=1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(
        default=20, ge=1, le=100, description="Number of items per page"
    ),
    db: Session = Depends(get_db),
):
    """List recorded violations with pagination, status filtering, and plate search."""
    query = select(Violation)

    if review_status:
        query = query.where(Violation.review_status == review_status)

    if helmet_detected is not None:
        query = query.where(Violation.helmet_detected == helmet_detected)

    if search:
        query = query.where(Violation.number_plate.ilike(f"%{search.strip()}%"))

    # Compute total matching count
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    # Paginate and order by newest first
    offset = (page - 1) * page_size
    items = (
        db.scalars(
            query.order_by(Violation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        .all()
    )

    return ViolationsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=list(items),
    )


@app.get("/api/violations/{violation_id}", response_model=ViolationResponse, tags=["Violations"])
def get_violation(
    violation_id: int,
    db: Session = Depends(get_db),
):
    """Retrieve detailed information for a specific violation by ID."""
    violation = db.get(Violation, violation_id)
    if not violation:
        raise HTTPException(
            status_code=404,
            detail=f"Violation with ID {violation_id} not found.",
        )
    return violation


@app.patch("/api/violations/{violation_id}/status", response_model=ViolationResponse, tags=["Violations"])
def update_violation_status(
    violation_id: int,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
):
    """Update review status (pending, approved, rejected, dismissed) for a violation."""
    violation = db.get(Violation, violation_id)
    if not violation:
        raise HTTPException(
            status_code=404,
            detail=f"Violation with ID {violation_id} not found.",
        )

    violation.review_status = payload.review_status
    db.commit()
    db.refresh(violation)

    return violation


@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
def get_statistics(db: Session = Depends(get_db)):
    """Aggregate detection and violation statistics for the monitoring dashboard."""
    total_detections = db.scalar(select(func.count(Violation.id))) or 0
    helmet_compliant = (
        db.scalar(
            select(func.count(Violation.id)).where(Violation.helmet_detected.is_(True))
        )
        or 0
    )
    violations = (
        db.scalar(
            select(func.count(Violation.id)).where(Violation.helmet_detected.is_(False))
        )
        or 0
    )
    pending_review = (
        db.scalar(
            select(func.count(Violation.id)).where(Violation.review_status == "pending")
        )
        or 0
    )
    approved_violations = (
        db.scalar(
            select(func.count(Violation.id)).where(Violation.review_status == "approved")
        )
        or 0
    )
    rejected_violations = (
        db.scalar(
            select(func.count(Violation.id)).where(Violation.review_status == "rejected")
        )
        or 0
    )

    return StatsResponse(
        total_detections=total_detections,
        helmet_compliant=helmet_compliant,
        violations=violations,
        pending_review=pending_review,
        approved_violations=approved_violations,
        rejected_violations=rejected_violations,
    )
