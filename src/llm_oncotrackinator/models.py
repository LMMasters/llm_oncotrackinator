"""
Data models for lesion tracking.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class Lesion(BaseModel):
    """A single lesion observation at a specific timepoint."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    lesion_id: str = Field(..., description="Unique identifier for this lesion across timepoints")
    location: str = Field(..., description="Anatomical location of the lesion")
    size_mm: Optional[float] = Field(None, description="Size in millimeters")
    size_cm: Optional[float] = Field(None, description="Size in centimeters")
    characteristics: Optional[str] = Field(None, description="Additional characteristics or description")
    timepoint_date: datetime = Field(..., description="Date of this observation")
    raw_text: Optional[str] = Field(None, description="Raw text from the report describing this lesion")


class TimePoint(BaseModel):
    """A single timepoint observation with all lesions."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    date: datetime = Field(..., description="Date of this timepoint")
    report_text: str = Field(..., description="Full report text")
    lesions: List[Lesion] = Field(default_factory=list, description="Lesions identified at this timepoint")


class PatientLesionHistory(BaseModel):
    """Complete lesion tracking history for a patient."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    patient_id: str = Field(..., description="Patient identifier")
    timepoints: List[TimePoint] = Field(default_factory=list, description="Chronologically ordered timepoints")
    summary: Optional[str] = Field(None, description="Optional summary of lesion progression")

    def get_lesion_timeline(self, lesion_id: str) -> List[Lesion]:
        """
        Get the timeline for a specific lesion across all timepoints.

        Args:
            lesion_id: The lesion identifier to track

        Returns:
            List of lesion observations sorted by date
        """
        timeline = []
        for tp in self.timepoints:
            for lesion in tp.lesions:
                if lesion.lesion_id == lesion_id:
                    timeline.append(lesion)
        return sorted(timeline, key=lambda l: l.timepoint_date)

    def get_all_lesion_ids(self) -> List[str]:
        """
        Get all unique lesion IDs tracked for this patient.

        Returns:
            List of unique lesion identifiers
        """
        lesion_ids = set()
        for tp in self.timepoints:
            for lesion in tp.lesions:
                lesion_ids.add(lesion.lesion_id)
        return sorted(list(lesion_ids))


class ExtractionResult(BaseModel):
    """Result from LLM extraction."""

    model_config = ConfigDict()

    lesions: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted lesions")
    raw_response: str = Field(..., description="Raw LLM response")
    success: bool = Field(default=True, description="Whether extraction was successful")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")
