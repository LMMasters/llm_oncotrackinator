"""
Lesion tracking across multiple timepoints.
"""

from typing import List, Dict, Optional
from datetime import datetime

from llm_oncotrackinator.config import Config
from llm_oncotrackinator.data_loader import MedicalReport
from llm_oncotrackinator.lesion_extractor import LesionExtractor
from llm_oncotrackinator.models import (
    Lesion,
    TimePoint,
    PatientLesionHistory,
    ExtractionResult
)


class LesionTracker:
    """Track lesions across multiple timepoints for patients."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LesionTracker.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()
        self.extractor = LesionExtractor(config=self.config)

    def track_patient(
        self,
        patient_id: str,
        reports: List[MedicalReport]
    ) -> PatientLesionHistory:
        """
        Track lesions for a single patient across all timepoints.

        Args:
            patient_id: Patient identifier
            reports: List of medical reports, should be chronologically sorted

        Returns:
            PatientLesionHistory with all tracked lesions

        Raises:
            ValueError: If reports list is empty
        """
        if not reports:
            raise ValueError("Reports list cannot be empty")

        # Sort reports by date to ensure chronological order
        sorted_reports = sorted(reports, key=lambda r: r.date)

        history = PatientLesionHistory(patient_id=patient_id)

        # Process first timepoint
        first_report = sorted_reports[0]
        first_timepoint = self._process_first_timepoint(first_report)
        history.timepoints.append(first_timepoint)

        # Track lesions for ID assignment
        tracked_lesions = self._extract_lesion_summaries(first_timepoint.lesions)

        # Process subsequent timepoints
        for report in sorted_reports[1:]:
            timepoint = self._process_followup_timepoint(report, tracked_lesions)
            history.timepoints.append(timepoint)

            # Update tracked lesions
            tracked_lesions = self._extract_lesion_summaries(timepoint.lesions)

        return history

    def track_all_patients(
        self,
        patient_reports: Dict[str, List[MedicalReport]]
    ) -> List[PatientLesionHistory]:
        """
        Track lesions for multiple patients.

        Args:
            patient_reports: Dictionary mapping patient_id to list of reports

        Returns:
            List of PatientLesionHistory objects
        """
        results = []

        for patient_id, reports in patient_reports.items():
            try:
                history = self.track_patient(patient_id, reports)
                results.append(history)
            except Exception as e:
                print(f"Warning: Failed to track patient {patient_id}: {str(e)}")
                # Still add patient with empty history
                results.append(PatientLesionHistory(
                    patient_id=patient_id,
                    summary=f"Tracking failed: {str(e)}"
                ))

        return results

    def _process_first_timepoint(self, report: MedicalReport) -> TimePoint:
        """
        Process the first timepoint for a patient.

        Args:
            report: The first medical report

        Returns:
            TimePoint with extracted lesions
        """
        extraction_result = self.extractor.extract_first_timepoint(report.report_text)

        if not extraction_result.success:
            print(f"Warning: Extraction failed for first timepoint: {extraction_result.error_message}")
            return TimePoint(
                date=report.date,
                report_text=report.report_text,
                lesions=[]
            )

        # Convert extracted data to Lesion objects with auto-generated IDs
        lesions = []
        for idx, lesion_data in enumerate(extraction_result.lesions, start=1):
            lesion_id = f"L{idx}"
            lesion = self._create_lesion(lesion_id, lesion_data, report.date)
            lesions.append(lesion)

        return TimePoint(
            date=report.date,
            report_text=report.report_text,
            lesions=lesions
        )

    def _process_followup_timepoint(
        self,
        report: MedicalReport,
        previous_lesions: List[Dict]
    ) -> TimePoint:
        """
        Process a follow-up timepoint.

        Args:
            report: The follow-up medical report
            previous_lesions: Summary of previously tracked lesions

        Returns:
            TimePoint with tracked lesions
        """
        extraction_result = self.extractor.extract_followup_timepoint(
            report.report_text,
            previous_lesions
        )

        if not extraction_result.success:
            print(f"Warning: Extraction failed for follow-up: {extraction_result.error_message}")
            return TimePoint(
                date=report.date,
                report_text=report.report_text,
                lesions=[]
            )

        # Convert extracted data to Lesion objects
        lesions = []
        for lesion_data in extraction_result.lesions:
            # Use lesion_id from LLM or generate new one
            lesion_id = lesion_data.get("lesion_id", f"L{len(lesions) + 1}")
            lesion = self._create_lesion(lesion_id, lesion_data, report.date)
            lesions.append(lesion)

        return TimePoint(
            date=report.date,
            report_text=report.report_text,
            lesions=lesions
        )

    def _create_lesion(
        self,
        lesion_id: str,
        lesion_data: Dict,
        timepoint_date: datetime
    ) -> Lesion:
        """
        Create a Lesion object from extracted data.

        Args:
            lesion_id: Lesion identifier
            lesion_data: Dictionary with lesion data
            timepoint_date: Date of observation

        Returns:
            Lesion object
        """
        return Lesion(
            lesion_id=lesion_id,
            location=lesion_data.get("location", "Unknown"),
            size_mm=lesion_data.get("size_mm"),
            size_cm=lesion_data.get("size_cm"),
            characteristics=lesion_data.get("characteristics"),
            timepoint_date=timepoint_date,
            raw_text=lesion_data.get("raw_text")
        )

    def _extract_lesion_summaries(self, lesions: List[Lesion]) -> List[Dict]:
        """
        Create summaries of lesions for tracking in next timepoint.

        Args:
            lesions: List of Lesion objects

        Returns:
            List of simplified lesion dictionaries
        """
        summaries = []
        for lesion in lesions:
            summary = {
                "lesion_id": lesion.lesion_id,
                "location": lesion.location,
            }
            if lesion.size_cm:
                summary["size_cm"] = lesion.size_cm
            if lesion.size_mm:
                summary["size_mm"] = lesion.size_mm
            if lesion.characteristics:
                summary["characteristics"] = lesion.characteristics
            summaries.append(summary)
        return summaries
