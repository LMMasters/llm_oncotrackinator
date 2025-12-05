"""
Tests for models module.
"""

import pytest
from datetime import datetime

from llm_oncotrackinator.models import (
    Lesion,
    TimePoint,
    PatientLesionHistory,
    ExtractionResult
)


class TestLesion:
    """Tests for Lesion model."""

    def test_create_lesion_with_all_fields(self):
        """Test creating a lesion with all fields."""
        lesion = Lesion(
            lesion_id="L1",
            location="right upper lobe",
            size_mm=23.0,
            size_cm=2.3,
            characteristics="nodular",
            timepoint_date=datetime(2024, 1, 15),
            raw_text="2.3 cm nodule in the right upper lobe"
        )
        assert lesion.lesion_id == "L1"
        assert lesion.location == "right upper lobe"
        assert lesion.size_cm == 2.3
        assert lesion.size_mm == 23.0

    def test_create_lesion_with_minimal_fields(self):
        """Test creating a lesion with only required fields."""
        lesion = Lesion(
            lesion_id="L1",
            location="liver",
            timepoint_date=datetime(2024, 1, 15)
        )
        assert lesion.lesion_id == "L1"
        assert lesion.location == "liver"
        assert lesion.size_cm is None
        assert lesion.size_mm is None
        assert lesion.characteristics is None

    def test_lesion_json_serialization(self):
        """Test that lesion can be serialized to JSON."""
        lesion = Lesion(
            lesion_id="L1",
            location="right upper lobe",
            size_cm=2.3,
            timepoint_date=datetime(2024, 1, 15)
        )
        data = lesion.model_dump(mode='json')
        assert data["lesion_id"] == "L1"
        assert data["location"] == "right upper lobe"
        assert isinstance(data["timepoint_date"], str)  # Should be ISO format string


class TestTimePoint:
    """Tests for TimePoint model."""

    def test_create_timepoint(self):
        """Test creating a timepoint."""
        lesions = [
            Lesion(
                lesion_id="L1",
                location="right upper lobe",
                size_cm=2.3,
                timepoint_date=datetime(2024, 1, 15)
            ),
            Lesion(
                lesion_id="L2",
                location="left lower lobe",
                size_cm=1.5,
                timepoint_date=datetime(2024, 1, 15)
            )
        ]

        tp = TimePoint(
            date=datetime(2024, 1, 15),
            report_text="CT scan shows two nodules.",
            lesions=lesions
        )

        assert tp.date == datetime(2024, 1, 15)
        assert len(tp.lesions) == 2
        assert tp.lesions[0].lesion_id == "L1"

    def test_create_timepoint_without_lesions(self):
        """Test creating a timepoint with no lesions."""
        tp = TimePoint(
            date=datetime(2024, 1, 15),
            report_text="No significant findings."
        )
        assert len(tp.lesions) == 0


class TestPatientLesionHistory:
    """Tests for PatientLesionHistory model."""

    def test_create_patient_history(self):
        """Test creating a patient history."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[],
            summary="Test patient"
        )
        assert history.patient_id == "P001"
        assert len(history.timepoints) == 0
        assert history.summary == "Test patient"

    def test_get_lesion_timeline(self):
        """Test getting timeline for a specific lesion."""
        timepoints = [
            TimePoint(
                date=datetime(2024, 1, 15),
                report_text="First scan",
                lesions=[
                    Lesion(
                        lesion_id="L1",
                        location="right upper lobe",
                        size_cm=2.3,
                        timepoint_date=datetime(2024, 1, 15)
                    )
                ]
            ),
            TimePoint(
                date=datetime(2024, 3, 20),
                report_text="Follow-up scan",
                lesions=[
                    Lesion(
                        lesion_id="L1",
                        location="right upper lobe",
                        size_cm=2.8,
                        timepoint_date=datetime(2024, 3, 20)
                    )
                ]
            )
        ]

        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=timepoints
        )

        timeline = history.get_lesion_timeline("L1")
        assert len(timeline) == 2
        assert timeline[0].size_cm == 2.3
        assert timeline[1].size_cm == 2.8
        # Verify chronological order
        assert timeline[0].timepoint_date < timeline[1].timepoint_date

    def test_get_lesion_timeline_nonexistent(self):
        """Test getting timeline for a lesion that doesn't exist."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[]
        )
        timeline = history.get_lesion_timeline("L999")
        assert len(timeline) == 0

    def test_get_all_lesion_ids(self):
        """Test getting all unique lesion IDs."""
        timepoints = [
            TimePoint(
                date=datetime(2024, 1, 15),
                report_text="First scan",
                lesions=[
                    Lesion(
                        lesion_id="L1",
                        location="right upper lobe",
                        size_cm=2.3,
                        timepoint_date=datetime(2024, 1, 15)
                    ),
                    Lesion(
                        lesion_id="L2",
                        location="liver",
                        size_cm=4.5,
                        timepoint_date=datetime(2024, 1, 15)
                    )
                ]
            ),
            TimePoint(
                date=datetime(2024, 3, 20),
                report_text="Follow-up scan",
                lesions=[
                    Lesion(
                        lesion_id="L1",
                        location="right upper lobe",
                        size_cm=2.8,
                        timepoint_date=datetime(2024, 3, 20)
                    ),
                    Lesion(
                        lesion_id="L3",
                        location="left lower lobe",
                        size_cm=1.5,
                        timepoint_date=datetime(2024, 3, 20)
                    )
                ]
            )
        ]

        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=timepoints
        )

        lesion_ids = history.get_all_lesion_ids()
        assert len(lesion_ids) == 3
        assert "L1" in lesion_ids
        assert "L2" in lesion_ids
        assert "L3" in lesion_ids
        # Should be sorted
        assert lesion_ids == sorted(lesion_ids)

    def test_patient_history_json_serialization(self):
        """Test that patient history can be serialized to JSON."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="Test",
                    lesions=[
                        Lesion(
                            lesion_id="L1",
                            location="test location",
                            timepoint_date=datetime(2024, 1, 15)
                        )
                    ]
                )
            ]
        )
        data = history.model_dump(mode='json')
        assert data["patient_id"] == "P001"
        assert len(data["timepoints"]) == 1
        assert isinstance(data["timepoints"][0]["date"], str)


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_successful_extraction(self):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            lesions=[{"lesion_id": "L1", "location": "lung"}],
            raw_response='[{"lesion_id": "L1", "location": "lung"}]',
            success=True
        )
        assert result.success is True
        assert len(result.lesions) == 1
        assert result.error_message is None

    def test_create_failed_extraction(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            lesions=[],
            raw_response="",
            success=False,
            error_message="Connection failed"
        )
        assert result.success is False
        assert len(result.lesions) == 0
        assert result.error_message == "Connection failed"
