"""
Tests for tracker module.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from llm_oncotrackinator import Config, LesionTracker, MedicalReport
from llm_oncotrackinator.models import ExtractionResult, Lesion


class TestLesionTracker:
    """Tests for LesionTracker class."""

    def test_initialization(self):
        """Test LesionTracker initialization."""
        config = Config(ollama_model="llama3.1:8b")
        tracker = LesionTracker(config=config)
        assert tracker.config.ollama_model == "llama3.1:8b"
        assert tracker.extractor is not None

    def test_initialization_with_default_config(self):
        """Test LesionTracker initialization with default config."""
        tracker = LesionTracker()
        assert tracker.config is not None
        assert isinstance(tracker.config, Config)

    def test_create_lesion(self):
        """Test _create_lesion helper method."""
        tracker = LesionTracker()
        lesion_data = {
            "location": "right upper lobe",
            "size_cm": 2.3,
            "size_mm": 23.0,
            "characteristics": "nodular",
            "raw_text": "2.3 cm nodule"
        }
        lesion = tracker._create_lesion(
            "L1",
            lesion_data,
            datetime(2024, 1, 15)
        )
        assert lesion.lesion_id == "L1"
        assert lesion.location == "right upper lobe"
        assert lesion.size_cm == 2.3
        assert lesion.timepoint_date == datetime(2024, 1, 15)

    def test_create_lesion_minimal_data(self):
        """Test _create_lesion with minimal data."""
        tracker = LesionTracker()
        lesion_data = {"location": "liver"}
        lesion = tracker._create_lesion(
            "L1",
            lesion_data,
            datetime(2024, 1, 15)
        )
        assert lesion.lesion_id == "L1"
        assert lesion.location == "liver"
        assert lesion.size_cm is None
        assert lesion.size_mm is None

    def test_create_lesion_missing_location(self):
        """Test _create_lesion handles missing location."""
        tracker = LesionTracker()
        lesion_data = {"size_cm": 2.3}
        lesion = tracker._create_lesion(
            "L1",
            lesion_data,
            datetime(2024, 1, 15)
        )
        assert lesion.location == "Unknown"

    def test_extract_lesion_summaries(self):
        """Test _extract_lesion_summaries method."""
        tracker = LesionTracker()
        lesions = [
            Lesion(
                lesion_id="L1",
                location="right upper lobe",
                size_cm=2.3,
                size_mm=23.0,
                characteristics="nodular",
                timepoint_date=datetime(2024, 1, 15)
            ),
            Lesion(
                lesion_id="L2",
                location="liver",
                timepoint_date=datetime(2024, 1, 15)
            )
        ]
        summaries = tracker._extract_lesion_summaries(lesions)
        assert len(summaries) == 2
        assert summaries[0]["lesion_id"] == "L1"
        assert summaries[0]["location"] == "right upper lobe"
        assert summaries[0]["size_cm"] == 2.3
        assert summaries[1]["lesion_id"] == "L2"
        assert "size_cm" not in summaries[1]  # Should not include None values

    def test_track_patient_empty_reports_raises_error(self):
        """Test that tracking with empty reports list raises error."""
        tracker = LesionTracker()
        with pytest.raises(ValueError, match="Reports list cannot be empty"):
            tracker.track_patient("P001", [])

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_first_timepoint')
    def test_process_first_timepoint(self, mock_extract):
        """Test processing the first timepoint."""
        # Mock the extraction result
        mock_extract.return_value = ExtractionResult(
            lesions=[
                {"location": "right upper lobe", "size_cm": 2.3}
            ],
            raw_response="mock response",
            success=True
        )

        tracker = LesionTracker()
        report = MedicalReport(
            patient_id="P001",
            date=datetime(2024, 1, 15),
            report_text="Test report"
        )

        timepoint = tracker._process_first_timepoint(report)

        assert timepoint.date == datetime(2024, 1, 15)
        assert len(timepoint.lesions) == 1
        assert timepoint.lesions[0].lesion_id == "L1"  # Auto-generated
        assert timepoint.lesions[0].location == "right upper lobe"

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_first_timepoint')
    def test_process_first_timepoint_extraction_failure(self, mock_extract):
        """Test processing first timepoint when extraction fails."""
        mock_extract.return_value = ExtractionResult(
            lesions=[],
            raw_response="",
            success=False,
            error_message="Test error"
        )

        tracker = LesionTracker()
        report = MedicalReport(
            patient_id="P001",
            date=datetime(2024, 1, 15),
            report_text="Test report"
        )

        timepoint = tracker._process_first_timepoint(report)

        assert timepoint.date == datetime(2024, 1, 15)
        assert len(timepoint.lesions) == 0

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_followup_timepoint')
    def test_process_followup_timepoint(self, mock_extract):
        """Test processing a follow-up timepoint."""
        mock_extract.return_value = ExtractionResult(
            lesions=[
                {"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.8}
            ],
            raw_response="mock response",
            success=True
        )

        tracker = LesionTracker()
        report = MedicalReport(
            patient_id="P001",
            date=datetime(2024, 3, 20),
            report_text="Follow-up report"
        )
        previous_lesions = [{"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.3}]

        timepoint = tracker._process_followup_timepoint(report, previous_lesions)

        assert timepoint.date == datetime(2024, 3, 20)
        assert len(timepoint.lesions) == 1
        assert timepoint.lesions[0].lesion_id == "L1"  # Maintained from LLM
        assert timepoint.lesions[0].size_cm == 2.8

    def test_track_all_patients_empty_dict(self):
        """Test tracking all patients with empty dictionary."""
        tracker = LesionTracker()
        results = tracker.track_all_patients({})
        assert len(results) == 0

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_first_timepoint')
    def test_track_all_patients_with_failure(self, mock_extract):
        """Test that tracking continues even if one patient fails."""
        mock_extract.side_effect = Exception("Test error")

        tracker = LesionTracker()
        reports = {
            "P001": [
                MedicalReport(
                    patient_id="P001",
                    date=datetime(2024, 1, 15),
                    report_text="Test"
                )
            ]
        }

        results = tracker.track_all_patients(reports)

        assert len(results) == 1
        assert results[0].patient_id == "P001"
        assert "Tracking failed" in results[0].summary


class TestTrackerIntegration:
    """Integration-style tests for the tracker (without actual LLM calls)."""

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_first_timepoint')
    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_followup_timepoint')
    def test_track_patient_multiple_timepoints(self, mock_followup, mock_first):
        """Test tracking a patient across multiple timepoints."""
        # Mock first timepoint extraction
        mock_first.return_value = ExtractionResult(
            lesions=[
                {"location": "right upper lobe", "size_cm": 2.3}
            ],
            raw_response="",
            success=True
        )

        # Mock follow-up extraction
        mock_followup.return_value = ExtractionResult(
            lesions=[
                {"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.8}
            ],
            raw_response="",
            success=True
        )

        tracker = LesionTracker()
        reports = [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="First scan"
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Follow-up scan"
            )
        ]

        history = tracker.track_patient("P001", reports)

        assert history.patient_id == "P001"
        assert len(history.timepoints) == 2
        assert len(history.get_all_lesion_ids()) == 1
        assert "L1" in history.get_all_lesion_ids()

        # Verify timeline
        timeline = history.get_lesion_timeline("L1")
        assert len(timeline) == 2
        assert timeline[0].size_cm == 2.3
        assert timeline[1].size_cm == 2.8

    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_first_timepoint')
    @patch('llm_oncotrackinator.lesion_extractor.LesionExtractor.extract_followup_timepoint')
    def test_track_patient_reports_sorted_by_date(self, mock_followup, mock_first):
        """Test that reports are sorted by date even if provided out of order."""
        mock_first.return_value = ExtractionResult(lesions=[], raw_response="", success=True)
        mock_followup.return_value = ExtractionResult(lesions=[], raw_response="", success=True)

        tracker = LesionTracker()
        # Provide reports out of chronological order
        reports = [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Second scan"
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="First scan"
            )
        ]

        history = tracker.track_patient("P001", reports)

        # Should be sorted by date
        assert history.timepoints[0].date == datetime(2024, 1, 15)
        assert history.timepoints[1].date == datetime(2024, 3, 20)
