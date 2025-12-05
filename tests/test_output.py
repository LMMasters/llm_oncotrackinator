"""
Tests for output module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from llm_oncotrackinator.output import OutputGenerator
from llm_oncotrackinator.models import (
    Lesion,
    TimePoint,
    PatientLesionHistory
)


class TestOutputGenerator:
    """Tests for OutputGenerator class."""

    def test_to_json_single_patient(self):
        """Test converting a single patient history to JSON."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="Test report",
                    lesions=[
                        Lesion(
                            lesion_id="L1",
                            location="right upper lobe",
                            size_cm=2.3,
                            timepoint_date=datetime(2024, 1, 15)
                        )
                    ]
                )
            ]
        )

        json_str = OutputGenerator.to_json(history)

        # Parse to verify it's valid JSON
        data = json.loads(json_str)
        assert "generated_at" in data
        assert data["total_patients"] == 1
        assert len(data["patients"]) == 1
        assert data["patients"][0]["patient_id"] == "P001"

    def test_to_json_multiple_patients(self):
        """Test converting multiple patient histories to JSON."""
        histories = [
            PatientLesionHistory(patient_id="P001", timepoints=[]),
            PatientLesionHistory(patient_id="P002", timepoints=[])
        ]

        json_str = OutputGenerator.to_json(histories)
        data = json.loads(json_str)

        assert data["total_patients"] == 2
        assert len(data["patients"]) == 2

    def test_to_json_with_file_save(self):
        """Test saving JSON to a file."""
        history = PatientLesionHistory(patient_id="P001", timepoints=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output" / "test.json"

            json_str = OutputGenerator.to_json(history, file_path=str(file_path))

            # Verify file was created
            assert file_path.exists()

            # Verify file content
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            assert file_data["total_patients"] == 1

    def test_to_json_custom_indent(self):
        """Test JSON generation with custom indentation."""
        history = PatientLesionHistory(patient_id="P001", timepoints=[])

        json_str = OutputGenerator.to_json(history, indent=4)

        # Check that indentation is applied (4 spaces)
        assert "    " in json_str

    def test_to_summary_empty_histories(self):
        """Test generating summary with empty histories list."""
        summary = OutputGenerator.to_summary([])

        assert "Total Patients: 0" in summary
        assert "=" in summary  # Has header

    def test_to_summary_single_patient_no_lesions(self):
        """Test summary for patient with no lesions."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="No findings",
                    lesions=[]
                )
            ]
        )

        summary = OutputGenerator.to_summary([history])

        assert "Patient: P001" in summary
        assert "Timepoints: 1" in summary
        assert "Unique Lesions: 0" in summary

    def test_to_summary_patient_with_lesions(self):
        """Test summary for patient with lesions."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
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
        )

        summary = OutputGenerator.to_summary([history])

        assert "Patient: P001" in summary
        assert "Timepoints: 2" in summary
        assert "Unique Lesions: 1" in summary
        assert "L1" in summary
        assert "right upper lobe" in summary
        assert "2.3 cm" in summary
        assert "2.8 cm" in summary

    def test_to_summary_multiple_lesions(self):
        """Test summary with multiple lesions."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="Scan",
                    lesions=[
                        Lesion(
                            lesion_id="L1",
                            location="lung",
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
                )
            ]
        )

        summary = OutputGenerator.to_summary([history])

        assert "Unique Lesions: 2" in summary
        assert "L1, L2" in summary or "L1" in summary and "L2" in summary

    def test_to_summary_lesion_with_mm_size(self):
        """Test summary displays mm size when cm is not available."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="Scan",
                    lesions=[
                        Lesion(
                            lesion_id="L1",
                            location="lung",
                            size_mm=15.0,
                            timepoint_date=datetime(2024, 1, 15)
                        )
                    ]
                )
            ]
        )

        summary = OutputGenerator.to_summary([history])

        assert "15.0 mm" in summary or "15 mm" in summary

    def test_to_summary_date_range(self):
        """Test that summary shows date range correctly."""
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="First",
                    lesions=[]
                ),
                TimePoint(
                    date=datetime(2024, 6, 10),
                    report_text="Last",
                    lesions=[]
                )
            ]
        )

        summary = OutputGenerator.to_summary([history])

        assert "2024-01-15" in summary
        assert "2024-06-10" in summary

    def test_save_summary(self):
        """Test saving summary to a file."""
        history = PatientLesionHistory(patient_id="P001", timepoints=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output" / "summary.txt"

            OutputGenerator.save_summary([history], file_path=str(file_path))

            # Verify file was created
            assert file_path.exists()

            # Verify file content
            with open(file_path, 'r') as f:
                content = f.read()
            assert "Patient: P001" in content
            assert "LESION TRACKING SUMMARY" in content

    def test_save_summary_creates_parent_directory(self):
        """Test that save_summary creates parent directories if they don't exist."""
        history = PatientLesionHistory(patient_id="P001", timepoints=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            file_path = Path(tmpdir) / "a" / "b" / "c" / "summary.txt"

            OutputGenerator.save_summary([history], file_path=str(file_path))

            assert file_path.exists()

    def test_to_json_creates_parent_directory(self):
        """Test that to_json creates parent directories if they don't exist."""
        history = PatientLesionHistory(patient_id="P001", timepoints=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            file_path = Path(tmpdir) / "x" / "y" / "z" / "output.json"

            OutputGenerator.to_json(history, file_path=str(file_path))

            assert file_path.exists()
