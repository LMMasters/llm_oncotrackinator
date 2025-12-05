"""
Tests for data_loader module.
"""

import pytest
from datetime import datetime
from pathlib import Path

from llm_oncotrackinator import DataLoader, Config, MedicalReport


class TestMedicalReport:
    """Tests for MedicalReport model."""

    def test_valid_report(self):
        """Test creating a valid medical report."""
        report = MedicalReport(
            patient_id="P001",
            date=datetime(2024, 1, 15),
            report_text="CT scan shows a nodule."
        )
        assert report.patient_id == "P001"
        assert report.date == datetime(2024, 1, 15)
        assert report.report_text == "CT scan shows a nodule."

    def test_empty_report_text_raises_error(self):
        """Test that empty report text raises ValueError."""
        with pytest.raises(ValueError, match="Report text cannot be empty"):
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text=""
            )

    def test_empty_patient_id_raises_error(self):
        """Test that empty patient ID raises ValueError."""
        with pytest.raises(ValueError, match="Patient ID cannot be empty"):
            MedicalReport(
                patient_id="",
                date=datetime(2024, 1, 15),
                report_text="CT scan shows a nodule."
            )

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from patient_id and report_text."""
        report = MedicalReport(
            patient_id="  P001  ",
            date=datetime(2024, 1, 15),
            report_text="  CT scan shows a nodule.  "
        )
        assert report.patient_id == "P001"
        assert report.report_text == "CT scan shows a nodule."


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_initialization_with_default_config(self):
        """Test DataLoader initialization with default config."""
        loader = DataLoader()
        assert loader.config.patient_id_column == "patient_id"
        assert loader.config.date_column == "date"
        assert loader.config.report_column == "report"

    def test_initialization_with_custom_config(self):
        """Test DataLoader initialization with custom config."""
        config = Config(
            patient_id_column="id",
            date_column="report_date",
            report_column="text"
        )
        loader = DataLoader(config=config)
        assert loader.config.patient_id_column == "id"
        assert loader.config.date_column == "report_date"
        assert loader.config.report_column == "text"

    def test_load_csv_sample_data(self):
        """Test loading the sample CSV data."""
        loader = DataLoader()
        sample_path = Path(__file__).parent.parent / "examples" / "sample_data.csv"

        if not sample_path.exists():
            pytest.skip("Sample data file not found")

        reports = loader.load_csv(str(sample_path))

        assert len(reports) > 0
        assert all(isinstance(r, MedicalReport) for r in reports)
        assert all(r.patient_id for r in reports)
        assert all(r.report_text for r in reports)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent_file.csv")

    def test_get_patient_timelines(self):
        """Test organizing reports into patient timelines."""
        loader = DataLoader()
        reports = [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="First report"
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Second report"
            ),
            MedicalReport(
                patient_id="P002",
                date=datetime(2024, 2, 1),
                report_text="Report for patient 2"
            ),
        ]

        timelines = loader.get_patient_timelines(reports)

        assert len(timelines) == 2
        assert "P001" in timelines
        assert "P002" in timelines
        assert len(timelines["P001"]) == 2
        assert len(timelines["P002"]) == 1

        # Verify chronological order
        assert timelines["P001"][0].date < timelines["P001"][1].date
