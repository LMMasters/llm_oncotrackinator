"""
Integration tests for the complete pipeline.

These tests require Ollama to be running with a model installed.
Mark them with @pytest.mark.integration to skip by default.

Run with: pytest -v -m integration
"""

import pytest
import tempfile
import json
from pathlib import Path

from llm_oncotrackinator import (
    Config,
    DataLoader,
    LesionTracker,
    OutputGenerator
)


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def integration_config(self):
        """Config for integration tests."""
        return Config(
            ollama_model="llama3.1:8b",
            temperature=0.0,
            max_retries=2
        )

    def test_data_loading_pipeline(self, sample_reports, integration_config):
        """Test that data can be loaded and organized."""
        loader = DataLoader(config=integration_config)

        # Create reports from fixture
        timelines = loader.get_patient_timelines(sample_reports)

        assert len(timelines) == 1
        assert "P001" in timelines
        assert len(timelines["P001"]) == 3

    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_single_patient_tracking(self, sample_reports, integration_config):
        """Test tracking a single patient through the pipeline."""
        tracker = LesionTracker(config=integration_config)

        # Track the patient
        history = tracker.track_patient("P001", sample_reports)

        # Verify results
        assert history.patient_id == "P001"
        assert len(history.timepoints) == 3
        assert len(history.timepoints[0].lesions) > 0  # Should extract lesions

    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_multiple_patient_tracking(self, multiple_patient_reports, integration_config):
        """Test tracking multiple patients."""
        tracker = LesionTracker(config=integration_config)

        histories = tracker.track_all_patients(multiple_patient_reports)

        assert len(histories) == 2
        assert any(h.patient_id == "P001" for h in histories)
        assert any(h.patient_id == "P002" for h in histories)

    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_output_generation(self, sample_reports, integration_config):
        """Test the complete pipeline including output generation."""
        tracker = LesionTracker(config=integration_config)
        history = tracker.track_patient("P001", sample_reports)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate JSON output
            json_path = Path(tmpdir) / "results.json"
            OutputGenerator.to_json(history, file_path=str(json_path))

            assert json_path.exists()

            # Verify JSON structure
            with open(json_path, 'r') as f:
                data = json.load(f)

            assert data["total_patients"] == 1
            assert data["patients"][0]["patient_id"] == "P001"

            # Generate summary
            summary_path = Path(tmpdir) / "summary.txt"
            OutputGenerator.save_summary([history], file_path=str(summary_path))

            assert summary_path.exists()

            with open(summary_path, 'r') as f:
                summary_content = f.read()

            assert "P001" in summary_content

    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_csv_to_tracked_output(self, integration_config):
        """Test complete pipeline from CSV to tracked output."""
        # This test would load from the sample CSV file
        loader = DataLoader(config=integration_config)

        # Find the sample data file
        sample_csv = Path(__file__).parent.parent / "examples" / "sample_data.csv"

        if not sample_csv.exists():
            pytest.skip("Sample data file not found")

        reports = loader.load_csv(str(sample_csv))
        timelines = loader.get_patient_timelines(reports)

        tracker = LesionTracker(config=integration_config)
        histories = tracker.track_all_patients(timelines)

        # Should have 3 patients from sample data
        assert len(histories) == 3

        # Generate outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            OutputGenerator.to_json(histories, file_path=str(json_path))

            assert json_path.exists()


class TestPipelineWithoutLLM:
    """Pipeline tests that don't require LLM (using mocks)."""

    def test_data_loader_to_tracker_interface(self, sample_reports):
        """Test that DataLoader output works with LesionTracker input."""
        loader = DataLoader()
        timelines = loader.get_patient_timelines(sample_reports)

        # Verify the interface between components
        assert isinstance(timelines, dict)
        assert "P001" in timelines
        assert all(hasattr(r, 'patient_id') for r in timelines["P001"])
        assert all(hasattr(r, 'date') for r in timelines["P001"])
        assert all(hasattr(r, 'report_text') for r in timelines["P001"])

    def test_output_generator_accepts_tracker_output(self, sample_reports):
        """Test that OutputGenerator can handle tracker output structure."""
        from llm_oncotrackinator.models import PatientLesionHistory, TimePoint
        from datetime import datetime

        # Create minimal history
        history = PatientLesionHistory(
            patient_id="P001",
            timepoints=[
                TimePoint(
                    date=datetime(2024, 1, 15),
                    report_text="Test report",
                    lesions=[]
                )
            ]
        )

        # Should be able to generate JSON
        json_str = OutputGenerator.to_json(history)
        assert json_str is not None
        data = json.loads(json_str)
        assert data["patients"][0]["patient_id"] == "P001"

        # Should be able to generate summary
        summary = OutputGenerator.to_summary([history])
        assert "P001" in summary


@pytest.mark.integration
@pytest.mark.skip(reason="Requires Ollama - enable manually for testing")
class TestRealLLMExtraction:
    """Tests that actually call Ollama - only run manually."""

    def test_extract_single_lesion(self):
        """Test extracting a single clear lesion."""
        from llm_oncotrackinator import LesionExtractor, Config

        config = Config(temperature=0.0)
        extractor = LesionExtractor(config=config)

        report = "CT scan shows a 2.3 cm nodule in the right upper lobe."

        result = extractor.extract_first_timepoint(report)

        assert result.success
        assert len(result.lesions) > 0
        # Should extract the nodule
        assert any("lobe" in str(l.get("location", "")).lower() for l in result.lesions)

    def test_extract_multiple_lesions(self):
        """Test extracting multiple lesions."""
        from llm_oncotrackinator import LesionExtractor, Config

        config = Config(temperature=0.0)
        extractor = LesionExtractor(config=config)

        report = "CT shows a 2.3 cm nodule in the right upper lobe and a 4.5 cm mass in the liver segment 7."

        result = extractor.extract_first_timepoint(report)

        assert result.success
        assert len(result.lesions) >= 2  # Should extract both lesions
