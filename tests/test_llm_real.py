"""
Real LLM integration tests.

These tests actually call Ollama and verify the quality of extractions.
They are skipped by default and should be run manually when Ollama is available.

Run with: pytest tests/test_llm_real.py -v -s
"""

import pytest
from datetime import datetime

from llm_oncotrackinator import Config, LesionExtractor, LesionTracker, MedicalReport


@pytest.mark.integration
class TestRealLLMQuality:
    """Test actual LLM extraction quality."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with deterministic settings."""
        return LesionExtractor(config=Config(
            ollama_model="llama3.1:8b",
            temperature=0.0  # Deterministic
        ))

    def test_extract_single_clear_lesion(self, extractor):
        """Test extracting a single, clearly described lesion."""
        report = "CT scan of chest shows a 2.3 cm nodule in the right upper lobe."

        result = extractor.extract_first_timepoint(report)

        assert result.success, f"Extraction failed: {result.error_message}"
        assert len(result.lesions) >= 1, "Should extract at least one lesion"

        # Check that it extracted the nodule
        lesion = result.lesions[0]
        location = lesion.get("location", "").lower()
        assert "lobe" in location or "lung" in location, f"Location incorrect: {location}"

        # Check size extraction
        size_cm = lesion.get("size_cm")
        assert size_cm is not None, "Should extract size"
        assert 2.0 <= size_cm <= 2.5, f"Size should be around 2.3, got {size_cm}"

    def test_extract_multiple_lesions(self, extractor):
        """Test extracting multiple lesions from one report."""
        report = (
            "CT demonstrates a 2.3 cm nodule in the right upper lobe. "
            "Additionally, a 4.5 cm mass is noted in liver segment 7. "
            "Small 0.8 cm nodule in left lower lobe."
        )

        result = extractor.extract_first_timepoint(report)

        assert result.success
        assert len(result.lesions) >= 2, f"Should extract at least 2 lesions, got {len(result.lesions)}"

        # Check that different locations were identified
        locations = [l.get("location", "").lower() for l in result.lesions]
        assert any("liver" in loc for loc in locations), "Should identify liver lesion"
        assert any("lobe" in loc or "lung" in loc for loc in locations), "Should identify lung lesion"

    def test_extract_no_lesions(self, extractor):
        """Test handling of reports with no significant findings."""
        report = "CT scan of chest and abdomen shows no focal lesions. No significant abnormalities."

        result = extractor.extract_first_timepoint(report)

        assert result.success
        # Should return empty array or very minimal findings
        assert len(result.lesions) == 0, "Should not extract lesions from negative report"

    def test_follow_up_maintains_tracking(self, extractor):
        """Test that follow-up extraction maintains lesion IDs."""
        first_report = "CT shows a 2.3 cm nodule in the right upper lobe."

        # Extract first timepoint
        first_result = extractor.extract_first_timepoint(first_report)
        assert first_result.success

        # Simulate assigning IDs (normally done by tracker)
        previous_lesions = []
        for idx, lesion in enumerate(first_result.lesions, 1):
            lesion["lesion_id"] = f"L{idx}"
            previous_lesions.append({
                "lesion_id": lesion["lesion_id"],
                "location": lesion.get("location"),
                "size_cm": lesion.get("size_cm")
            })

        # Extract follow-up
        followup_report = "Follow-up CT shows the right upper lobe nodule has increased to 2.8 cm."
        followup_result = extractor.extract_followup_timepoint(followup_report, previous_lesions)

        assert followup_result.success
        assert len(followup_result.lesions) >= 1

        # Check that it maintained the lesion ID
        lesion = followup_result.lesions[0]
        assert "lesion_id" in lesion, "Should include lesion_id"
        assert lesion["lesion_id"] == "L1", f"Should maintain L1, got {lesion.get('lesion_id')}"

    def test_extract_with_characteristics(self, extractor):
        """Test extraction includes lesion characteristics."""
        report = "MRI brain reveals a 1.2 cm enhancing lesion in the left frontal lobe."

        result = extractor.extract_first_timepoint(report)

        assert result.success
        assert len(result.lesions) >= 1

        lesion = result.lesions[0]
        characteristics = lesion.get("characteristics", "").lower()
        # Should capture "enhancing"
        assert characteristics is not None and len(characteristics) > 0, "Should extract characteristics"


@pytest.mark.integration
class TestRealEndToEnd:
    """Test complete end-to-end pipeline with real LLM."""

    def test_track_patient_with_progression(self):
        """Test tracking a patient with lesion progression."""
        config = Config(temperature=0.0, ollama_model="llama3.1:8b")
        tracker = LesionTracker(config=config)

        reports = [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="CT scan of chest shows a 2.3 cm nodule in the right upper lobe."
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Follow-up CT shows the right upper lobe nodule has increased to 2.8 cm."
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 6, 10),
                report_text="CT demonstrates further growth of right upper lobe lesion to 3.2 cm. New 1.5 cm nodule identified in left lower lobe."
            )
        ]

        history = tracker.track_patient("P001", reports)

        # Verify structure
        assert history.patient_id == "P001"
        assert len(history.timepoints) == 3

        # Verify lesions were extracted
        assert len(history.timepoints[0].lesions) >= 1, "First timepoint should have lesions"
        assert len(history.timepoints[1].lesions) >= 1, "Second timepoint should have lesions"
        assert len(history.timepoints[2].lesions) >= 1, "Third timepoint should have lesions"

        # Verify progression tracking
        lesion_ids = history.get_all_lesion_ids()
        assert len(lesion_ids) >= 1, "Should track at least one lesion"

        # Get timeline for first lesion
        if lesion_ids:
            timeline = history.get_lesion_timeline(lesion_ids[0])
            print(f"\nLesion {lesion_ids[0]} timeline:")
            for obs in timeline:
                print(f"  {obs.timepoint_date.date()}: {obs.size_cm} cm at {obs.location}")

            # Should show progression
            assert len(timeline) >= 2, "Lesion should appear in multiple timepoints"


@pytest.mark.integration
class TestLLMResponseQuality:
    """Test the quality and consistency of LLM responses."""

    def test_json_format_consistency(self):
        """Test that LLM consistently returns valid JSON."""
        extractor = LesionExtractor(config=Config(temperature=0.0))

        test_reports = [
            "CT shows a 2.3 cm nodule in the right lung.",
            "MRI reveals a 1.5 cm lesion in the brain.",
            "PET scan demonstrates a 3.0 cm mass in the liver.",
            "No significant findings on imaging.",
        ]

        for report in test_reports:
            result = extractor.extract_first_timepoint(report)
            assert result.success, f"Failed to extract from: {report}"
            assert isinstance(result.lesions, list), "Should return a list"

    def test_deterministic_extraction(self):
        """Test that temperature=0 gives consistent results."""
        config = Config(temperature=0.0, ollama_model="llama3.1:8b")
        extractor = LesionExtractor(config=config)

        report = "CT scan shows a 2.3 cm nodule in the right upper lobe."

        # Extract twice
        result1 = extractor.extract_first_timepoint(report)
        result2 = extractor.extract_first_timepoint(report)

        assert result1.success and result2.success
        assert len(result1.lesions) == len(result2.lesions), "Should be deterministic"

        # Results should be very similar (allowing for minor variations)
        if len(result1.lesions) > 0:
            loc1 = result1.lesions[0].get("location", "").lower()
            loc2 = result2.lesions[0].get("location", "").lower()
            # Locations should at least share key terms
            assert "lobe" in loc1 and "lobe" in loc2, "Locations should be consistent"

    def test_size_unit_conversion(self):
        """Test that LLM properly extracts and converts sizes."""
        extractor = LesionExtractor(config=Config(temperature=0.0))

        report = "CT shows a 23 mm nodule in the lung."

        result = extractor.extract_first_timepoint(report)
        assert result.success

        if len(result.lesions) > 0:
            lesion = result.lesions[0]
            # Should extract mm and ideally also cm
            size_mm = lesion.get("size_mm")
            size_cm = lesion.get("size_cm")

            assert size_mm is not None or size_cm is not None, "Should extract size"

            if size_mm:
                assert 20 <= size_mm <= 25, f"Size should be around 23mm, got {size_mm}"
            if size_cm:
                assert 2.0 <= size_cm <= 2.5, f"Size should be around 2.3cm, got {size_cm}"


# Helper to check if Ollama is available
def is_ollama_available():
    """Check if Ollama is running and accessible."""
    try:
        import ollama
        ollama.list()
        return True
    except:
        return False


# This can be run standalone to test Ollama connectivity
if __name__ == "__main__":
    if is_ollama_available():
        print("Ollama is available!")
        print("You can run these tests with: pytest tests/test_llm_real.py -v -s")
        print("Make sure to remove the skipif decorator first.")
    else:
        print("Ollama is not available or not running.")
        print("Install Ollama from https://ollama.ai/")
        print("Then run: ollama pull llama3.1:8b")
