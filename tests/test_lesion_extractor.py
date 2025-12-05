"""
Tests for lesion_extractor module.
"""

import pytest
import json

from llm_oncotrackinator import Config, LesionExtractor
from llm_oncotrackinator.models import ExtractionResult


class TestLesionExtractor:
    """Tests for LesionExtractor class."""

    def test_initialization(self):
        """Test LesionExtractor initialization."""
        config = Config(ollama_model="llama3.1:8b")
        extractor = LesionExtractor(config=config)
        assert extractor.config.ollama_model == "llama3.1:8b"

    def test_initialization_with_default_config(self):
        """Test LesionExtractor initialization with default config."""
        extractor = LesionExtractor()
        assert extractor.config is not None
        assert isinstance(extractor.config, Config)

    def test_parse_json_response_valid(self):
        """Test parsing a valid JSON response."""
        extractor = LesionExtractor()
        response = '[{"lesion_id": "L1", "location": "lung", "size_cm": 2.3}]'
        lesions = extractor._parse_json_response(response)
        assert len(lesions) == 1
        assert lesions[0]["lesion_id"] == "L1"
        assert lesions[0]["location"] == "lung"

    def test_parse_json_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        extractor = LesionExtractor()
        response = '```json\n[{"lesion_id": "L1", "location": "lung"}]\n```'
        lesions = extractor._parse_json_response(response)
        assert len(lesions) == 1
        assert lesions[0]["lesion_id"] == "L1"

    def test_parse_json_response_with_generic_markdown(self):
        """Test parsing JSON wrapped in generic markdown blocks."""
        extractor = LesionExtractor()
        response = '```\n[{"lesion_id": "L1", "location": "lung"}]\n```'
        lesions = extractor._parse_json_response(response)
        assert len(lesions) == 1

    def test_parse_json_response_empty_array(self):
        """Test parsing an empty JSON array."""
        extractor = LesionExtractor()
        response = '[]'
        lesions = extractor._parse_json_response(response)
        assert len(lesions) == 0

    def test_parse_json_response_invalid(self):
        """Test parsing invalid JSON raises error."""
        extractor = LesionExtractor()
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            extractor._parse_json_response('not valid json')

    def test_parse_json_response_not_array(self):
        """Test parsing JSON that's not an array raises error."""
        extractor = LesionExtractor()
        with pytest.raises(ValueError, match="not a JSON array"):
            extractor._parse_json_response('{"key": "value"}')

    def test_parse_json_response_with_whitespace(self):
        """Test parsing JSON with extra whitespace."""
        extractor = LesionExtractor()
        response = '  \n  [{"lesion_id": "L1"}]  \n  '
        lesions = extractor._parse_json_response(response)
        assert len(lesions) == 1


class TestExtractionPrompts:
    """Test that extraction methods have proper prompt structure."""

    def test_extract_first_timepoint_callable(self):
        """Test that extract_first_timepoint method exists and is callable."""
        extractor = LesionExtractor()
        assert callable(extractor.extract_first_timepoint)

    def test_extract_followup_timepoint_callable(self):
        """Test that extract_followup_timepoint method exists and is callable."""
        extractor = LesionExtractor()
        assert callable(extractor.extract_followup_timepoint)


class TestExtractionResultHandling:
    """Test extraction result handling."""

    def test_extraction_result_structure(self):
        """Test that ExtractionResult has the expected structure."""
        result = ExtractionResult(
            lesions=[{"test": "data"}],
            raw_response="test response",
            success=True
        )
        assert hasattr(result, 'lesions')
        assert hasattr(result, 'raw_response')
        assert hasattr(result, 'success')
        assert hasattr(result, 'error_message')

    def test_failed_extraction_result(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            lesions=[],
            raw_response="",
            success=False,
            error_message="Test error"
        )
        assert result.success is False
        assert result.error_message == "Test error"
        assert len(result.lesions) == 0


# Note: Integration tests that actually call Ollama should be in a separate
# test file or marked with pytest.mark.integration and skipped by default
# since they require Ollama to be running.
