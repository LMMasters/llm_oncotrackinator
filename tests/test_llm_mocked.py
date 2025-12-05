"""
Tests for LLM calls using mocking strategies.

This file demonstrates different approaches to testing LLM interactions
without requiring Ollama to be running.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from llm_oncotrackinator import Config, LesionExtractor, LesionTracker, MedicalReport
from llm_oncotrackinator.models import ExtractionResult


class TestLLMCallsMocked:
    """Test LLM calls with mocked responses."""

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_first_timepoint_success(self, mock_chat):
        """Test first timepoint extraction with mocked successful response."""
        # Mock the Ollama response
        mock_chat.return_value = {
            "message": {
                "content": '[{"location": "right upper lobe", "size_cm": 2.3, "size_mm": 23.0, "characteristics": "nodular", "raw_text": "2.3 cm nodule in right upper lobe"}]'
            }
        }

        extractor = LesionExtractor(config=Config(temperature=0.0))
        result = extractor.extract_first_timepoint(
            "CT scan shows a 2.3 cm nodule in the right upper lobe."
        )

        # Verify result
        assert result.success is True
        assert len(result.lesions) == 1
        assert result.lesions[0]["location"] == "right upper lobe"
        assert result.lesions[0]["size_cm"] == 2.3

        # Verify the LLM was called correctly
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args.kwargs["model"] == "llama3.1:8b"
        assert call_args.kwargs["options"]["temperature"] == 0.0
        assert len(call_args.kwargs["messages"]) == 2  # system + user

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_with_markdown_wrapped_response(self, mock_chat):
        """Test extraction when LLM returns JSON wrapped in markdown."""
        mock_chat.return_value = {
            "message": {
                "content": '```json\n[{"location": "liver", "size_cm": 4.5}]\n```'
            }
        }

        extractor = LesionExtractor()
        result = extractor.extract_first_timepoint("Liver mass 4.5 cm")

        assert result.success is True
        assert len(result.lesions) == 1
        assert result.lesions[0]["location"] == "liver"

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_with_empty_lesions(self, mock_chat):
        """Test extraction when no lesions are found."""
        mock_chat.return_value = {
            "message": {
                "content": '[]'
            }
        }

        extractor = LesionExtractor()
        result = extractor.extract_first_timepoint("No significant findings.")

        assert result.success is True
        assert len(result.lesions) == 0

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_with_invalid_json(self, mock_chat):
        """Test extraction when LLM returns invalid JSON."""
        mock_chat.return_value = {
            "message": {
                "content": 'This is not valid JSON'
            }
        }

        extractor = LesionExtractor(config=Config(max_retries=1))
        result = extractor.extract_first_timepoint("Test report")

        assert result.success is False
        assert result.error_message is not None
        assert "Failed to parse JSON" in result.error_message

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_with_connection_error(self, mock_chat):
        """Test extraction when Ollama connection fails."""
        mock_chat.side_effect = ConnectionError("Could not connect to Ollama")

        extractor = LesionExtractor(config=Config(max_retries=2))
        result = extractor.extract_first_timepoint("Test report")

        assert result.success is False
        assert "Could not connect to Ollama" in result.error_message
        # Should retry max_retries times
        assert mock_chat.call_count == 2

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_followup_with_tracking(self, mock_chat):
        """Test follow-up extraction maintains lesion IDs."""
        mock_chat.return_value = {
            "message": {
                "content": '[{"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.8}]'
            }
        }

        extractor = LesionExtractor()
        previous_lesions = [
            {"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.3}
        ]
        result = extractor.extract_followup_timepoint(
            "Follow-up shows the nodule has increased to 2.8 cm.",
            previous_lesions
        )

        assert result.success is True
        assert len(result.lesions) == 1
        assert result.lesions[0]["lesion_id"] == "L1"
        assert result.lesions[0]["size_cm"] == 2.8

        # Verify previous lesions were included in the prompt
        call_args = mock_chat.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "L1" in user_message
        assert "2.3" in user_message

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_extract_with_retry_success_on_second_attempt(self, mock_chat):
        """Test that retry logic works correctly."""
        # First call fails, second succeeds
        mock_chat.side_effect = [
            Exception("Temporary error"),
            {
                "message": {
                    "content": '[{"location": "lung", "size_cm": 2.0}]'
                }
            }
        ]

        extractor = LesionExtractor(config=Config(max_retries=3))
        result = extractor.extract_first_timepoint("Test report")

        assert result.success is True
        assert len(result.lesions) == 1
        assert mock_chat.call_count == 2


class TestEndToEndWithMockedLLM:
    """Test the complete pipeline with mocked LLM calls."""

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_track_patient_full_pipeline(self, mock_chat):
        """Test tracking a patient through multiple timepoints with mocked LLM."""
        # Mock responses for each timepoint
        mock_chat.side_effect = [
            # First timepoint
            {
                "message": {
                    "content": '[{"location": "right upper lobe", "size_cm": 2.3}]'
                }
            },
            # Second timepoint
            {
                "message": {
                    "content": '[{"lesion_id": "L1", "location": "right upper lobe", "size_cm": 2.8}]'
                }
            },
            # Third timepoint
            {
                "message": {
                    "content": '[{"lesion_id": "L1", "location": "right upper lobe", "size_cm": 3.2}, {"lesion_id": "L2", "location": "left lower lobe", "size_cm": 1.5}]'
                }
            }
        ]

        tracker = LesionTracker(config=Config())
        reports = [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="CT scan shows a 2.3 cm nodule in the right upper lobe."
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Follow-up CT shows the nodule has increased to 2.8 cm."
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 6, 10),
                report_text="CT shows further growth to 3.2 cm. New 1.5 cm nodule in left lower lobe."
            )
        ]

        history = tracker.track_patient("P001", reports)

        # Verify results
        assert history.patient_id == "P001"
        assert len(history.timepoints) == 3

        # Check first timepoint
        assert len(history.timepoints[0].lesions) == 1
        assert history.timepoints[0].lesions[0].lesion_id == "L1"
        assert history.timepoints[0].lesions[0].size_cm == 2.3

        # Check second timepoint
        assert len(history.timepoints[1].lesions) == 1
        assert history.timepoints[1].lesions[0].lesion_id == "L1"
        assert history.timepoints[1].lesions[0].size_cm == 2.8

        # Check third timepoint
        assert len(history.timepoints[2].lesions) == 2

        # Verify lesion tracking
        lesion_ids = history.get_all_lesion_ids()
        assert len(lesion_ids) == 2
        assert "L1" in lesion_ids
        assert "L2" in lesion_ids

        # Verify L1 progression
        l1_timeline = history.get_lesion_timeline("L1")
        assert len(l1_timeline) == 3
        assert l1_timeline[0].size_cm == 2.3
        assert l1_timeline[1].size_cm == 2.8
        assert l1_timeline[2].size_cm == 3.2

        # Verify LLM was called 3 times
        assert mock_chat.call_count == 3


class TestPromptContent:
    """Test that prompts contain the expected information."""

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_first_timepoint_prompt_structure(self, mock_chat):
        """Test that first timepoint prompt has correct structure."""
        mock_chat.return_value = {
            "message": {"content": "[]"}
        }

        extractor = LesionExtractor()
        report_text = "Test medical report with specific content"
        extractor.extract_first_timepoint(report_text)

        # Get the messages that were sent
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # System message should mention key concepts
        system_msg = messages[0]["content"]
        assert "lesion" in system_msg.lower()
        assert "json" in system_msg.lower()
        assert "location" in system_msg.lower()

        # User message should contain the report
        user_msg = messages[1]["content"]
        assert report_text in user_msg

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_followup_prompt_includes_previous_lesions(self, mock_chat):
        """Test that follow-up prompt includes previous lesion information."""
        mock_chat.return_value = {
            "message": {"content": "[]"}
        }

        extractor = LesionExtractor()
        previous_lesions = [
            {"lesion_id": "L1", "location": "lung", "size_cm": 2.3},
            {"lesion_id": "L2", "location": "liver", "size_cm": 4.5}
        ]
        report_text = "Follow-up report"

        extractor.extract_followup_timepoint(report_text, previous_lesions)

        call_args = mock_chat.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]

        # Should include previous lesion information
        assert "L1" in user_msg
        assert "L2" in user_msg
        assert "lung" in user_msg
        assert "liver" in user_msg
        assert report_text in user_msg


class TestLLMConfiguration:
    """Test that LLM configuration is correctly applied."""

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_custom_model_is_used(self, mock_chat):
        """Test that custom model configuration is applied."""
        mock_chat.return_value = {
            "message": {"content": "[]"}
        }

        config = Config(ollama_model="custom-model:latest", temperature=0.5)
        extractor = LesionExtractor(config=config)
        extractor.extract_first_timepoint("Test")

        call_args = mock_chat.call_args
        assert call_args.kwargs["model"] == "custom-model:latest"
        assert call_args.kwargs["options"]["temperature"] == 0.5

    @patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
    def test_temperature_affects_determinism(self, mock_chat):
        """Test that temperature setting is passed to Ollama."""
        mock_chat.return_value = {
            "message": {"content": "[]"}
        }

        # Test with deterministic (temp=0)
        config = Config(temperature=0.0)
        extractor = LesionExtractor(config=config)
        extractor.extract_first_timepoint("Test")

        call_args = mock_chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.0
