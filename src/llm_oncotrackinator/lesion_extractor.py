"""
LLM-based lesion extraction from medical reports.
"""

import json
import time
from typing import List, Optional, Dict, Any

import ollama

from llm_oncotrackinator.config import Config
from llm_oncotrackinator.models import ExtractionResult


class LesionExtractor:
    """Extract lesion information from medical reports using LLM."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LesionExtractor.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()

    def extract_first_timepoint(self, report_text: str) -> ExtractionResult:
        """
        Extract lesions from the first timepoint report.

        Args:
            report_text: The medical report text

        Returns:
            ExtractionResult containing extracted lesions

        Raises:
            Exception: If extraction fails after all retries
        """
        system_prompt = """You are a medical AI assistant specialized in extracting structured lesion information from radiology reports.

Your task is to extract ALL lesions mentioned in the report and return them as a JSON array.

For each lesion, extract:
- location: anatomical location (e.g., "right upper lobe", "liver segment 7", "left frontal lobe")
- size: the size with unit (extract as both size_mm and size_cm if possible)
- characteristics: any additional descriptors (e.g., "enhancing", "nodular")
- raw_text: the exact phrase from the report describing this lesion

Return ONLY a valid JSON array of lesions, with no additional text or explanation.

Example format:
[
  {
    "location": "right upper lobe",
    "size_cm": 2.3,
    "size_mm": 23.0,
    "characteristics": "nodule",
    "raw_text": "2.3 cm nodule in the right upper lobe"
  }
]

If no lesions are found, return an empty array: []
"""

        user_prompt = f"""Extract all lesions from this medical report:

{report_text}

Return the lesions as a JSON array."""

        return self._extract_with_retry(system_prompt, user_prompt)

    def extract_followup_timepoint(
        self,
        report_text: str,
        previous_lesions: List[Dict[str, Any]]
    ) -> ExtractionResult:
        """
        Extract lesions from a follow-up timepoint, tracking previous lesions.

        Args:
            report_text: The medical report text
            previous_lesions: List of lesions from previous timepoints

        Returns:
            ExtractionResult containing extracted lesions with tracking

        Raises:
            Exception: If extraction fails after all retries
        """
        system_prompt = """You are a medical AI assistant specialized in tracking lesions across multiple radiology reports.

Your task is to extract ALL lesions from the current report and match them with previously tracked lesions when possible.

For each lesion in the current report:
1. If it appears to be the same lesion as a previous one (same or similar location), use the same lesion_id
2. If it's a new lesion, assign it a new lesion_id (e.g., "L5", "L6", etc.)

Extract for each lesion:
- lesion_id: identifier matching previous timepoints or new ID for new lesions
- location: anatomical location
- size: the size with unit (extract as both size_mm and size_cm if possible)
- characteristics: any additional descriptors
- raw_text: the exact phrase from the report

Return ONLY a valid JSON array of lesions, with no additional text or explanation.

Example format:
[
  {
    "lesion_id": "L1",
    "location": "right upper lobe",
    "size_cm": 2.8,
    "size_mm": 28.0,
    "characteristics": "nodule, increased",
    "raw_text": "right upper lobe nodule has increased to 2.8 cm"
  }
]
"""

        previous_lesions_json = json.dumps(previous_lesions, indent=2)

        user_prompt = f"""Here are the previously tracked lesions:

{previous_lesions_json}

Now extract all lesions from this follow-up report, maintaining lesion_id for tracked lesions:

{report_text}

Return the lesions as a JSON array."""

        return self._extract_with_retry(system_prompt, user_prompt)

    def _extract_with_retry(self, system_prompt: str, user_prompt: str) -> ExtractionResult:
        """
        Call LLM with retry logic.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM

        Returns:
            ExtractionResult with extracted data

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = ollama.chat(
                    model=self.config.ollama_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": self.config.temperature
                    }
                )

                raw_response = response["message"]["content"].strip()

                # Try to parse JSON response
                lesions = self._parse_json_response(raw_response)

                return ExtractionResult(
                    lesions=lesions,
                    raw_response=raw_response,
                    success=True
                )

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue

        # All retries failed
        return ExtractionResult(
            lesions=[],
            raw_response="",
            success=False,
            error_message=f"Extraction failed after {self.config.max_retries} attempts: {str(last_error)}"
        )

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response text

        Returns:
            List of lesion dictionaries

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Parse JSON
        try:
            lesions = json.loads(response)
            if not isinstance(lesions, list):
                raise ValueError("Response is not a JSON array")
            return lesions
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}\nResponse: {response[:200]}")
