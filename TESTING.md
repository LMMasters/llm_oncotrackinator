# Testing Guide

This document explains the testing strategy for LLM OncoTrackinator and how to run different types of tests.

## Test Organization

### Test Files

- **test_data_loader.py** - Data loading and validation tests
- **test_models.py** - Pydantic model tests
- **test_lesion_extractor.py** - LLM extractor tests (without actual LLM calls)
- **test_tracker.py** - Lesion tracking logic tests (mocked)
- **test_output.py** - Output generation tests
- **test_integration.py** - Pipeline integration tests
- **test_llm_mocked.py** - LLM tests with fully mocked responses
- **test_llm_real.py** - Real LLM integration tests (requires Ollama)

### Test Categories

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions (marked with `@pytest.mark.integration`)
3. **Mocked LLM Tests** - Test LLM interaction logic without calling Ollama
4. **Real LLM Tests** - Test actual LLM quality (requires Ollama running)

## Running Tests

### Run All Tests (Excluding Integration)

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=src/llm_oncotrackinator --cov-report=term-missing
```

### Run Specific Test File

```bash
pytest tests/test_models.py -v
```

### Run Specific Test Class or Function

```bash
# Run a specific class
pytest tests/test_models.py::TestLesion -v

# Run a specific test
pytest tests/test_models.py::TestLesion::test_create_lesion_with_all_fields -v
```

### Run Integration Tests (Requires Ollama)

```bash
# Run all integration tests
pytest tests/ -v -m integration

# Run only real LLM tests
pytest tests/test_llm_real.py -v -s
```

**Note:** Real LLM tests are skipped by default. To enable them:
1. Ensure Ollama is running (`ollama serve`)
2. Pull a model (`ollama pull llama3.1:8b`)
3. Remove the `skipif` decorator in test_llm_real.py

### Run Tests Excluding Integration

```bash
pytest tests/ -v -m "not integration"
```

## Testing LLM Calls

We use **three strategies** for testing LLM interactions:

### 1. Mocked Tests (Default)

These tests mock the `ollama.chat` function to test logic without calling the actual LLM.

**Advantages:**
- Fast execution
- No Ollama dependency
- Deterministic results
- Test error handling

**Example:**
```python
@patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
def test_extract_first_timepoint(self, mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": '[{"location": "lung", "size_cm": 2.3}]'
        }
    }

    extractor = LesionExtractor()
    result = extractor.extract_first_timepoint("Test report")

    assert result.success
    assert len(result.lesions) == 1
```

**Use Cases:**
- Testing retry logic
- Testing error handling
- Testing JSON parsing
- Testing prompt construction
- CI/CD pipelines

### 2. Integration Tests with Mocked LLM

These tests verify that components work together correctly, but still mock the LLM.

**Example from test_llm_mocked.py:**
```python
@patch('llm_oncotrackinator.lesion_extractor.ollama.chat')
def test_track_patient_full_pipeline(self, mock_chat):
    # Mock responses for each timepoint
    mock_chat.side_effect = [
        {"message": {"content": '[...]'}},  # First timepoint
        {"message": {"content": '[...]'}},  # Second timepoint
        {"message": {"content": '[...]'}},  # Third timepoint
    ]

    tracker = LesionTracker()
    history = tracker.track_patient("P001", reports)

    # Verify complete tracking
    assert len(history.timepoints) == 3
```

**Use Cases:**
- Testing complete workflows
- Verifying data flow between components
- Testing state management

### 3. Real LLM Tests (Manual)

These tests actually call Ollama and verify extraction quality.

**Example from test_llm_real.py:**
```python
@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires Ollama")
def test_extract_single_clear_lesion(self, extractor):
    report = "CT scan shows a 2.3 cm nodule in the right upper lobe."

    result = extractor.extract_first_timepoint(report)

    assert result.success
    assert len(result.lesions) >= 1
    assert result.lesions[0]["size_cm"] ≈ 2.3
```

**Use Cases:**
- Validating LLM prompt quality
- Testing with real medical reports
- Evaluating extraction accuracy
- Tuning temperature and other parameters
- Comparing different models

**To enable real LLM tests:**
1. Start Ollama: `ollama serve`
2. Pull model: `ollama pull llama3.1:8b`
3. Edit `test_llm_real.py` and remove `@pytest.mark.skipif`
4. Run: `pytest tests/test_llm_real.py -v -s`

## Test Coverage

Current coverage: **88%**

```
Name                                          Coverage
--------------------------------------------------------
src/llm_oncotrackinator/__init__.py           100%
src/llm_oncotrackinator/config.py             100%
src/llm_oncotrackinator/models.py             100%
src/llm_oncotrackinator/output.py             100%
src/llm_oncotrackinator/tracker.py             96%
src/llm_oncotrackinator/data_loader.py         81%
src/llm_oncotrackinator/lesion_extractor.py    58%
```

**Note:** lesion_extractor.py has lower coverage because actual LLM calls aren't tested in the default test suite.

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run all tests except integration
pytest tests/ -v -m "not integration" --cov=src/llm_oncotrackinator

# Or with stricter settings
pytest tests/ -v -m "not integration" --cov=src/llm_oncotrackinator --cov-fail-under=80
```

## Writing New Tests

### Testing New LLM Functionality

When adding new LLM-dependent code:

1. **Write mocked tests first** (in test_llm_mocked.py)
   - Test the logic independently
   - Test error cases
   - Fast and deterministic

2. **Write real LLM tests** (in test_llm_real.py)
   - Mark with `@pytest.mark.integration`
   - Add `skipif` decorator
   - Document how to enable

### Best Practices

1. **Use fixtures** (in conftest.py) for common test data
2. **Mock external dependencies** for unit tests
3. **Test error paths** as well as happy paths
4. **Use descriptive test names** that explain what is being tested
5. **Keep tests focused** - one concept per test
6. **Document integration test requirements**

### Example Test Structure

```python
class TestNewFeature:
    """Test suite for new feature."""

    def test_basic_functionality(self):
        """Test that basic case works."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = process(input_data)

        # Assert
        assert result.is_valid()

    def test_edge_case(self):
        """Test handling of edge case."""
        # Test edge case
        pass

    @patch('module.external_call')
    def test_with_mock(self, mock_call):
        """Test with mocked external dependency."""
        mock_call.return_value = expected_value
        # Test using mock
        pass
```

## Troubleshooting

### Tests Fail with "Connection Error"

This usually means:
- Ollama is not running
- You're running integration tests without Ollama

**Solution:** Either install/start Ollama, or exclude integration tests:
```bash
pytest tests/ -m "not integration"
```

### "Unknown pytest.mark.integration" Warning

This is expected if pytest.mark.integration is not registered. We've added it to pyproject.toml, but you can ignore this warning.

### Flaky LLM Tests

Real LLM tests can be non-deterministic even with temperature=0. This is normal. For CI/CD, always use mocked tests.

### Slow Test Suite

If tests are slow, you're probably running integration tests. Use:
```bash
pytest tests/ -m "not integration"
```

## Future Testing Improvements

- [ ] Add property-based testing with Hypothesis
- [ ] Add performance benchmarks
- [ ] Add fuzzing for LLM prompt injection
- [ ] Create golden test datasets with known extractions
- [ ] Add visual regression tests for output formats
- [ ] Implement LLM response caching for faster development

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Ollama Documentation](https://github.com/ollama/ollama)
