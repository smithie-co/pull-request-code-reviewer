import pytest
from unittest import mock
import logging
import json # Added for test data

from src.analysis_service import AnalysisService
from src.bedrock_handler import BedrockHandler # Needed for type hinting if not already imported
from src import config

@pytest.fixture
def mock_bedrock_handler():
    """Returns a mock BedrockHandler."""
    return mock.MagicMock()

@pytest.fixture
def mock_config_model_ids(monkeypatch):
    """Mocks the model IDs in config for tests."""
    monkeypatch.setattr(config, 'HEAVY_MODEL_ID', "test_heavy_model")
    monkeypatch.setattr(config, 'LIGHT_MODEL_ID', "test_light_model")
    monkeypatch.setattr(config, 'DEEPSEEK_MODEL_ID', "test_deepseek_model")

@pytest.fixture
def analysis_service(mock_bedrock_handler, mock_config_model_ids):
    """Returns an AnalysisService with mocked dependencies."""
    return AnalysisService(mock_bedrock_handler)

@pytest.fixture
def sample_diff_content():
    """Returns sample diff content for testing."""
    return "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n def hello():\n-    print('world')\n+    print('universe')\n"

# --- analyze_code_changes Tests ---

def test_analyze_code_changes_success(analysis_service, mock_bedrock_handler):
    """Test successful code analysis."""
    diff = "some diff content"
    expected_analysis = "Heavy model analysis result"
    mock_bedrock_handler.invoke_model.return_value = expected_analysis

    result = analysis_service.analyze_code_changes(diff)

    assert result == expected_analysis
    mock_bedrock_handler.invoke_model.assert_called_once_with(
        model_id="test_heavy_model",
        prompt=mock.ANY, # Prompt construction is complex, check basics if needed
        analysis_type='heavy_analysis',  # Now uses dynamic token calculation
        temperature=0.5
    )
    # Check if prompt contains the diff
    assert diff in mock_bedrock_handler.invoke_model.call_args.kwargs['prompt']

def test_analyze_code_changes_empty_diff(analysis_service):
    """Test ValueError if diff content is empty."""
    with pytest.raises(ValueError, match="Diff content cannot be empty for analysis."):
        analysis_service.analyze_code_changes("")

def test_analyze_code_changes_no_model_id(analysis_service, monkeypatch):
    """Test ValueError if HEAVY_MODEL_ID is not configured."""
    monkeypatch.setattr(config, 'HEAVY_MODEL_ID', None)
    with pytest.raises(ValueError, match="HEAVY_MODEL_ID is not configured"):
        analysis_service.analyze_code_changes("some diff")

def test_analyze_code_changes_model_invocation_error(analysis_service, mock_bedrock_handler):
    """Test RuntimeError if BedrockHandler raises an error."""
    mock_bedrock_handler.invoke_model.side_effect = RuntimeError("Bedrock API error")
    with pytest.raises(RuntimeError, match="Heavy model analysis failed: Bedrock API error"):
        analysis_service.analyze_code_changes("some diff")

# --- analyze_heavy_model_output Tests (Updated and New) ---

def test_analyze_heavy_model_output_success_structured_json(analysis_service, mock_bedrock_handler, sample_diff_content):
    """Test successful extraction of structured JSON from DeepSeek model output."""
    heavy_output = "Raw analysis from heavy model, mentioning file1.py and line 2."
    valid_json_payload = [
        {"file_path": "file1.py", "line": 2, "suggestion": "Consider refactoring this line."},
        {"file_path": "file1.py", "line": 3, "suggestion": "Add a comment here."}
    ]
    # Simulate model returning JSON string, possibly with some surrounding text
    mock_bedrock_handler.invoke_model.return_value = f"Some introductory text.\n{json.dumps(valid_json_payload)}\nSome concluding text."

    result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)

    assert result == valid_json_payload
    mock_bedrock_handler.invoke_model.assert_called_once()
    args, kwargs = mock_bedrock_handler.invoke_model.call_args
    assert kwargs["model_id"] == "test_deepseek_model"
    assert heavy_output in kwargs["prompt"]
    assert sample_diff_content in kwargs["prompt"]
    assert "JSON Output:" in kwargs["prompt"]

def test_analyze_heavy_model_output_empty_json_array(analysis_service, mock_bedrock_handler, sample_diff_content):
    """Test handling when the model returns an empty JSON array."""
    heavy_output = "Raw analysis, but no specific line items found."
    mock_bedrock_handler.invoke_model.return_value = "[]" # Model returns an empty array

    result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)
    assert result == []

def test_analyze_heavy_model_output_malformed_json(analysis_service, mock_bedrock_handler, sample_diff_content, caplog):
    """Test handling of malformed JSON from the model - specifically when extraction fails after initial full parse fails."""
    heavy_output = "Some analysis."
    # This input will cause the initial json.loads to fail, then find will attempt to find '[' and ']'
    # Since ']' is not found after '[', it will hit the "Could not find or parse JSON array" warning path.
    mock_bedrock_handler.invoke_model.return_value = "Not a valid JSON string {["

    # The current input will trigger the warning about not finding a JSON array after initial parse fails
    with caplog.at_level(logging.WARNING): # Expect a WARNING
        result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)
    
    assert result == []
    # Check for the specific WARNING message
    assert "Could not find valid JSON array delimiters '[' and ']'" in caplog.text

def test_analyze_heavy_model_output_not_a_list(analysis_service, mock_bedrock_handler, sample_diff_content, caplog):
    """Test handling if parsed JSON is not a list."""
    heavy_output = "Some analysis."
    mock_bedrock_handler.invoke_model.return_value = json.dumps({"key": "value"}) # Returns a dict, not list
    
    with caplog.at_level(logging.WARNING):
        result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)
    
    assert result == []
    assert "Parsed JSON output is not a list as expected" in caplog.text

def test_analyze_heavy_model_output_items_missing_keys(analysis_service, mock_bedrock_handler, sample_diff_content, caplog):
    """Test filtering of items with missing keys."""
    heavy_output = "Analysis with partial items."
    suggestions_payload = [
        {"file_path": "file1.py", "line": 10, "suggestion": "Good one"},
        {"file_path": "file2.py", "suggestion": "Missing line"}, # Missing 'line'
        {"line": 20, "suggestion": "Missing path"}, # Missing 'file_path'
    ]
    mock_bedrock_handler.invoke_model.return_value = json.dumps(suggestions_payload)

    with caplog.at_level(logging.WARNING):
        result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)
    
    assert len(result) == 1
    assert result[0] == {"file_path": "file1.py", "line": 10, "suggestion": "Good one"}
    assert "Skipping malformed suggestion item" in caplog.text
    assert "{'file_path': 'file2.py', 'suggestion': 'Missing line'}" in caplog.text

def test_analyze_heavy_model_output_items_wrong_types(analysis_service, mock_bedrock_handler, sample_diff_content, caplog):
    """Test filtering of items with wrong data types."""
    heavy_output = "Analysis with type issues."
    suggestions_payload = [
        {"file_path": "file1.py", "line": 10, "suggestion": "Good one"},
        {"file_path": 123, "line": 20, "suggestion": "Path is int"}, # file_path is int
        {"file_path": "file2.py", "line": "30", "suggestion": "Line is str"}, # line is str
    ]
    mock_bedrock_handler.invoke_model.return_value = json.dumps(suggestions_payload)

    with caplog.at_level(logging.WARNING):
        result = analysis_service.analyze_heavy_model_output(heavy_output, diff_content=sample_diff_content)
    
    assert len(result) == 1
    assert result[0] == {"file_path": "file1.py", "line": 10, "suggestion": "Good one"}
    assert "Skipping malformed suggestion item" in caplog.text
    assert "{'file_path': 123," in caplog.text

def test_analyze_heavy_model_output_empty_input(analysis_service, caplog):
    """Test new behavior for empty heavy_model_output (returns empty list, logs warning)."""
    with caplog.at_level(logging.WARNING):
        result = analysis_service.analyze_heavy_model_output("", diff_content="some diff")
    assert result == []
    assert "Heavy model output is empty. Cannot perform refined analysis." in caplog.text

def test_analyze_heavy_model_output_no_model_id(analysis_service, monkeypatch, sample_diff_content):
    """Test ValueError if DEEPSEEK_MODEL_ID is not configured."""
    monkeypatch.setattr(config, 'DEEPSEEK_MODEL_ID', None)
    with pytest.raises(ValueError, match="DEEPSEEK_MODEL_ID is not configured"):
        analysis_service.analyze_heavy_model_output("some output", diff_content=sample_diff_content)

def test_analyze_heavy_model_output_invocation_error(analysis_service, mock_bedrock_handler, sample_diff_content, caplog):
    """Test that it returns empty list on BedrockHandler error and logs error."""
    mock_bedrock_handler.invoke_model.side_effect = RuntimeError("Bedrock API error")
    with caplog.at_level(logging.ERROR):
        result = analysis_service.analyze_heavy_model_output("some output", diff_content=sample_diff_content)
    assert result == []
    assert "Error during structured suggestion extraction" in caplog.text
    assert "Bedrock API error" in caplog.text

# --- summarize_changes Tests ---

def test_summarize_changes_success(analysis_service, mock_bedrock_handler):
    """Test successful change summarization."""
    diff = "some diff content for summary"
    expected_summary = "Light model summary"
    mock_bedrock_handler.invoke_model.return_value = expected_summary

    result = analysis_service.summarize_changes(diff)

    assert result == expected_summary
    mock_bedrock_handler.invoke_model.assert_called_once_with(        
        model_id="test_light_model",        
        prompt=mock.ANY,        
        analysis_type='summary',  # Now uses dynamic token calculation        
        temperature=0.7    
    )
    assert diff in mock_bedrock_handler.invoke_model.call_args.kwargs['prompt']

def test_summarize_changes_empty_diff(analysis_service):
    """Test ValueError if diff is empty for summarization."""
    with pytest.raises(ValueError, match="Diff content cannot be empty for summarization."):
        analysis_service.summarize_changes("")

def test_summarize_changes_no_model_id(analysis_service, monkeypatch):
    """Test ValueError if LIGHT_MODEL_ID is not configured."""
    monkeypatch.setattr(config, 'LIGHT_MODEL_ID', None)
    with pytest.raises(ValueError, match="LIGHT_MODEL_ID is not configured"):
        analysis_service.summarize_changes("some diff")

# --- generate_review_body Tests ---

def test_generate_review_body_with_summary_and_refined_analysis(analysis_service):
    """Test review body generation with summary and refined analysis."""
    summary = "Changes summarized."
    refined = "Refined points."
    body = analysis_service.generate_review_body(summary, refined_analysis=refined)

    assert "## AI Code Review Summary 🤖" in body
    assert summary in body
    assert "### Detailed Analysis (Refined)" in body
    assert refined in body
    assert "*This review was auto-generated by an AI assistant.*" in body
    assert "Detailed Analysis (Raw)" not in body # Raw should not be there if refined is present

def test_generate_review_body_with_summary_and_raw_analysis(analysis_service):
    """Test review body generation with summary and raw heavy analysis (refined is None)."""
    summary = "Changes summarized."
    raw_heavy = "Raw heavy points."
    body = analysis_service.generate_review_body(summary, refined_analysis=None, heavy_analysis_raw=raw_heavy)

    assert summary in body
    assert "### Detailed Analysis (Raw)" in body
    assert raw_heavy in body
    assert "Detailed Analysis (Refined)" not in body

def test_generate_review_body_with_summary_only(analysis_service):
    """Test review body with only summary if no detailed analysis is provided."""
    summary = "Just a summary."
    body = analysis_service.generate_review_body(summary)
    assert summary in body
    assert "No detailed analysis was performed" in body

def test_generate_review_body_empty_summary(analysis_service):
    """Test review body generation handles an empty summary gracefully."""
    # This scenario implies summarize_changes might have failed or returned empty
    with mock.patch.object(logging.getLogger('src.analysis_service'), 'warning') as mock_logger_warning:
        body = analysis_service.generate_review_body("")
        assert "No summary could be generated" in body
        mock_logger_warning.assert_called_once_with("generate_review_body called with an empty summary.")

def test_model_id_override(analysis_service, mock_bedrock_handler, sample_diff_content):
    """Test that model_id_override is correctly used."""
    diff = "diff content"
    custom_heavy_model = "custom_heavy"
    custom_light_model = "custom_light"
    custom_deepseek_model = "custom_deepseek"

    # Test for analyze_code_changes
    mock_bedrock_handler.invoke_model.reset_mock()
    analysis_service.analyze_code_changes(diff, model_id_override=custom_heavy_model)
    mock_bedrock_handler.invoke_model.assert_called_once_with(
        model_id=custom_heavy_model, 
        prompt=mock.ANY, 
        analysis_type='heavy_analysis',
        temperature=0.5
    )

    # Test for summarize_changes
    mock_bedrock_handler.invoke_model.reset_mock()
    analysis_service.summarize_changes(diff, model_id_override=custom_light_model)
    mock_bedrock_handler.invoke_model.assert_called_once_with(
        model_id=custom_light_model, 
        prompt=mock.ANY, 
        analysis_type='summary',
        temperature=0.7
    )

    # Test for analyze_heavy_model_output
    mock_bedrock_handler.invoke_model.reset_mock()
    # Add sample_diff_content to the call as it's now a required argument
    analysis_service.analyze_heavy_model_output("heavy output", diff_content=sample_diff_content, model_id_override=custom_deepseek_model)
    mock_bedrock_handler.invoke_model.assert_called_once_with(
        model_id=custom_deepseek_model, 
        prompt=mock.ANY, 
        analysis_type='structured_extraction',
        temperature=0.3
    ) 