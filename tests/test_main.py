import pytest
import json
import os
import sys
import logging
from unittest import mock

# Import main function and other necessary components
from src.main import main
from src import config
from src.bedrock_handler import BedrockHandler
from src.github_handler import GithubHandler, BOT_SIGNATURE
from src.analysis_service import AnalysisService

# --- Fixtures ---

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks all necessary environment variables for a successful run."""
    monkeypatch.setenv("GITHUB_TOKEN", "test_gh_token")
    # Use a fixed path for the test_event.json that the mock_event_file fixture will use
    # This path will be set in the environment and then used by the fixture to create the file.
    test_event_json_path = "test_event.json" 
    monkeypatch.setenv("GITHUB_EVENT_PATH", test_event_json_path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_aws_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_aws_secret_key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("HEAVY_MODEL_ID", "heavy_model")
    monkeypatch.setenv("LIGHT_MODEL_ID", "light_model")
    monkeypatch.setenv("DEEPSEEK_MODEL_ID", "deepseek_model")
    monkeypatch.setenv("BOT_NAME", "test-ai-bot") # For review dismissal logic
    # Ensure config reloads these by clearing its cached values if any, or re-importing
    # For this test structure, simple setattr on config module is often enough if config loads on import.
    # Or, ensure BedrockHandler/etc., pick up from os.environ if they call getenv directly in tests.

@pytest.fixture
def mock_event_file(mock_env_vars):
    """Creates a mock GITHUB_EVENT_PATH file with PR data."""
    event_data = {
        "pull_request": {"number": 123, "head": {"sha": "newcommitsha"}}, # Changed SHA for dismissal test
        "repository": {"full_name": "owner/repo"},
        "action": "synchronize" # Action that implies new commits
    }
    # Get the path from the environment variable that mock_env_vars should have set.
    # This ensures we use the path that the application code (main.py via config.py) will also see.
    event_file_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_file_path:
        # This should not happen if mock_env_vars ran correctly and set GITHUB_EVENT_PATH
        raise ValueError("GITHUB_EVENT_PATH not set in environment for mock_event_file")

    with open(event_file_path, 'w') as f:
        json.dump(event_data, f)
    yield event_file_path
    os.remove(event_file_path) # Cleanup

@pytest.fixture
def mock_handlers(monkeypatch):
    """Mocks BedrockHandler, GithubHandler, and AnalysisService instances and their key methods."""
    mock_bedrock = mock.MagicMock(spec=BedrockHandler)
    mock_github = mock.MagicMock(spec=GithubHandler)
    # Set repo_name on the mock_github instance, as main.py expects it.
    # This mirrors what the actual GithubHandler.__init__ would do.
    mock_github.repo_name = "owner/repo" 
    mock_analysis = mock.MagicMock(spec=AnalysisService)

    # Mock constructors to return our instances
    monkeypatch.setattr('src.main.BedrockHandler', lambda: mock_bedrock)
    monkeypatch.setattr('src.main.GithubHandler', lambda github_token, repo_name: mock_github)
    monkeypatch.setattr('src.main.AnalysisService', lambda bedrock_handler: mock_analysis)

    # Setup default return values for handler methods
    mock_github.get_pr_diff.return_value = "sample diff content"
    # Mock _get_pull_request_obj directly if main calls it (it does for head_commit_sha)
    mock_pr_obj = mock.MagicMock()
    mock_pr_obj.head.sha = "newcommitsha" # Matches event file
    mock_github._get_pull_request_obj.return_value = mock_pr_obj # Mocking the private method
    
    mock_analysis.summarize_changes.return_value = "Mocked summary."
    mock_analysis.analyze_code_changes.return_value = "Mocked heavy analysis."
    mock_analysis.analyze_heavy_model_output.return_value = [
        {"file_path": "src/main.py", "line": 10, "suggestion": "Consider this change."},
        {"file_path": "README.md", "line": 5, "suggestion": "Update docs."}
    ]
    mock_analysis.generate_review_body.return_value = "### AI Review\nMocked PR review body."

    # GithubHandler mocks
    mock_github.get_pr_reviews.return_value = [] # Default: no previous reviews
    mock_github.get_last_bot_review.return_value = None # Default: no last bot review
    mock_github.dismiss_review.return_value = True # Default: dismissal succeeds
    mock_github.post_pr_review.return_value = mock.MagicMock(id=999) # Successful post

    return mock_bedrock, mock_github, mock_analysis


# --- Main Orchestration Tests ---

def test_main_successful_flow_with_line_comments(mock_env_vars, mock_event_file, mock_handlers):
    """Test the main flow with successful review and line-specific comments."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers

    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called()

    mock_github._get_pull_request_obj.assert_called_with(pr_number=123)
    mock_github.get_pr_reviews.assert_called_once_with(pr_number=123)
    # get_last_bot_review is called with the result of get_pr_reviews
    mock_github.get_last_bot_review.assert_called_once_with([]) # Default no prior reviews
    mock_github.dismiss_review.assert_not_called() # No old review to dismiss by default

    mock_github.get_pr_diff.assert_called_once_with(pr_number=123)
    
    mock_analysis.summarize_changes.assert_called_once_with(diff_content="sample diff content")
    mock_analysis.analyze_code_changes.assert_called_once_with(diff_content="sample diff content")
    mock_analysis.analyze_heavy_model_output.assert_called_once_with(
        heavy_model_output="Mocked heavy analysis.",
        diff_content="sample diff content" # Ensure diff_content is passed
    )
    
    # Expected line comments based on mock_analysis.analyze_heavy_model_output
    expected_line_comments_for_review = [
        {"path": "src/main.py", "line": 10, "body": "Consider this change."},
        {"path": "README.md", "line": 5, "body": "Update docs."}
    ]

    mock_analysis.generate_review_body.assert_called_once_with(
        summary="Mocked summary.",
        refined_analysis=None, # None because refined_analysis_results exist
        heavy_analysis_raw=None  # None because refined_analysis_results exist
    )
    mock_github.post_pr_review.assert_called_once_with(
        pr_number=123,
        review_body="### AI Review\nMocked PR review body.",
        commit_id="newcommitsha",
        event='COMMENT', # Default event in main.py for this path
        line_comments=expected_line_comments_for_review
    )

def test_main_dismisses_old_bot_review(mock_env_vars, mock_event_file, mock_handlers, monkeypatch):
    """Test that an old bot review is dismissed if new commits are present."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers
    monkeypatch.setattr(config, 'BOT_NAME', "test-ai-bot") # Ensure BOT_NAME is set

    mock_old_review = mock.MagicMock()
    mock_old_review.user.login = "test-ai-bot"
    mock_old_review.commit_id = "oldcommitsha"
    mock_old_review.id = 777
    mock_old_review.body = f"Old review {BOT_SIGNATURE}"

    mock_github.get_pr_reviews.return_value = [mock_old_review]
    mock_github.get_last_bot_review.return_value = mock_old_review # Make it return the old review
    
    # Event file PR head SHA is "newcommitsha" from mock_event_file fixture
    # GithubHandler._get_pull_request_obj.return_value.head.sha is also "newcommitsha"

    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called()
    
    mock_github.get_last_bot_review.assert_called_once_with([mock_old_review])
    mock_github.dismiss_review.assert_called_once()
    dismiss_args, dismiss_kwargs = mock_github.dismiss_review.call_args
    assert dismiss_args[0] == mock_old_review # Correct review object passed
    assert "Dismissing old review as new commits have been pushed" in dismiss_args[1]
    assert "newcommitsha" in dismiss_args[1]
    mock_github.post_pr_review.assert_called_once() # New review should still be posted

def test_main_no_line_comments_from_analysis(mock_env_vars, mock_event_file, mock_handlers):
    """Test main flow when analyze_heavy_model_output returns an empty list (no line comments)."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers
    mock_analysis.analyze_heavy_model_output.return_value = [] # No structured suggestions

    # If analyze_heavy_model_output is empty, generate_review_body should get raw heavy_analysis
    mock_analysis.generate_review_body.reset_mock() # Reset from fixture default
    
    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called()

    mock_analysis.analyze_heavy_model_output.assert_called_once_with(
        heavy_model_output="Mocked heavy analysis.",
        diff_content="sample diff content"
    )
    mock_analysis.generate_review_body.assert_called_once_with(
        summary="Mocked summary.",
        refined_analysis=None, # Still None, as this arg is for the old string-based refinement
        heavy_analysis_raw="Mocked heavy analysis." # Pass raw heavy if refined_analysis_results is empty
    )
    mock_github.post_pr_review.assert_called_once_with(
        pr_number=123,
        review_body=mock.ANY, # Body will be generated based on summary and raw heavy
        commit_id="newcommitsha",
        event='COMMENT',
        line_comments=[] # No line comments to post
    )

def test_main_action_not_triggering_full_review(mock_env_vars, mock_event_file, mock_handlers, caplog):
    """Test main flow for actions that don't typically trigger a full new review dismissal (e.g., 'edited')."""
    # Modify event_file to have an action like 'edited'
    event_data = {
        "pull_request": {"number": 123, "head": {"sha": "newcommitsha"}},
        "repository": {"full_name": "owner/repo"},
        "action": "edited" 
    }
    with open(os.getenv("GITHUB_EVENT_PATH"), 'w') as f:
        json.dump(event_data, f)

    mock_bedrock, mock_github, mock_analysis = mock_handlers
    # Simulate an old bot review being present, but it should NOT be dismissed for 'edited' action
    mock_old_review = mock.MagicMock()
    mock_old_review.user.login = "test-ai-bot"
    mock_old_review.commit_id = "newcommitsha" # Same commit SHA, so no dismissal due to new commits
    mock_github.get_pr_reviews.return_value = [mock_old_review]
    mock_github.get_last_bot_review.return_value = mock_old_review

    with caplog.at_level(logging.INFO):
        with mock.patch.object(sys, 'exit') as mock_exit:
            main()
            mock_exit.assert_not_called()
    
    assert "Action is 'edited'" in caplog.text # Check for the info log
    mock_github.dismiss_review.assert_not_called() # Crucial: old review should not be dismissed
    mock_github.post_pr_review.assert_called_once() # Still posts a new review

def test_main_no_pr_number(mock_env_vars, mock_handlers, monkeypatch):
    """Test main exits if PR number cannot be determined."""
    event_data = {"repository": {"full_name": "owner/repo"}} # No PR info
    event_file_path = os.getenv("GITHUB_EVENT_PATH") # Get the path from env
    with open(event_file_path, 'w') as f:
        json.dump(event_data, f)
    
    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)
    # No need to os.remove, mock_event_file fixture will try, but this test rewrites it.
    # It's safer to let the fixture handle its own creation/deletion if possible
    # or ensure this test cleans up if it circumvents the fixture's standard path.

def test_main_no_diff_content(mock_env_vars, mock_event_file, mock_handlers):
    """Test main exits gracefully if no diff content is found."""
    _, mock_github, _ = mock_handlers
    mock_github.get_pr_diff.return_value = "" # Simulate no diff

    with mock.patch.object(sys, 'exit') as mock_exit:
        mock_exit.side_effect = SystemExit # Make the mock raise SystemExit
        with pytest.raises(SystemExit): # Expect SystemExit to be raised
            main()
        mock_exit.assert_called_once_with(0) # Graceful exit
    
    # Ensure analysis and posting review are not called
    mock_handlers[2].summarize_changes.assert_not_called()
    mock_github.post_pr_review.assert_not_called()

def test_main_detailed_analysis_fails(mock_env_vars, mock_event_file, mock_handlers):
    """Test that review is still posted if only detailed (heavy/refined) analysis fails."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers
    mock_analysis.analyze_code_changes.side_effect = RuntimeError("Heavy analysis explosion!")
    # analyze_heavy_model_output should not be called if analyze_code_changes fails
    # And refined_analysis_results will be None

    # Reset generate_review_body mock to check arguments precisely for this path
    mock_analysis.generate_review_body.reset_mock()
    mock_analysis.generate_review_body.return_value = "Review with summary only due to error"

    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called() # Should still complete and post summary

    mock_analysis.summarize_changes.assert_called_once()
    mock_analysis.analyze_code_changes.assert_called_once()
    mock_analysis.analyze_heavy_model_output.assert_not_called() # Because heavy failed
    
    # In this scenario (heavy_analysis error), heavy_analysis itself will be None in main.py
    mock_analysis.generate_review_body.assert_called_once_with(
        summary="Mocked summary.",
        refined_analysis=None, 
        heavy_analysis_raw=None  # heavy_analysis is None due to the exception
    )
    mock_github.post_pr_review.assert_called_once_with(
        pr_number=123,
        review_body="Review with summary only due to error",
        commit_id="newcommitsha",
        event='COMMENT',
        line_comments=[] # No line comments if detailed analysis failed
    )

def test_main_missing_required_env_var(mock_env_vars, mock_event_file, mock_handlers, monkeypatch):
    """Test main exits if a required environment variable is missing."""
    monkeypatch.delenv("HEAVY_MODEL_ID") 
    original_get_required = config.get_required_env_var
    def mock_get_required_selectively(var_name, default=None):
        if var_name == "HEAVY_MODEL_ID":
            # Simulate the actual behavior of get_required_env_var when var is missing
            if default is None: # Only raise if no default is provided internally by the function
                 raise ValueError(f"Required environment variable '{var_name}' is not set.")
            return default # Should not happen for HEAVY_MODEL_ID in main() as it's directly required
        # For other vars, if testing a specific one, ensure this doesn't interfere
        # For this test, focusing on HEAVY_MODEL_ID means other calls should pass if they are valid.
        # This mock is tricky if other get_required_env_var calls happen before the one we target.
        # A more robust way is to ensure BedrockHandler init fails if it checks this, or main() fails at its check.
        # The ValueError is raised by config.get_required_env_var itself.
        return os.getenv(var_name, default) if default is not None else os.getenv(var_name)

    # The error should be caught by main()'s own try-except for ValueError from config
    with mock.patch.object(sys, 'exit') as mock_exit, \
         mock.patch('src.config.get_required_env_var', side_effect=ValueError("Required environment variable 'HEAVY_MODEL_ID' is not set.")):
        main()
        mock_exit.assert_called_once_with(1)

def test_main_github_handler_init_fails(mock_env_vars, mock_event_file, monkeypatch):
    """Test main exits if GithubHandler initialization fails."""
    # Mock the constructor call within main.py for GithubHandler
    monkeypatch.setattr('src.main.GithubHandler', mock.MagicMock(side_effect=ValueError("GH Init Error")))
    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)

def test_main_bedrock_handler_init_fails(mock_env_vars, mock_event_file, monkeypatch):
    """Test main exits if BedrockHandler initialization fails."""
    monkeypatch.setattr('src.main.BedrockHandler', mock.MagicMock(side_effect=ValueError("Bedrock Init Error")))
    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)

def test_main_pr_number_from_issue_comment_event(mock_env_vars, mock_event_file, mock_handlers, monkeypatch):
    """Test PR number extraction from an issue_comment event."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers
    event_data = {
        "issue": {
            "number": 789, 
            "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/123"}
        },
        "repository": {"full_name": "owner/repo"},
        "action": "created" 
    }
    with open(mock_event_file, 'w') as f: 
        json.dump(event_data, f)

    # Modify mock_pr_obj head sha to match what this flow would expect if PR object is fetched.
    # In this test, we only care about the PR number extraction and initial calls based on it.
    mock_github._get_pull_request_obj.return_value.head.sha = "issuecommentprsha"

    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called()
    
    mock_github.get_pr_diff.assert_called_once_with(pr_number=123)
    mock_github.post_pr_review.assert_called_once_with(
        pr_number=123,
        review_body=mock.ANY,
        commit_id="issuecommentprsha",
        event='COMMENT',
        line_comments=mock.ANY # Check if it was called with line_comments kwarg
    )

# Consider adding tests for different event types if PR number extraction logic is complex
# (e.g., issue_comment, workflow_dispatch)

def test_main_pr_number_from_issue_comment_event(mock_env_vars, mock_event_file, mock_handlers, monkeypatch):
    """Test PR number extraction from an issue_comment event."""
    mock_bedrock, mock_github, mock_analysis = mock_handlers
    event_data = {
        "issue": {
            "number": 789, # Issue number
            "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/123"}
        },
        "repository": {"full_name": "owner/repo"},
        "action": "created" # issue_comment action
    }
    # mock_event_file fixture already creates and manages the GITHUB_EVENT_PATH file.
    # We just need to ensure its content is what this test expects.
    # The event_file_path from mock_event_file is used by main() via config.
    with open(mock_event_file, 'w') as f: # mock_event_file yields the path
        json.dump(event_data, f)

    with mock.patch.object(sys, 'exit') as mock_exit:
        main()
        mock_exit.assert_not_called()
    
    mock_github.get_pr_diff.assert_called_once_with(pr_number=123)
    # os.remove(mock_event_file) # Fixture handles cleanup 