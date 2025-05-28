import pytest
from unittest import mock
import json
import logging
import hashlib # For _generate_file_sha256 test

from github import Github, GithubException, UnknownObjectException
from github.PullRequest import PullRequest
from github.File import File as GithubFile
from github.Commit import Commit
from github.PullRequestReview import PullRequestReview
from github.IssueComment import IssueComment # Added

from src.github_handler import GithubHandler, BOT_SIGNATURE # Import BOT_SIGNATURE
from src import config

# --- Fixtures ---

@pytest.fixture
def mock_config_github(monkeypatch):
    """Mocks github related config values."""
    monkeypatch.setattr(config, 'GITHUB_TOKEN', "test_github_token")
    monkeypatch.setattr(config, 'GITHUB_EVENT_PATH', "/github/workflow/event.json")
    # We won't actually read this file in unit tests, repo_name will be passed or mocked in event data

@pytest.fixture
def mock_pygithub_client():
    """Mocks the Github class from PyGithub, targeting where it's used in GithubHandler."""
    with mock.patch('src.github_handler.Github') as mock_github_constructor:
        mock_gh_instance = mock.MagicMock(spec=Github)
        mock_github_constructor.return_value = mock_gh_instance
        yield mock_gh_instance

@pytest.fixture
def mock_repo(mock_pygithub_client):
    """Mocks a Repository object."""
    mock_repo_instance = mock.MagicMock(spec=['get_pull', 'get_commit'])
    mock_repo_instance.get_issue = mock.MagicMock()
    mock_pygithub_client.get_repo.return_value = mock_repo_instance
    return mock_repo_instance

@pytest.fixture
def mock_pr(mock_repo):
    """Mocks a PullRequest object."""
    mock_pr_instance = mock.MagicMock(spec=PullRequest)
    mock_pr_instance.number = 123
    mock_pr_instance.patch_url = "https://example.com/pull/123.patch"
    mock_pr_instance.diff_url = "https://example.com/pull/123.diff"
    
    # Mock head commit for review posting
    mock_head_commit = mock.MagicMock(spec=Commit)
    mock_head_commit.sha = "testheadsha123"
    mock_pr_instance.head = mock_head_commit
    
    # For get_pr_reviews and get_pr_issue_comments
    mock_pr_instance.get_reviews = mock.MagicMock()
    mock_pr_instance.get_issue_comments = mock.MagicMock() # PyGithub uses this for PR comments too

    mock_repo.get_pull.return_value = mock_pr_instance
    return mock_pr_instance

@pytest.fixture
def mock_issue_for_comments(mock_repo):
    """Mocks an Issue object, used for PR comments via get_issue()."""
    mock_issue_instance = mock.MagicMock()
    mock_issue_instance.get_comments = mock.MagicMock()
    mock_repo.get_issue.return_value = mock_issue_instance
    return mock_issue_instance

@pytest.fixture
def mock_github_file():
    """Creates a mock GithubFile object."""
    def _create_mock_file(filename="test.py", status="modified", patch="@@ -1,1 +1,1 @@\n-old\n+new", additions=1, deletions=1, changes=2):
        mock_file = mock.MagicMock(spec=GithubFile)
        mock_file.filename = filename
        mock_file.status = status
        mock_file.patch = patch
        mock_file.additions = additions
        mock_file.deletions = deletions
        mock_file.changes = changes
        mock_file.blob_url = f"https://example.com/blob/main/{filename}"
        mock_file.raw_url = f"https://example.com/raw/main/{filename}"
        mock_file.contents_url = f"https://example.com/contents/{filename}"
        return mock_file
    return _create_mock_file

# --- Initialization Tests ---

def test_github_handler_init_success_with_token_and_repo(mock_config_github, mock_pygithub_client):
    """Test successful initialization with explicit token and repo name."""
    with mock.patch('src.github_handler.Github') as local_mock_gh_constructor:
        mock_instance = mock.MagicMock()
        local_mock_gh_constructor.return_value = mock_instance
        handler = GithubHandler(github_token="override_token", repo_name="owner/repo")
        assert handler.token == "override_token"
        assert handler.repo_name == "owner/repo"
        local_mock_gh_constructor.assert_called_once_with("override_token")
        assert handler.gh == mock_instance

def test_github_handler_init_success_from_config(mock_config_github, mock_pygithub_client):
    """Test successful initialization using config for token and no explicit repo_name (relies on event path)."""
    mock_event_data = {"repository": {"full_name": "event/repo"}}
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mock_event_data))) as mock_file_open, \
         mock.patch('src.github_handler.Github') as local_mock_gh_constructor:
            mock_instance_inner = mock.MagicMock()
            local_mock_gh_constructor.return_value = mock_instance_inner
            handler = GithubHandler()
            mock_file_open.assert_called_once_with(config.GITHUB_EVENT_PATH, 'r')
            assert handler.token == "test_github_token"
            assert handler.repo_name == "event/repo"
            local_mock_gh_constructor.assert_called_once_with("test_github_token")
            assert handler.gh == mock_instance_inner

def test_github_handler_init_token_missing(mock_config_github, monkeypatch):
    """Test ValueError if GITHUB_TOKEN is not provided or found."""
    monkeypatch.setattr(config, 'GITHUB_TOKEN', None)
    with pytest.raises(ValueError, match="GitHub token is required but not provided or found in config."):
        GithubHandler()

def test_github_handler_init_repo_name_missing_and_no_event_path(mock_config_github, monkeypatch, mock_pygithub_client):
    """Test warning if repo_name is not given and cannot be inferred."""
    monkeypatch.setattr(config, 'GITHUB_EVENT_PATH', None)
    with mock.patch('os.path.exists', return_value=False), \
         mock.patch.object(logging.getLogger('src.github_handler'), 'warning') as mock_logger_warning:
        handler = GithubHandler(repo_name=None)
        assert handler.repo_name is None
        mock_logger_warning.assert_any_call("Repository name was not provided and could not be inferred. Operations requiring it will fail.")

def test_github_handler_init_pygithub_exception(mock_config_github, monkeypatch):
    """Test RuntimeError if PyGithub initialization fails."""
    with mock.patch('src.github_handler.Github', side_effect=GithubException(status=500, data={"message":"PyGithub connect error"}, headers={})) as mock_constructor:
        with pytest.raises(RuntimeError, match=r'PyGithub initialization failed: 500 {"message": "PyGithub connect error"}'):
            GithubHandler()
        mock_constructor.assert_called_once()

# --- _get_pull_request_obj Tests ---

def test_get_pull_request_obj_success(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test _get_pull_request_obj successfully retrieves a PR."""
    handler = GithubHandler(repo_name="owner/repo")
    pr_obj = handler._get_pull_request_obj(pr_number=123)
    mock_pygithub_client.get_repo.assert_called_once_with("owner/repo")
    mock_repo.get_pull.assert_called_once_with(123)
    assert pr_obj == mock_pr

def test_get_pull_request_obj_repo_name_missing(mock_config_github, monkeypatch, mock_pygithub_client):
    """Test _get_pull_request_obj fails if repo_name is not set."""
    monkeypatch.setattr(config, 'GITHUB_EVENT_PATH', None)
    with mock.patch('os.path.exists', return_value=False):
        handler = GithubHandler(repo_name=None)
    with pytest.raises(ValueError, match="Repository name is not set. Cannot fetch pull request."):
        handler._get_pull_request_obj(pr_number=123)

def test_get_pull_request_obj_unknown_object(mock_config_github, mock_pygithub_client, mock_repo):
    """Test _get_pull_request_obj handles UnknownObjectException (PR not found)."""
    mock_repo.get_pull.side_effect = UnknownObjectException(status=404, data={"message": "Not Found"}, headers={})
    handler = GithubHandler(repo_name="owner/repo")
    with pytest.raises(ValueError, match="Pull request #123 not found in owner/repo."):
        handler._get_pull_request_obj(pr_number=123)

def test_get_pull_request_obj_github_exception(mock_config_github, mock_pygithub_client, mock_repo):
    """Test _get_pull_request_obj handles generic GithubException."""
    mock_repo.get_pull.side_effect = GithubException(status=500, data={"message": "Server Error"}, headers={})
    handler = GithubHandler(repo_name="owner/repo")
    with pytest.raises(RuntimeError, match="GitHub API error fetching PR 123: Server Error"):
        handler._get_pull_request_obj(pr_number=123)

# --- get_pr_changed_files Tests ---

def test_get_pr_changed_files_success(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_github_file):
    """Test get_pr_changed_files successfully retrieves and formats file data."""
    file1 = mock_github_file(filename="file1.py", status="modified", patch="patch1", additions=10, deletions=2)
    file2 = mock_github_file(filename="file2.txt", status="added", patch="patch2", additions=5, deletions=0)
    mock_pr.get_files.return_value = [file1, file2]

    handler = GithubHandler(repo_name="owner/repo")
    changed_files = handler.get_pr_changed_files(pr_number=123)

    assert len(changed_files) == 2
    assert changed_files[0]["filename"] == "file1.py"
    assert changed_files[0]["patch"] == "patch1"
    assert changed_files[1]["filename"] == "file2.txt"
    assert changed_files[1]["status"] == "added"
    mock_pr.get_files.assert_called_once()

def test_get_pr_changed_files_no_files(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test get_pr_changed_files with a PR that has no changed files."""
    mock_pr.get_files.return_value = []
    handler = GithubHandler(repo_name="owner/repo")
    changed_files = handler.get_pr_changed_files(pr_number=123)
    assert len(changed_files) == 0

def test_get_pr_changed_files_api_error(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test get_pr_changed_files handles GithubException when fetching files."""
    mock_pr.get_files.side_effect = GithubException(status=500, data={"message": "Files Error"}, headers={})
    handler = GithubHandler(repo_name="owner/repo")
    with pytest.raises(RuntimeError, match="GitHub API error fetching changed files for PR 123: Files Error"):
        handler.get_pr_changed_files(pr_number=123)

# --- get_pr_diff Tests ---

def test_get_pr_diff_success(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_github_file):
    """Test get_pr_diff successfully combines patches from multiple files."""
    file1 = mock_github_file(filename="src/file1.py", patch="@@ -1 +1 @@\n-old1\n+new1")
    file2 = mock_github_file(filename="tests/file2.py", patch="@@ -10 +10 @@\n-old_test\n+new_test")
    mock_pr.get_files.return_value = [file1, file2]

    handler = GithubHandler(repo_name="owner/repo")
    diff = handler.get_pr_diff(pr_number=123)

    expected_diff = (
        "diff --git a/src/file1.py b/src/file1.py\n"
        "--- a/src/file1.py\n"
        "+++ b/src/file1.py\n"
        "@@ -1 +1 @@\n-old1\n+new1\n\n"
        "diff --git a/tests/file2.py b/tests/file2.py\n"
        "--- a/tests/file2.py\n"
        "+++ b/tests/file2.py\n"
        "@@ -10 +10 @@\n-old_test\n+new_test"
    )
    assert diff == expected_diff 

def test_get_pr_diff_no_files(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test get_pr_diff returns an empty string if no files have patches."""
    mock_pr.get_files.return_value = []
    handler = GithubHandler(repo_name="owner/repo")
    diff = handler.get_pr_diff(pr_number=123)
    assert diff == ""

def test_get_pr_diff_file_no_patch(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_github_file, caplog):
    """Test get_pr_diff handles files without patch data (e.g., binary)."""
    file_with_patch = mock_github_file(filename="text.txt", patch="patch_data")
    file_no_patch = mock_github_file(filename="image.png", patch=None)
    mock_pr.get_files.return_value = [file_with_patch, file_no_patch]

    handler = GithubHandler(repo_name="owner/repo")
    with caplog.at_level(logging.INFO):
        diff = handler.get_pr_diff(pr_number=123)
    
    expected_diff_for_text_file = (
        "diff --git a/text.txt b/text.txt\n"
        "--- a/text.txt\n"
        "+++ b/text.txt\n"
        "patch_data"
    )
    assert diff == expected_diff_for_text_file.strip()
    assert "image.png" not in diff
    assert "File image.png in PR #123 has no patch data" in caplog.text

# --- post_pr_review Tests ---

def test_post_pr_review_success_no_line_comments(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test successfully posting a PR review comment without line-specific comments."""
    mock_commit_obj = mock.MagicMock(spec=Commit)
    mock_repo.get_commit.return_value = mock_commit_obj
    
    mock_review_obj = mock.MagicMock(spec=PullRequestReview)
    mock_review_obj.id = 999
    mock_pr.create_review.return_value = mock_review_obj

    handler = GithubHandler(repo_name="owner/repo")
    review_body_main = "Test review content"
    expected_full_body = f"{review_body_main}\n\n{BOT_SIGNATURE}"
    
    posted_review = handler.post_pr_review(pr_number=123, review_body=review_body_main, event="COMMENT")

    mock_repo.get_commit.assert_called_once_with("testheadsha123") # From mock_pr.head.sha
    mock_pr.create_review.assert_called_once_with(
        commit=mock_commit_obj,
        body=expected_full_body,
        event="COMMENT",
        comments=[] # Ensure empty comments list when none provided
    )
    assert posted_review == mock_review_obj

def test_post_pr_review_with_line_comments_and_hyperlinks(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test posting a review with line-specific comments and generated hyperlinks."""
    mock_commit_obj = mock.MagicMock(spec=Commit)
    mock_repo.get_commit.return_value = mock_commit_obj
    mock_review_obj = mock.MagicMock(spec=PullRequestReview); mock_review_obj.id = 1000
    mock_pr.create_review.return_value = mock_review_obj

    handler = GithubHandler(repo_name="owner/repo_name_val") # Specific repo name for URL
    pr_num_for_url = mock_pr.number # 123
    
    line_comments_data = [
        {"path": "src/file1.py", "line": 10, "body": "Suggestion for file1"},
        {"path": "docs/file2.md", "line": 5, "body": "Typo in docs"},
    ]
    
    # Expected SHA256 hashes for filenames (must match _generate_file_sha256)
    sha_file1 = GithubHandler._generate_file_sha256("src/file1.py")
    sha_file2 = GithubHandler._generate_file_sha256("docs/file2.md")

    expected_gh_comments_arg = [
        {
            "path": "src/file1.py", 
            "line": 10, 
            "body": f"Suggestion for file1\n\n[Link to file](https://github.com/owner/repo_name_val/pull/{pr_num_for_url}/files#diff-{sha_file1}R10)"
        },
        {
            "path": "docs/file2.md", 
            "line": 5, 
            "body": f"Typo in docs\n\n[Link to file](https://github.com/owner/repo_name_val/pull/{pr_num_for_url}/files#diff-{sha_file2}R5)"
        },
    ]
    review_body_main = "Main review summary."
    expected_full_body = f"{review_body_main}\n\n{BOT_SIGNATURE}"

    posted_review = handler.post_pr_review(
        pr_number=pr_num_for_url, 
        review_body=review_body_main, 
        line_comments=line_comments_data,
        event="REQUEST_CHANGES"
    )

    mock_pr.create_review.assert_called_once_with(
        commit=mock_commit_obj,
        body=expected_full_body,
        event="REQUEST_CHANGES",
        comments=expected_gh_comments_arg
    )
    assert posted_review == mock_review_obj

def test_post_pr_review_line_comments_truncation_and_overflow(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test truncation of line comments to 50 and appending overflow to body."""
    mock_commit_obj = mock.MagicMock(spec=Commit)
    mock_repo.get_commit.return_value = mock_commit_obj
    mock_review_obj = mock.MagicMock(spec=PullRequestReview); mock_review_obj.id = 1001
    mock_pr.create_review.return_value = mock_review_obj

    handler = GithubHandler(repo_name="owner/repo")
    pr_num = mock_pr.number
    
    # Create 52 line comments
    line_comments_data = []
    for i in range(52):
        filename = f"file_{i}.py"
        line_comments_data.append({
            "path": filename, 
            "line": i + 1, 
            "body": f"Suggestion {i+1} for {filename}"
        })

    review_body_main = "Main summary."
    
    # Expected first 50 comments for the 'comments' argument
    expected_gh_comments_arg = []
    for i in range(50):
        item = line_comments_data[i]
        sha = GithubHandler._generate_file_sha256(item["path"])
        expected_gh_comments_arg.append({
            "path": item["path"],
            "line": item["line"],
            "body": f"{item['body']}\n\n[Link to file](https://github.com/owner/repo/pull/{pr_num}/files#diff-{sha}R{item['line']})"
        })
        
    # --- Construct expected_full_body step-by-step, mirroring code --- 
    # Step 1: Main body with signature (code: main_review_body = f"{review_body.strip()}\n\n{BOT_SIGNATURE}")
    expected_full_body = f"{review_body_main.strip()}\n\n{BOT_SIGNATURE}"
    
    # Step 2: Collect overflow items exactly as code's overflow_comment_bodies
    # (code: overflow_comment_bodies.append(f"- **{path} (line {line})**: {formatted_comment_body}"))
    # (code: formatted_comment_body = f"{original_body}\n\n[Link to file]({hyperlink})")
    temp_overflow_strings_for_join = []
    for i in range(50, 52): # The two overflow items
        item = line_comments_data[i]
        original_body = item['body']
        path = item['path']
        line = item['line']
        
        sha_for_link = GithubHandler._generate_file_sha256(path)
        hyperlink_for_link = f"https://github.com/owner/repo/pull/{pr_num}/files#diff-{sha_for_link}R{line}"
        formatted_comment_body_for_item = f"{original_body}\n\n[Link to file]({hyperlink_for_link})"
        temp_overflow_strings_for_join.append(f"- **{path} (line {line})**: {formatted_comment_body_for_item}")

    # Step 3: Append overflow section if items exist (code: if overflow_comment_bodies: main_review_body += ...)
    if temp_overflow_strings_for_join:
        expected_full_body += "\n\n--- Additional Suggestions ---\n" + "\n".join(temp_overflow_strings_for_join)
    
    handler.post_pr_review(
        pr_number=pr_num, 
        review_body=review_body_main, 
        line_comments=line_comments_data,
        commit_id="explicit_sha" # Test with explicit commit_id too
    )

    mock_repo.get_commit.assert_called_once_with("explicit_sha")
    args, kwargs = mock_pr.create_review.call_args
    
    assert kwargs["body"] == expected_full_body
    assert kwargs["comments"] == expected_gh_comments_arg
    assert len(kwargs["comments"]) == 50
    assert kwargs["event"] is None  # Default event when not specified

def test_post_pr_review_malformed_line_comment_item(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, caplog):
    """Test that malformed line comment items are skipped."""
    handler = GithubHandler(repo_name="owner/repo")
    pr_num = mock_pr.number

    line_comments_data = [
        {"path": "good.py", "line": 1, "body": "Good"},
        {"path": "bad.py"}, # Missing line and body
        {"file_path": "ignored.py", "line": 2, "suggestion": "Wrong keys"} # Using 'file_path', 'suggestion'
    ]
    
    review_body_main = "Review with some malformed line comments."
    sha_good = GithubHandler._generate_file_sha256("good.py")
    expected_gh_comments_arg = [{
        "path": "good.py", "line": 1, 
        "body": f"Good\n\n[Link to file](https://github.com/owner/repo/pull/{pr_num}/files#diff-{sha_good}R1)"
    }]
    expected_full_body = f"{review_body_main}\n\n{BOT_SIGNATURE}"

    with caplog.at_level(logging.WARNING):
        handler.post_pr_review(pr_number=pr_num, review_body=review_body_main, line_comments=line_comments_data)
    
    mock_pr.create_review.assert_called_once()
    _, kwargs = mock_pr.create_review.call_args
    assert kwargs["comments"] == expected_gh_comments_arg
    assert kwargs["body"] == expected_full_body
    assert "Skipping invalid line comment data: {'path': 'bad.py'}" in caplog.text
    assert "Skipping invalid line comment data: {'file_path': 'ignored.py'" in caplog.text

def test_post_pr_review_with_explicit_commit_id(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test posting a review with an explicitly provided commit_id."""
    mock_commit_obj = mock.MagicMock(spec=Commit)
    mock_repo.get_commit.return_value = mock_commit_obj
    explicit_sha = "anotherSha123"
    
    handler = GithubHandler(repo_name="owner/repo")
    handler.post_pr_review(pr_number=123, review_body="Review for specific commit", commit_id=explicit_sha)

    mock_repo.get_commit.assert_called_once_with(explicit_sha)
    mock_pr.create_review.assert_called_once_with(
        commit=mock_commit_obj,
        body=f"Review for specific commit\n\n{BOT_SIGNATURE}",
        event=None, # Default event when not specified
        comments=[]
    )

def test_post_pr_review_empty_body_with_bot_signature(mock_config_github, mock_pygithub_client, mock_repo, mock_pr):
    """Test that an empty review body still gets the bot signature."""
    handler = GithubHandler(repo_name="owner/repo")
    # Intentionally passing an empty string for review_body
    handler.post_pr_review(pr_number=123, review_body="", event="APPROVE")
    
    expected_body_with_signature = f"\n\n{BOT_SIGNATURE}" # Empty body + signature
    
    mock_pr.create_review.assert_called_once()
    args, kwargs = mock_pr.create_review.call_args
    assert kwargs["body"] == expected_body_with_signature
    assert kwargs["event"] == "APPROVE"

def test_post_pr_review_github_api_error_422(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, caplog):
    """Test handling of 422 error (e.g., review on old commit) from GitHub API."""
    mock_pr.create_review.side_effect = GithubException(
        status=422, 
        data={"message": "Validation Failed", "errors": [{"field": "commit_id", "code": "stale"}]},
        headers={}
    )
    handler = GithubHandler(repo_name="owner/repo")
    
    with pytest.raises(ValueError, match=r"Invalid parameters for PR review: Validation Failed Errors: .*stale") as excinfo:
        handler.post_pr_review(pr_number=123, review_body="Content")
    
    # Check log messages after the exception has been caught and handled
    with caplog.at_level(logging.ERROR):
        # The error is logged before ValueError is raised
        pass # Allow previous logging to be checked, caplog captures across the test

    assert "GitHub API error posting review to PR #123: 422" in caplog.text
    assert "Unprocessable entity error for PR #123 review" in caplog.text
    assert "stale" in str(excinfo.value) # Check the raised exception message

def test_post_pr_review_generic_github_api_error(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, caplog):
    """Test handling of generic GitHub API errors during review posting."""
    mock_pr.create_review.side_effect = GithubException(status=500, data={"message": "Server Error"}, headers={})
    handler = GithubHandler(repo_name="owner/repo")
    
    with pytest.raises(RuntimeError, match="GitHub API error posting review: Server Error") as excinfo:
        handler.post_pr_review(pr_number=123, review_body="Content")

    with caplog.at_level(logging.ERROR): # Generic errors are logged as ERROR
        pass # Allow previous logging to be checked
    assert "GitHub API error posting review to PR #123: 500" in caplog.text
    assert "Server Error" in str(excinfo.value)

# --- _generate_file_sha256 Tests ---
def test_generate_file_sha256():
    """Test SHA256 hash generation for a filename."""
    # Not using GithubHandler instance as it's a static method
    filename = "src/my_module/file.py"
    expected_hash = hashlib.sha256(filename.encode('utf-8')).hexdigest()
    assert GithubHandler._generate_file_sha256(filename) == expected_hash

# --- PR Comment and Review Fetching Tests ---
@pytest.fixture
def mock_comments_and_reviews(mock_pr, mock_issue_for_comments):
    """Fixture to set up mock comments and reviews for a PR."""
    # PR Issue Comments (general comments on the PR, not tied to lines)
    comment1 = mock.MagicMock(spec=IssueComment)
    comment1.user = mock.MagicMock()
    comment1.user.login = "user1"
    comment1.body = "This is a general comment."
    
    comment_bot = mock.MagicMock(spec=IssueComment)
    comment_bot.user = mock.MagicMock()
    comment_bot.user.login = config.BOT_NAME # Assume BOT_NAME is configured or use a fixed test bot name
    comment_bot.body = f"Automated comment. {BOT_SIGNATURE}" # Bot comment with signature
    
    # mock_pr.get_issue_comments.return_value = [comment1, comment_bot] # For older PyGithub
    # For newer PyGithub, PR comments are often through get_issue().get_comments()
    mock_pr.get_issue_comments.return_value = [comment1, comment_bot]


    # PR Reviews (formal reviews, can include a body and line comments)
    review1 = mock.MagicMock(spec=PullRequestReview)
    review1.user = mock.MagicMock()
    review1.user.login = "user2"
    review1.body = "LGTM!"
    review1.state = "APPROVED"
    review1.commit_id = "commitsha1"
    review1.id = 1001
    review1.submitted_at = 1 # Using simple integers for mock comparison

    review_bot_old = mock.MagicMock(spec=PullRequestReview)
    review_bot_old.user = mock.MagicMock()
    review_bot_old.user.login = config.BOT_NAME 
    review_bot_old.body = f"Old AI Review. {BOT_SIGNATURE}"
    review_bot_old.state = "COMMENTED"
    review_bot_old.commit_id = "commitsha_old" # Older commit
    review_bot_old.id = 1002
    review_bot_old.submitted_at = 2

    review_bot_current = mock.MagicMock(spec=PullRequestReview)
    review_bot_current.user = mock.MagicMock()
    review_bot_current.user.login = config.BOT_NAME
    review_bot_current.body = f"Current AI Review. {BOT_SIGNATURE}"
    review_bot_current.state = "CHANGES_REQUESTED"
    review_bot_current.commit_id = "testheadsha123" # Current head SHA from mock_pr
    review_bot_current.id = 1003
    review_bot_current.submitted_at = 3 # Most recent
    
    mock_pr.get_reviews.return_value = [review1, review_bot_old, review_bot_current]
    
    return {
        "pr_comments": [comment1, comment_bot],
        "pr_reviews": [review1, review_bot_old, review_bot_current]
    }

def test_get_pr_issue_comments(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_issue_for_comments, mock_comments_and_reviews):
    """Test fetching general PR issue comments."""
    handler = GithubHandler(repo_name="owner/repo")
    comments = handler.get_pr_issue_comments(pr_number=123)
    
    mock_pr.get_issue_comments.assert_called_once() # Assert that the PR object's method was called
    assert len(comments) == 2
    assert comments == mock_comments_and_reviews["pr_comments"]

def test_get_bot_issue_comments(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_issue_for_comments, mock_comments_and_reviews, monkeypatch):
    """Test filtering for bot's general PR issue comments."""
    monkeypatch.setattr(config, 'BOT_NAME', "test-bot") # Ensure BOT_NAME is set for the test
    
    # Adjust mock_comments_and_reviews for this specific test's BOT_NAME expectation
    bot_comment = mock_comments_and_reviews["pr_comments"][1]
    bot_comment.user.login = "test-bot" # Align with monkeypatched BOT_NAME
    
    handler = GithubHandler(repo_name="owner/repo")
    # Pass the already fetched comments to the filtering method
    bot_comments = handler.get_bot_issue_comments(mock_comments_and_reviews["pr_comments"])
    
    assert len(bot_comments) == 1
    assert bot_comments[0].body.endswith(BOT_SIGNATURE)
    assert bot_comments[0].user.login == "test-bot"

def test_get_pr_reviews(mock_config_github, mock_pygithub_client, mock_repo, mock_pr, mock_comments_and_reviews):
    """Test fetching PR reviews."""
    handler = GithubHandler(repo_name="owner/repo")
    reviews = handler.get_pr_reviews(pr_number=123)
    
    mock_pr.get_reviews.assert_called_once()
    assert len(reviews) == 3
    assert reviews == mock_comments_and_reviews["pr_reviews"]

def test_get_last_bot_review(mock_config_github, mock_comments_and_reviews, monkeypatch):
    """Test finding the last review posted by the bot."""
    monkeypatch.setattr(config, 'BOT_NAME', "test-bot")
    
    # Adjust bot review user logins for this test's BOT_NAME
    mock_comments_and_reviews["pr_reviews"][1].user.login = "test-bot"
    mock_comments_and_reviews["pr_reviews"][2].user.login = "test-bot"

    handler = GithubHandler(repo_name="owner/repo") # Not strictly needed as method processes list
    
    # Scenario 1: Bot reviews exist
    last_review = handler.get_last_bot_review(mock_comments_and_reviews["pr_reviews"])
    assert last_review is not None
    assert last_review.id == 1003 # Should be the most recent one by date (mocked by order here)
    assert last_review.body.endswith(BOT_SIGNATURE)

    # Scenario 2: No bot reviews
    non_bot_reviews = [r for r in mock_comments_and_reviews["pr_reviews"] if r.user.login != "test-bot"]
    last_review_none = handler.get_last_bot_review(non_bot_reviews)
    assert last_review_none is None

    # Scenario 3: Empty review list
    assert handler.get_last_bot_review([]) is None

# --- dismiss_review Tests ---
def test_dismiss_review_success(mock_config_github):
    """Test successfully dismissing a review."""
    mock_review_to_dismiss = mock.MagicMock(spec=PullRequestReview)
    mock_review_to_dismiss.dismiss = mock.MagicMock()
    
    handler = GithubHandler(repo_name="owner/repo")
    message = "Dismissing due to new commits."
    assert handler.dismiss_review(mock_review_to_dismiss, message) is True
    mock_review_to_dismiss.dismiss.assert_called_once_with(message)

def test_dismiss_review_failure_exception(mock_config_github, caplog):
    """Test failure when dismissing a review due to API error."""
    mock_review_to_dismiss = mock.MagicMock(spec=PullRequestReview)
    mock_review_to_dismiss.dismiss.side_effect = GithubException(status=500, data={"message": "Dismissal Error"}, headers={})
    
    handler = GithubHandler(repo_name="owner/repo")
    message = "Test dismissal."
    
    with pytest.raises(RuntimeError, match="GitHub API error dismissing review: Dismissal Error") as excinfo:
        handler.dismiss_review(mock_review_to_dismiss, message)
    
    with caplog.at_level(logging.ERROR):
        pass # Check logs captured from the call within pytest.raises
    assert "GitHub API error dismissing review ID" in caplog.text # Check for part of the log
    assert "Dismissal Error" in str(excinfo.value)

def test_dismiss_review_already_dismissed_404(mock_config_github, caplog):
    """Test handling when review is already dismissed (GitHub returns 404)."""
    mock_review_to_dismiss = mock.MagicMock(spec=PullRequestReview)
    mock_review_to_dismiss.id = 789
    mock_review_to_dismiss.dismiss.side_effect = GithubException(status=404, data={"message": "Not Found"}, headers={})

    handler = GithubHandler(repo_name="owner/repo")
    message = "Test dismissal."
    # The dismiss_review logs an ERROR first, then a WARNING for 404
    with caplog.at_level(logging.WARNING): # Capture WARNING and above
        assert handler.dismiss_review(mock_review_to_dismiss, message) is False
    
    # Check for the specific WARNING message related to 404
    # Ensure the full caplog.text is inspected as it might contain the preceding ERROR log too
    assert f"Review ID {mock_review_to_dismiss.id} could not be dismissed (perhaps already dismissed or permissions issue). Message: Not Found" in caplog.text

# Add more tests as needed, e.g., for repo_name_override in various methods. 