import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from github import Github, GithubException, UnknownObjectException
from github.PullRequest import PullRequest
from github.File import File as GithubFile
from github.IssueComment import IssueComment
from github.PullRequestReview import PullRequestReview
from github.Commit import Commit

from src import config

logger = logging.getLogger(__name__)

# Signature to identify comments/reviews made by this bot
# This should be unique and consistently used in review bodies/comments posted by the bot.
BOT_SIGNATURE = "<!-- AI_CODE_REVIEWER_BOT_PR_REVIEW -->"

class GithubHandler:
    """Handles interactions with the GitHub API via PyGithub."""

    def __init__(self, github_token: Optional[str] = None, repo_name: Optional[str] = None):
        """
        Initializes the Github client.

        Args:
            github_token: GitHub token. Defaults to config.GITHUB_TOKEN.
            repo_name: Full repository name (e.g., 'owner/repo'). 
                       Required if not inferable from GITHUB_EVENT_PATH.

        Raises:
            ValueError: If GitHub token or repository name is not available.
        """
        self.token = github_token or config.GITHUB_TOKEN
        if not self.token:
            raise ValueError("GitHub token is required but not provided or found in config.")

        try:
            self.gh = Github(self.token)
            logger.info("PyGithub client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PyGithub client: {e}")
            raise RuntimeError(f"PyGithub initialization failed: {e}") from e

        self.repo_name = repo_name
        if not self.repo_name and config.GITHUB_EVENT_PATH:
            try:
                # Attempt to load event data to get repository name
                with open(config.GITHUB_EVENT_PATH, 'r') as f:
                    event_data = json.load(f)
                self.repo_name = event_data.get('repository', {}).get('full_name')
                if self.repo_name:
                    logger.info(f"Inferred repository name: {self.repo_name} from GITHUB_EVENT_PATH.")
            except Exception as e:
                logger.warning(f"Could not infer repository name from GITHUB_EVENT_PATH: {e}. It must be provided.")
        
        if not self.repo_name:
            # This check is important if GITHUB_EVENT_PATH is not available or doesn't contain repo info
            # In a reusable workflow, the event path context might be different.
            # The calling workflow should ideally pass the repository explicitly if needed.
            logger.warning("Repository name was not provided and could not be inferred. Operations requiring it will fail.")
            # Not raising error immediately, as some operations (if any) might not need it,
            # or it might be set later. But most will.

    @staticmethod
    def _generate_file_sha256(filename: str) -> str:
        """Generates a SHA256 hash for the given filename."""
        return hashlib.sha256(filename.encode('utf-8')).hexdigest()

    def _get_pull_request_obj(self, pr_number: int, repo_name_override: Optional[str] = None) -> PullRequest:
        """Helper to get a PullRequest object."""
        target_repo_name = repo_name_override or self.repo_name
        if not target_repo_name:
            raise ValueError("Repository name is not set. Cannot fetch pull request.")
        try:
            repo = self.gh.get_repo(target_repo_name)
            pr = repo.get_pull(pr_number)
            return pr
        except UnknownObjectException as e:
            logger.error(f"Pull request #{pr_number} not found in repository {target_repo_name}.")
            raise ValueError(f"Pull request #{pr_number} not found in {target_repo_name}.") from e
        except GithubException as e:
            logger.error(f"GitHub API error fetching PR #{pr_number} from {target_repo_name}: {e.status} {e.data}")
            # Match test expectation for error message string
            raise RuntimeError(f"GitHub API error fetching PR {pr_number}: {e.data.get('message', str(e))}") from e

    def get_pr_diff(self, pr_number: int, repo_name_override: Optional[str] = None) -> str:
        """
        Fetches the diff for a given pull request number.

        Args:
            pr_number: The pull request number.
            repo_name_override: Optional repository name to override the one set during init.

        Returns:
            The diff content as a string.
        
        Raises:
            ValueError: If PR not found or repository name not set.
            RuntimeError: For GitHub API errors.
        """
        pr = self._get_pull_request_obj(pr_number, repo_name_override)
        try:
            # Re-evaluating the requirement: "Diff for each changed file"
            # This implies we should iterate files.
            diffs: Dict[str, str] = {}
            files: List[GithubFile] = list(pr.get_files())
            if not files:
                logger.info(f"PR #{pr_number} has no changed files.")
                return ""
            
            for file in files:
                if file.patch: # file.patch contains the diff for that file
                    diffs[file.filename] = file.patch
                else:
                    logger.info(f"File {file.filename} in PR #{pr_number} has no patch data (e.g., binary, renamed, or mode change only).")
            
            # For this function, a single string is expected by the plan. Concatenate them.
            # This matches the structure of a unified diff more closely if formatted well.
            full_diff_str = ""
            for filename, patch_content in diffs.items():
                full_diff_str += f"diff --git a/{filename} b/{filename}\n"
                # Standard diff headers might be missing, depending on PyGithub's patch content.
                # The file.patch from PyGithub is usually just the hunk lines.
                # Example: @@ -1,1 +1,1 @@
                # -old line
                # +new line
                full_diff_str += f"--- a/{filename}\n"
                full_diff_str += f"+++ b/{filename}\n"
                full_diff_str += patch_content + "\n\n"
            
            logger.info(f"Successfully fetched diffs for {len(diffs)} files in PR #{pr_number}.")
            return full_diff_str.strip()

        except GithubException as e:
            logger.error(f"GitHub API error fetching diff for PR #{pr_number}: {e.status} {e.data}")
            raise RuntimeError(f"GitHub API error fetching diff: {e.data.get('message', str(e))}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching diff for PR #{pr_number}: {e}")
            raise RuntimeError(f"Unexpected error fetching diff: {str(e)}") from e

    def get_pr_changed_files(self, pr_number: int, repo_name_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches a list of changed files in a pull request, including their diffs/patches.

        Args:
            pr_number: The pull request number.
            repo_name_override: Optional repository name to override the one set during init.

        Returns:
            A list of dictionaries, where each dictionary contains details for a changed file
            including 'filename', 'status' (added, modified, removed), and 'patch' (diff).
        """
        pr = self._get_pull_request_obj(pr_number, repo_name_override)
        changed_files_data = []
        try:
            files: List[GithubFile] = list(pr.get_files())
            for file in files:
                file_data = {
                    "filename": file.filename,
                    "status": file.status, # e.g., 'added', 'modified', 'removed', 'renamed', 'copied', 'changed', 'unchanged'
                    "patch": file.patch if file.patch else "", # The diff for this file
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "blob_url": file.blob_url,
                    "raw_url": file.raw_url,
                    "contents_url": file.contents_url
                }
                changed_files_data.append(file_data)
            logger.info(f"Fetched {len(changed_files_data)} changed files for PR #{pr_number}.")
            return changed_files_data
        except GithubException as e:
            logger.error(f"GitHub API error fetching changed files for PR #{pr_number}: {e.status} {e.data}")
            raise RuntimeError(f"GitHub API error fetching changed files for PR {pr_number}: {e.data.get('message', str(e))}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching changed files for PR #{pr_number}: {e}")
            raise RuntimeError(f"Unexpected error fetching changed files: {str(e)}") from e

    def get_pr_issue_comments(self, pr_number: int, repo_name_override: Optional[str] = None) -> List[IssueComment]:
        """Fetches all issue comments for a given pull request."""
        pr = self._get_pull_request_obj(pr_number, repo_name_override)
        try:
            comments = list(pr.get_issue_comments())
            logger.info(f"Fetched {len(comments)} issue comments for PR #{pr_number}.")
            return comments
        except GithubException as e:
            logger.error(f"GitHub API error fetching issue comments for PR #{pr_number}: {e.status} {e.data}")
            raise RuntimeError(f"GitHub API error fetching issue comments: {e.data.get('message', str(e))}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching issue comments for PR #{pr_number}: {e}")
            raise RuntimeError(f"Unexpected error fetching issue comments: {str(e)}") from e

    def get_bot_issue_comments(self, pr_comments: List[IssueComment]) -> List[IssueComment]:
        """Filters a list of PR comments to find those made by this bot."""
        bot_comments = [comment for comment in pr_comments if comment.body and BOT_SIGNATURE in comment.body]
        logger.info(f"Found {len(bot_comments)} comments made by the bot.")
        return bot_comments

    def get_pr_reviews(self, pr_number: int, repo_name_override: Optional[str] = None) -> List[PullRequestReview]:
        """Fetches all reviews for a given pull request."""
        pr = self._get_pull_request_obj(pr_number, repo_name_override)
        try:
            reviews = list(pr.get_reviews())
            logger.info(f"Fetched {len(reviews)} reviews for PR #{pr_number}.")
            return reviews
        except GithubException as e:
            logger.error(f"GitHub API error fetching reviews for PR #{pr_number}: {e.status} {e.data}")
            raise RuntimeError(f"GitHub API error fetching reviews: {e.data.get('message', str(e))}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching reviews for PR #{pr_number}: {e}")
            raise RuntimeError(f"Unexpected error fetching reviews: {str(e)}") from e

    def get_last_bot_review(self, pr_reviews: List[PullRequestReview]) -> Optional[PullRequestReview]:
        """
        Finds the last review submitted by this bot from a list of PR reviews.
        Sorts by submission date to find the most recent.
        """
        bot_reviews = []
        for review in pr_reviews:
            # Reviews can have state PENDING, check for submitted_at and body
            if review.body and BOT_SIGNATURE in review.body and review.state != 'PENDING' and review.submitted_at:
                bot_reviews.append(review)
        
        if not bot_reviews:
            logger.info("No past reviews found from this bot.")
            return None

        # Sort by submitted_at, most recent first
        bot_reviews.sort(key=lambda r: r.submitted_at, reverse=True)
        last_review = bot_reviews[0]
        logger.info(f"Found last bot review: ID {last_review.id}, Commit SHA: {last_review.commit_id}, Submitted: {last_review.submitted_at}")
        return last_review

    def dismiss_review(self, review: PullRequestReview, message: str) -> bool:
        """
        Dismisses a previously submitted pull request review.

        Args:
            review: The PullRequestReview object to dismiss.
            message: The message to include with the dismissal.

        Returns:
            True if dismissal was successful, False otherwise.
        """
        if not message:
            raise ValueError("Dismissal message cannot be empty.")
        try:
            review.dismiss(message)
            logger.info(f"Successfully dismissed review ID {review.id} with message: {message}")
            return True
        except GithubException as e:
            logger.error(f"GitHub API error dismissing review ID {review.id}: {e.status} {e.data}")
            # Specific check for 404 if review already dismissed or not dismissible by user
            if e.status == 404:
                 logger.warning(f"Review ID {review.id} could not be dismissed (perhaps already dismissed or permissions issue). Message: {e.data.get('message')}")
                 return False
            raise RuntimeError(f"GitHub API error dismissing review: {e.data.get('message', str(e))}") from e
        except Exception as e:
            logger.error(f"Unexpected error dismissing review ID {review.id}: {e}")
            raise RuntimeError(f"Unexpected error dismissing review: {str(e)}") from e
    
    def post_pr_review(self, pr_number: int, review_body: str, event: Optional[str] = None,
                       commit_id: Optional[str] = None, repo_name_override: Optional[str] = None,
                       line_comments: Optional[List[Dict[str, Any]]] = None) -> Optional[PullRequestReview]:
        """
        Posts a review to a pull request, potentially with line-specific comments.

        Args:
            pr_number: The pull request number.
            review_body: The content of the main review.
            event: The type of review event ('APPROVE', 'REQUEST_CHANGES', 'COMMENT').
            commit_id: The SHA of the commit to comment on.
            repo_name_override: Optional repository name.
            line_comments: A list of dictionaries for line-specific comments.
                           Each dict should have:
                           {'path': str, 'line': int, 'body': str}
                           The method will add a hyperlink to the body.

        Returns:
            The created PullRequestReview object if successful, None otherwise.
        """
        # if not review_body: # Allow initially empty body, signature will be added
        #     raise ValueError("Review body cannot be empty.")

        pr = self._get_pull_request_obj(pr_number, repo_name_override)
        target_repo_name = repo_name_override or self.repo_name
        if not target_repo_name:
            raise ValueError("Repository name is not set. Cannot post review.")
        
        owner, repo_name_only = target_repo_name.split('/') if '/' in target_repo_name else (None, None)
        if not owner or not repo_name_only:
            logger.error(f"Could not parse owner and repo name from {target_repo_name} for hyperlink generation.")
            # Fallback or error, for now, links might be broken if this occurs.

        processed_line_comments = []
        overflow_comment_bodies = []
        
        # Append bot signature to the main review body
        main_review_body = f"{review_body.strip()}\n\n{BOT_SIGNATURE}"

        if line_comments:
            for idx, comment_data in enumerate(line_comments):
                path = comment_data.get("path")
                line = comment_data.get("line")
                original_body = comment_data.get("body")

                if not all([path, isinstance(line, int), original_body]):
                    logger.warning(f"Skipping invalid line comment data: {comment_data}")
                    continue

                file_sha = self._generate_file_sha256(path)
                # GitHub uses different anchors for left/right side of diff: #diff-<filesha>L<line> or #diff-<filesha>R<line>
                # For a review comment on a specific line of a commit, the 'line' parameter refers to the line in the new version.
                # So, R<line> is appropriate.
                hyperlink = f"https://github.com/{owner}/{repo_name_only}/pull/{pr_number}/files#diff-{file_sha}R{line}" if owner and repo_name_only else ""
                
                formatted_comment_body = f"{original_body}\n\n[Link to file]({hyperlink})" if hyperlink else original_body

                if idx < 50: # GitHub API limit for review comments
                    processed_line_comments.append({
                        "path": path,
                        "line": line, # Use 'line' for absolute file line on the commit
                        "body": formatted_comment_body
                    })
                else:
                    overflow_comment_bodies.append(f"- **{path} (line {line})**: {formatted_comment_body}")
            
            if overflow_comment_bodies:
                main_review_body += "\n\n--- Additional Suggestions ---\n" + "\n".join(overflow_comment_bodies)

        try:
            if not commit_id:
                commit_id = pr.head.sha 
                logger.info(f"Using PR head commit SHA ({commit_id}) for review on PR #{pr_number}.")

            repo = self.gh.get_repo(target_repo_name)
            commit_obj = repo.get_commit(commit_id)

            review = pr.create_review(
                commit=commit_obj,
                body=main_review_body,
                event=event,
                comments=processed_line_comments if processed_line_comments else []
            )
            logger.info(f"Successfully posted a '{event}' review to PR #{pr_number}. Review ID: {review.id}")
            return review
        except GithubException as e:
            logger.error(f"GitHub API error posting review to PR #{pr_number}: {e.status} {e.data}")
            message = e.data.get('message', str(e))
            if e.status == 422: 
                errors = e.data.get('errors', [])
                if errors:
                    message += f" Errors: {errors}"
                logger.error(f"Unprocessable entity error for PR #{pr_number} review. Body: '{main_review_body[:200]}...', Event: {event}, Commit: {commit_id}. Details: {message}")
                # Log processed_line_comments for debugging if needed, carefully due to size
                logger.debug(f"Processed line comments for 422 error: {processed_line_comments}")
                raise ValueError(f"Invalid parameters for PR review: {message}") from e
            raise RuntimeError(f"GitHub API error posting review: {message}") from e
        except Exception as e:
            logger.error(f"Unexpected error posting review to PR #{pr_number}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error posting review: {str(e)}") from e
