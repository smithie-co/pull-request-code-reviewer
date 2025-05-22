import json
import logging
import os
import sys
from typing import Dict

from src import config # To ensure it's loaded, though individual configs are used directly
from src.bedrock_handler import BedrockHandler
from src.github_handler import GithubHandler
from src.analysis_service import AnalysisService

# Setup basic logging
# Consider moving this to a shared utility or ensuring it's robustly configured once.
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to orchestrate the PR code review process."""
    logger.info("Starting AI Pull Request Code Reviewer...")

    try:
        # --- Configuration & Initialization ---
        logger.info("Loading configuration and initializing handlers...")
        
        # Ensure required configurations are present (using get_required_env_var from config)
        # Some of these are directly used by handlers, but good to check upfront.
        gh_token = config.get_required_env_var("GITHUB_TOKEN")
        repo_name_env = os.getenv("GITHUB_REPOSITORY") # e.g., 'owner/repo' from action env
        event_path = config.get_required_env_var("GITHUB_EVENT_PATH")
        
        # AWS creds are checked within BedrockHandler init via config
        config.get_required_env_var("AWS_ACCESS_KEY_ID")
        config.get_required_env_var("AWS_SECRET_ACCESS_KEY")
        config.get_required_env_var("AWS_DEFAULT_REGION")
        config.get_required_env_var("HEAVY_MODEL_ID")
        config.get_required_env_var("LIGHT_MODEL_ID")
        config.get_required_env_var("DEEPSEEK_MODEL_ID")

        # Initialize handlers
        bedrock_h = BedrockHandler() # Uses config values by default
        # GithubHandler will try to infer repo_name from event_path if repo_name_env is None,
        # but explicit is better if available from action context.
        github_h = GithubHandler(github_token=gh_token, repo_name=repo_name_env)
        analysis_s = AnalysisService(bedrock_handler=bedrock_h)

        logger.info("Handlers initialized successfully.")

        # --- Get PR Details from Event Payload --- 
        logger.info(f"Loading event data from: {event_path}")
        with open(event_path, 'r') as f:
            event_data = json.load(f)

        # Extract PR number - structure depends on the event (e.g., pull_request, pull_request_target)
        # This assumes a 'pull_request' event or similar structure where PR number is available.
        if 'pull_request' in event_data:
            pr_number = event_data.get('pull_request', {}).get('number')
            action = event_data.get('action')
            if action not in ['opened', 'reopened', 'synchronize']:
                 logger.info(f"Action is '{action}'. For full review features like dismissal, only 'opened', 'reopened', or 'synchronize' are typically processed for new reviews.")
                 # Allow proceeding for potential manual triggers or other specific needs
                 pass

        elif 'issue' in event_data and event_data.get('issue', {}).get('pull_request'):
            # For issue_comment events that are on a PR
            # We need to extract PR number from the issue's pull_request URL or similar
            pr_url = event_data.get('issue', {}).get('pull_request', {}).get('url')
            if pr_url:
                pr_number = int(pr_url.split('/')[-1])
                logger.info(f"Extracted PR number {pr_number} from issue_comment event.")
            else:
                pr_number = None
        else:
            pr_number = event_data.get('number') # Fallback for workflow_dispatch with a 'number' input, or other events

        if not pr_number:
            logger.error("Could not determine pull request number from the event payload.")
            logger.error(f"Event data keys: {list(event_data.keys())}")
            if 'pull_request' in event_data:
                logger.error(f"PR data in event: {event_data.get('pull_request')}")
            sys.exit(1)
        
        logger.info(f"Processing Pull Request #{pr_number}")

        # Ensure GithubHandler has the definitive repository name if it wasn't passed via GITHUB_REPOSITORY
        if not github_h.repo_name:
            current_repo_name = event_data.get('repository', {}).get('full_name')
            if not current_repo_name:
                logger.error("Repository name could not be determined from GITHUB_REPOSITORY env var or event payload.")
                sys.exit(1)
            github_h.repo_name = current_repo_name # Set it if inferred from event after init
            logger.info(f"Using repository: {github_h.repo_name} from event payload.")
        else:
            logger.info(f"Using repository: {github_h.repo_name}")

        # --- Fetch current PR state and past bot reviews ---
        logger.info(f"Fetching PR data for PR #{pr_number}...")
        pr_object = github_h._get_pull_request_obj(pr_number=pr_number)
        current_head_sha = pr_object.head.sha
        logger.info(f"Current PR head SHA: {current_head_sha}")

        all_reviews = github_h.get_pr_reviews(pr_number=pr_number)
        last_bot_review = github_h.get_last_bot_review(all_reviews)

        if last_bot_review:
            logger.info(f"Last bot review found: ID {last_bot_review.id}, Commit SHA: {last_bot_review.commit_id}")
            if last_bot_review.commit_id != current_head_sha and (last_bot_review.state != "DISMISSED" or last_bot_review.state != "COMMENTED"):
                logger.info(f"New commits detected since last bot review (last: {last_bot_review.commit_id}, current: {current_head_sha}). Dismissing old review.")
                dismiss_message = f"Dismissing old review as new commits have been pushed. Current head: {current_head_sha}."
                if github_h.dismiss_review(last_bot_review, dismiss_message):
                    logger.info(f"Successfully dismissed previous review ID {last_bot_review.id}.")
                    # Attempt to delete the line comments from the dismissed review
                    logger.info(f"Attempting to delete line comments from dismissed review ID {last_bot_review.id}.")
                    if github_h.delete_review_line_comments(pr_number=pr_number, review_id=last_bot_review.id):
                        logger.info(f"Successfully initiated deletion of line comments for review ID {last_bot_review.id}.")
                    else:
                        logger.warning(f"Failed to initiate deletion of line comments for review ID {last_bot_review.id}. Some old comments might remain visible.")
                else:
                    logger.warning(f"Failed to dismiss previous review ID {last_bot_review.id} or it was already dismissed.")
            else:
                if last_bot_review.state == "DISMISSED":
                    logger.info("Last bot review was dismissed. Proceeding with new review.")
                elif last_bot_review.state == "COMMENTED":
                    logger.info("Last bot review was a comment which the Git api does not allow to be dismissed. Proceeding with new review.")
                else:
                    logger.info("No new commits since last bot review. A new review might duplicate comments unless content has changed significantly.")
        else:
            logger.info("No previous reviews by this bot found for this PR.")

        # --- Fetch PR Data --- 
        logger.info(f"Fetching changed files and diff for PR #{pr_number}...")
        # We need the combined diff for overall analysis and summarization.
        # Changed files list can be used if we want to iterate or provide per-file feedback later.
        pr_diff_content = github_h.get_pr_diff(pr_number=pr_number)
        changed_files = github_h.get_pr_changed_files(pr_number=pr_number)

        if not pr_diff_content:
            logger.info(f"No diff content found for PR #{pr_number}. This might be an empty PR or only non-code changes. Exiting gracefully.")
            # Potentially post a comment saying no changes to analyze, or just exit.
            # github_h.post_pr_review(pr_number, "AI Review: No code changes detected to analyze.")
            sys.exit(0)

        logger.info(f"Successfully fetched diff for PR #{pr_number}. Diff length: {len(pr_diff_content)} chars.")

        # --- Perform Heavy Analysis (needed for Release Notes first, then detailed review) ---
        heavy_analysis = None
        try:
            logger.info("Performing initial AI-driven heavy analysis (for release notes and detailed review comment)...")
            heavy_analysis = analysis_s.analyze_code_changes(diff_content=pr_diff_content)
            if heavy_analysis:
                logger.info(f"Initial heavy analysis completed. Output preview: {str(heavy_analysis)[:100]}...")
            else:
                logger.warning("Initial heavy analysis via analyze_code_changes produced no output.")
        except Exception as e:
            logger.error(f"Error during initial heavy_analysis stage: {e}. Release notes and detailed review might be affected.")

        # --- Generate and Update Release Notes in PR Description ---
        if heavy_analysis: # Only proceed if heavy_analysis was successful and produced output
            try:
                logger.info("Generating non-technical release notes from the heavy analysis...")
                # generate_release_notes expects a string input (which was diff_content, now heavy_analysis output)
                release_notes_summary_text = analysis_s.generate_release_notes(diff_content=heavy_analysis)
                if release_notes_summary_text:
                    logger.info(f"Release notes generated: {release_notes_summary_text[:100]}...")
                    logger.info(f"Attempting to update PR #{pr_number} description with these release notes...")
                    if github_h.update_pr_description(pr_number=pr_number, release_notes_summary=release_notes_summary_text):
                        logger.info(f"PR #{pr_number} description updated successfully with release notes.")
                    else:
                        logger.warning(f"Failed to update PR #{pr_number} description with release notes (handler returned False or no update made).")
                else:
                    logger.warning("Release notes generation (from heavy_analysis) produced no output.")
            except Exception as e:
                logger.error(f"Error occurred during release notes generation or PR description update: {e}")
        else:
            logger.info("Skipping release notes generation and PR description update because initial heavy analysis failed or was empty.")

        # --- Perform Analysis for the Review Comment Body ---
        logger.info("Performing AI analysis for the main review comment body...")
        summary_for_review_comment = ""
        try:
            summary_for_review_comment = analysis_s.summarize_changes(diff_content=pr_diff_content) # Uses original diff
            logger.info(f"Summary for review comment generated: {summary_for_review_comment[:100]}...")
        except Exception as e:
            logger.error(f"Error generating summary for review comment: {e}. An empty summary will be used.")
        
        # Refined analysis for line comments (uses the heavy_analysis from above)
        refined_analysis_results = None 
        if heavy_analysis: # Reuse the heavy_analysis from before
            try:
                logger.info("Refining heavy analysis to extract actionable line-specific suggestions...")
                refined_analysis_results = analysis_s.analyze_heavy_model_output(
                    heavy_model_output=heavy_analysis, # Output from analyze_code_changes
                    diff_content=pr_diff_content # Original diff for context
                )
                if refined_analysis_results:
                    logger.info(f"Refined analysis for line comments completed. Found {len(refined_analysis_results)} actionable suggestions.")
                else:
                    logger.info("Refined analysis for line comments did not produce actionable suggestions.")
            except Exception as e:
                logger.error(f"Error during refined analysis (analyze_heavy_model_output): {e}. Line comments may be missing.")
        else:
            logger.info("Skipping refined analysis for line comments as initial heavy analysis was not available or failed.")
        
        # --- Perform Individual File Analysis ---
        individual_file_analyses: Dict[str, str] = {}
        if changed_files: # Ensure changed_files is populated
            logger.info(f"Starting individual analysis for {len(changed_files)} changed files...")
            for file_info in changed_files:
                filename = file_info.get("filename")
                file_patch = file_info.get("patch") # This is the diff for the individual file
                
                if filename and file_patch:
                    try:
                        logger.info(f"Performing analysis for file: {filename}")
                        file_analysis_result = analysis_s.analyze_individual_file_diff(file_patch=file_patch, filename=filename)
                        if file_analysis_result:
                            individual_file_analyses[filename] = file_analysis_result
                            logger.info(f"Completed analysis for file: {filename}. Result length: {len(file_analysis_result)}")
                        else:
                            logger.info(f"Analysis for file: {filename} returned no content.")
                    except Exception as e:
                        logger.error(f"Error during analysis of file {filename}: {e}. This file's analysis will be skipped.")
                        individual_file_analyses[filename] = f"Could not analyze {filename} due to an error: {str(e)}"
                elif filename and not file_patch:
                    logger.info(f"Skipping analysis for file {filename} as it has no patch content (e.g., binary, renamed, or mode change only). ")        
        else:
            logger.info("No changed files data available to perform individual file analysis.")

        # --- Prepare Line-Specific Comments ---
        line_specific_comments_for_review = []
        if refined_analysis_results and isinstance(refined_analysis_results, list):
            for suggestion_item in refined_analysis_results:
                if isinstance(suggestion_item, dict) and \
                   all(k in suggestion_item for k in ("file_path", "line", "suggestion")):
                    line_specific_comments_for_review.append({
                        "path": suggestion_item["file_path"],
                        "line": suggestion_item["line"],
                        "body": suggestion_item["suggestion"]
                    })
                else:
                    logger.warning(f"Skipping malformed suggestion item: {suggestion_item}")
        
        if line_specific_comments_for_review:
            logger.info(f"Prepared {len(line_specific_comments_for_review)} line-specific comments for the review.")

        # --- Generate Review Body & Post Review ---
        logger.info("Generating main review body...")
        # The refined_analysis parameter for generate_review_body might need adjustment
        # if it was originally expecting a single string. For now, we pass None if we have line comments,
        # or the raw heavy_analysis if refined_analysis_results is empty.
        # The line comments will be handled separately by post_pr_review.
        
        # If refined_analysis_results were processed into line_comments, 
        # we might not need to pass them to generate_review_body in the same way.
        # Let's assume generate_review_body primarily uses summary and can optionally include a textual version of analysis.
        # The overflow of line comments is now handled by github_handler.post_pr_review.
        
        review_body_text = analysis_s.generate_review_body(
            summary=summary_for_review_comment,
            refined_analysis=heavy_analysis, # Pass the raw heavy_analysis string for the detailed section
            heavy_analysis_raw=heavy_analysis if not refined_analysis_results else None 
        )

        # Append individual file analyses to the review body
        if individual_file_analyses:
            review_body_text += "\n\n---\n### Per-File Analysis Details\n"
            for filename, analysis_text in individual_file_analyses.items():
                review_body_text += f"\n#### File: `{filename}`\n{analysis_text}\n"

        logger.info(f"Posting review to PR #{pr_number} on commit {current_head_sha}...")
        
        # The github_h.post_pr_review will now handle BOT_SIGNATURE and line comment formatting.
        posted_review = github_h.post_pr_review(
            pr_number=pr_number,
            review_body=review_body_text,
            commit_id=current_head_sha,
            event= "REQUEST_CHANGES" if len(line_specific_comments_for_review) > 0 else "APPROVE", 
            line_comments=line_specific_comments_for_review
        )

        if posted_review:
            logger.info(f"Review successfully posted. Review ID: {posted_review.id}")
        else:
            logger.warning("Review posting did not return a review object (possibly failed or an issue).")

    except ValueError as ve:
        logger.critical(f"Configuration or input validation error: {ve}")
        sys.exit(1)
    except RuntimeError as rte:
        logger.critical(f"Runtime error during PR review process: {rte}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("AI Pull Request Code Reviewer finished.")

if __name__ == "__main__":
    # For local testing or direct execution
    main() 