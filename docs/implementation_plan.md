# Implementation Plan: AI-Powered Pull Request Code Reviewer

## 1. Overview

This document outlines the plan to create a reusable GitHub workflow that leverages AWS Bedrock language models to analyze pull requests. The workflow will provide a summary of changes and a deeper analysis of the code modifications.

## 2. Goals

*   Create a reusable GitHub workflow.
*   Integrate with AWS Bedrock for AI-driven code analysis.
*   Use `PyGithub` for GitHub API interactions.
*   Follow SOLID principles and Python best practices.
*   Include a comprehensive test suite.
*   Provide clear documentation for usage.

## 3. Core Components

### 3.1. Reusable GitHub Workflow (`.github/workflows/pr_code_review.yml`)

*   **Trigger:** `on: workflow_call`
*   **Inputs:**
    *   `github_token`: (Required) `secrets.GITHUB_TOKEN` for authenticating API calls.
    *   `aws_access_key_id`: (Required) AWS access key ID.
    *   `aws_secret_access_key`: (Required) AWS secret access key.
    *   `aws_region`: (Required) AWS region for Bedrock.
    *   `heavy_model_id`: (Required, default from calling repo's secrets or vars) ID for the Bedrock model for deep code analysis.
    *   `light_model_id`: (Required, default from calling repo's secrets or vars) ID for the Bedrock model for summarizing changes.
    *   `deepseek_model_id`: (Required, default from calling repo's secrets or vars) ID for the Bedrock model to analyze the heavy model's output.
    *   `calling_repo_token`: (Optional) A token with permissions to the calling repository if `GITHUB_TOKEN` is not sufficient (e.g., for posting comments from a separate app).
*   **Environment:** Runs on `ubuntu-latest`.
*   **Steps:**
    1.  Checkout the pull request code from the calling repository.
    2.  Checkout this reviewer workflow's repository code.
    3.  Set up Python environment.
    4.  Install Python dependencies (`requirements.txt`).
    5.  Set environment variables for AWS credentials and model IDs from workflow inputs/secrets.
        *   `HEAVY_MODEL_ENV_VAR`: `inputs.heavy_model_id`
        *   `LIGHT_MODEL_ENV_VAR`: `inputs.light_model_id`
        *   `DEEPSEEK_MODEL_ENV_VAR`: `inputs.deepseek_model_id`
        *   `AWS_ACCESS_KEY_ID`: `inputs.aws_access_key_id`
        *   `AWS_SECRET_ACCESS_KEY`: `inputs.aws_secret_access_key`
        *   `AWS_DEFAULT_REGION`: `inputs.aws_region`
        *   `GITHUB_EVENT_PATH`: Path to the event payload JSON.
        *   `GITHUB_TOKEN_ENV_VAR`: `inputs.github_token`
        *   `CALLING_REPO_TOKEN_ENV_VAR`: `inputs.calling_repo_token` (if provided)
    6.  Execute the Python application (`src/main.py`).

### 3.2. Python Application (`src/`)

*   **`main.py`**:
    *   Entry point of the application.
    *   Parses GitHub event payload to get PR details (PR number, repository owner/name, changed files, diff).
    *   Retrieves AWS credentials and model IDs from environment variables.
    *   Orchestrates the analysis process.
    *   Handles error reporting and graceful exit.
*   **`github_handler.py`**:
    *   Initializes `PyGithub` with the appropriate token.
    *   Fetches PR details:
        *   List of changed files.
        *   Diff for each changed file.
    *   Posts a review comment to the PR with the analysis results.
*   **`bedrock_handler.py`**:
    *   Initializes `boto3` Bedrock client.
    *   Contains a generic function to invoke a Bedrock model (given model ID and prompt).
    *   Handles Bedrock API errors.
*   **`analysis_service.py`**:
    *   **`analyze_code_changes(diff_content, model_id)`**:
        *   Constructs a prompt for the `HEAVY_MODEL` using the diff content.
        *   Invokes `bedrock_handler.invoke_model`.
        *   Returns the raw analysis from the heavy model.
    *   **`analyze_heavy_model_output(heavy_model_output, model_id)`**:
        *   Constructs a prompt for the `DEEPSEEK_MODEL` using the output from the `HEAVY_MODEL`.
        *   Invokes `bedrock_handler.invoke_model`.
        *   Returns the refined analysis.
    *   **`summarize_changes(diff_content, model_id)`**:
        *   Constructs a prompt for the `LIGHT_MODEL` using the diff content.
        *   Invokes `bedrock_handler.invoke_model`.
        *   Returns a concise summary.
    *   **`generate_review_body(summary, deepseek_analysis)`**:
        *   Formats the summary and the refined analysis into a presentable review body (Markdown).
*   **`config.py`**:
    *   Loads and provides access to configuration (e.g., model IDs from environment variables).
    *   Could use `python-dotenv` for local development to load from a `.env` file (this `.env` will be in the root of *this* project, not the calling project, and should be in `.gitignore`).
*   **`utils.py` (Optional):**
    *   Common utility functions (e.g., logging setup).

### 3.3. Tests (`tests/`)

*   **`test_github_handler.py`**: Unit tests for `github_handler.py`, mocking `PyGithub`.
*   **`test_bedrock_handler.py`**: Unit tests for `bedrock_handler.py`, mocking `boto3` Bedrock client.
*   **`test_analysis_service.py`**: Unit tests for `analysis_service.py`, mocking `bedrock_handler.py` calls.
*   **`test_main.py`**: Integration-style tests for `main.py`, mocking external services and environment variables.
*   Fixtures for sample PR data, diffs, and Bedrock responses.
*   Test runner: `pytest`.

## 4. Project Structure

```
pull-request-code-reviewer/
├── .github/
│   └── workflows/
│       └── pr_code_review.yml
├── .env              # For local development (contains dummy or dev AWS keys & model IDs, in .gitignore)
├── .gitignore
├── docs/
│   └── implementation_plan.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── github_handler.py
│   ├── bedrock_handler.py
│   ├── analysis_service.py
│   ├── config.py
│   └── utils.py (optional)
├── tests/
│   ├── __init__.py
│   ├── fixtures/       # Sample data for tests
│   │   ├── sample_diff.txt
│   │   └── sample_event.json
│   ├── test_github_handler.py
│   ├── test_bedrock_handler.py
│   ├── test_analysis_service.py
│   └── test_main.py
├── requirements.txt  # Python dependencies (PyGithub, boto3, python-dotenv, pytest)
└── README.md
```

## 5. Dependencies

*   `PyGithub`: For GitHub API interaction.
*   `boto3`: For AWS SDK, specifically Bedrock.
*   `python-dotenv`: For loading `.env` files during local development.
*   `pytest`: For running tests.

## 6. Secrets and Configuration Management

*   **GitHub Workflow:**
    *   AWS Credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) will be passed as secrets to the reusable workflow from the calling repository.
    *   Bedrock Model IDs (`HEAVY_MODEL_ID`, `LIGHT_MODEL_ID`, `DEEPSEEK_MODEL_ID`) will also be passed as secrets or workflow variables from the calling repository.
    *   `GITHUB_TOKEN` is automatically available.
*   **Python Application:**
    *   Reads credentials and model IDs from environment variables set by the GitHub Actions runner.
    *   For local development, uses `python-dotenv` to load these from a local `.env` file (which should be gitignored).

## 7. Development Workflow

1.  **Setup Project Structure:** Create directories and initial empty files.
2.  **Implement `config.py`**: Handle loading of environment variables.
3.  **Implement `bedrock_handler.py`**: Core Bedrock interaction logic. Write unit tests.
4.  **Implement `github_handler.py`**: Core GitHub interaction logic. Write unit tests.
5.  **Implement `analysis_service.py`**: Logic for prompting and processing model outputs. Write unit tests.
6.  **Implement `main.py`**: Orchestrate the workflow. Write integration tests.
7.  **Develop `pr_code_review.yml`**: Define the reusable workflow.
8.  **Testing:**
    *   Run unit tests locally using `pytest`.
    *   Manually test the workflow with a test repository.
9.  **Documentation:** Update `README.md` with usage instructions, setup, and examples.

## 8. SOLID Principles and Best Practices

*   **Single Responsibility Principle (SRP):** Each module/class will have a distinct responsibility (e.g., `github_handler.py` for GitHub interactions, `bedrock_handler.py` for Bedrock).
*   **Open/Closed Principle (OCP):** Design modules to be extensible (e.g., adding new analysis types) without modifying existing code where possible.
*   **Dependency Inversion Principle (DIP):** High-level modules will depend on abstractions, not concrete implementations (e.g., `main.py` uses handlers).
*   **Type Hinting:** Use Python type hints for better code clarity and maintainability.
*   **Error Handling:** Implement robust error handling and logging.
*   **Modularity:** Break down the application into smaller, manageable modules.
*   **Testing:** Comprehensive unit and integration tests.
*   **Code Style:** Adhere to PEP 8.

## 9. Future Enhancements (Optional)

*   Support for other LLM providers.
*   More sophisticated prompt engineering.
*   Caching Bedrock responses for identical diffs.
*   Configuration for analysis depth or focus areas.
*   Interactive feedback loop (e.g., allow users to ask follow-up questions to the AI).

This plan should provide a solid foundation for building the AI-powered pull request code reviewer. 