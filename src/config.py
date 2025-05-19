import os
from typing import Optional

from dotenv import load_dotenv

# Load .env file for local development
# In a GitHub Action, environment variables will be set directly.
load_dotenv()

# AWS Bedrock Model IDs
HEAVY_MODEL_ID: Optional[str] = os.getenv("HEAVY_MODEL_ID")
LIGHT_MODEL_ID: Optional[str] = os.getenv("LIGHT_MODEL_ID")
DEEPSEEK_MODEL_ID: Optional[str] = os.getenv("DEEPSEEK_MODEL_ID")

# AWS Credentials
AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME: Optional[str] = os.getenv("AWS_DEFAULT_REGION") # boto3 uses AWS_DEFAULT_REGION

# GitHub Variables
GITHUB_EVENT_PATH: Optional[str] = os.getenv("GITHUB_EVENT_PATH")
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
# This is the token for the repository dispatching the workflow_call event or the GITHUB_TOKEN of the reusable workflow itself.

CALLING_REPO_TOKEN: Optional[str] = os.getenv("CALLING_REPO_TOKEN")
# Optional: A separate token for actions in the calling repository if GITHUB_TOKEN is insufficient.

# Bot Configuration
BOT_NAME: Optional[str] = os.getenv("BOT_NAME", "github-actions[bot]") # Default to common GH Actions bot name

def get_required_env_var(var_name: str) -> str:
    """Retrieve a required environment variable or raise an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Required environment variable '{var_name}' is not set.")
    return value
