import os
from typing import Optional
import re

from dotenv import load_dotenv

# Load .env file for local development
# In a GitHub Action, environment variables will be set directly.
load_dotenv()

# AWS Bedrock Model IDs
HEAVY_MODEL_ID: Optional[str] = os.getenv("HEAVY_MODEL_ID")
LIGHT_MODEL_ID: Optional[str] = os.getenv("LIGHT_MODEL_ID")
DEEPSEEK_MODEL_ID: Optional[str] = os.getenv("DEEPSEEK_MODEL_ID", "")

# AWS Credentials
AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME: Optional[str] = os.getenv("AWS_DEFAULT_REGION") # boto3 uses AWS_DEFAULT_REGION

# GitHub Variables
GITHUB_EVENT_PATH: Optional[str] = os.getenv("GITHUB_EVENT_PATH")
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
# This is the token for the repository dispatching the workflow_call event or the GITHUB_TOKEN of the reusable workflow itself.

# Bot Configuration
BOT_NAME: Optional[str] = os.getenv("BOT_NAME", "github-actions[bot]") # Default to common GH Actions bot name

def get_required_env_var(var_name: str) -> str:
    """Retrieve a required environment variable or raise an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Required environment variable '{var_name}' is not set.")
    return value

# Helper function to sanitize model ARNs from error messages
def sanitize_model_arn_in_message(message: str) -> str:
    """
    Replaces AWS Bedrock model ARNs and inference profile ARNs in a string 
    with a placeholder to avoid exposing them.
    It also tries to catch model IDs if they are not full ARNs but are used in errors.
    """
    if not message: # Ensure message is not None
        return ""

    # Regex to find Bedrock model ARNs and inference profile ARNs
    # Covers: arn:aws:bedrock:<region>:<account>:model/<model_name>
    # Covers: arn:aws:bedrock:<region>:<account>:inference-profile/<profile_name_with_model>
    # Covers: <region>.<provider>.<model_name> (common for inference profile IDs used as model_id)
    # Example: arn:aws:bedrock:us-east-1:123456789012:model/anthropic.claude-v2
    # Example: arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0
    # Example: us.anthropic.claude-3-7-sonnet-20250219-v1:0 (often used as modelId in API calls)

    # More general pattern for ARNs that include model IDs
    arn_pattern = r"arn:aws:bedrock:[^:]+:[^:]+:(model|inference-profile)/[^\s,\)\]]+"
    # Pattern for shorter model IDs like provider.model_name or region.provider.model_name
    short_model_id_pattern = r"[a-zA-Z0-9.-]+/[a-zA-Z0-9.-]+:[^\s,\)\]]*|[a-zA-Z0-9.-]+\.[a-zA-Z0-9.-]+\.[a-zA-Z0-9.-]+[^\s,\)\]]*"
    
    # More robust: replace known model IDs from config too, if they appear.
    # This helps if the error message only contains the model ID and not the full ARN.
    placeholder = "[MODEL_ID_REDACTED]"
    
    sanitized_message = re.sub(arn_pattern, placeholder, message)
    # Apply short model ID pattern after ARN pattern to catch remnants or different formats
    sanitized_message = re.sub(short_model_id_pattern, placeholder, sanitized_message, flags=re.IGNORECASE) 

    # Additionally, explicitly replace configured model IDs if they are present
    # This ensures that if an error message *only* contains the model ID string, it's caught.
    known_model_ids = [HEAVY_MODEL_ID, LIGHT_MODEL_ID, DEEPSEEK_MODEL_ID]
    for model_id_val in known_model_ids:
        if model_id_val and model_id_val in sanitized_message: # Check if model_id_val is not empty
            # Escape the model_id_val in case it contains regex special characters
            sanitized_message = sanitized_message.replace(model_id_val, placeholder)
            
    return sanitized_message
