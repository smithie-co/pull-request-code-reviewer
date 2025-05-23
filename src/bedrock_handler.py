import json
import logging
from typing import Any, Dict, Optional
import time # Added for sleep
import random # Added for jitter

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from src import config
from src.rate_limiter import get_global_rate_limiter
from src.token_calculator import TokenCalculator

# Configure logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Moved basicConfig to main.py to avoid multiple calls if imported elsewhere.
# Ensure logger has handlers if this module is run standalone or tested directly without main.py setup.
if not logger.handlers:
    # Fallback basic config if no handlers are configured.
    # This might be noisy if other modules also configure root logger.
    # A dedicated logging setup module/function is better for larger apps.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BedrockHandler:
    """Handles interactions with AWS Bedrock models."""

    def __init__(self,
                 aws_access_key_id: Optional[str] = None, # Default to None
                 aws_secret_access_key: Optional[str] = None, # Default to None
                 aws_region_name: Optional[str] = None): # Default to None
        """
        Initializes the Bedrock client.
        Credentials and region are sourced from arguments or src.config if arguments are None.

        Args:
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            aws_region_name: AWS region name.

        Raises:
            ValueError: If AWS credentials or region are not properly configured or provided.
            RuntimeError: For other Boto3/AWS SDK client initialization errors.
        """
        # Fetch from config if not provided in args
        _aws_access_key_id = aws_access_key_id if aws_access_key_id is not None else config.AWS_ACCESS_KEY_ID
        _aws_secret_access_key = aws_secret_access_key if aws_secret_access_key is not None else config.AWS_SECRET_ACCESS_KEY
        _aws_region_name = aws_region_name if aws_region_name is not None else config.AWS_REGION_NAME

        if not all([_aws_access_key_id, _aws_secret_access_key, _aws_region_name]):
            msg = (
                "AWS credentials or region not fully configured. "
                "Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION "
                "are set in environment variables, .env file, or passed to BedrockHandler."
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=_aws_region_name,
                aws_access_key_id=_aws_access_key_id,
                aws_secret_access_key=_aws_secret_access_key
            )
            logger.info(f"Bedrock client initialized for region: {_aws_region_name}")
        except (NoCredentialsError, PartialCredentialsError) as e:
            # These errors indicate issues with the credentials themselves, even if paths/vars were found.
            logger.error(f"AWS SDK credentials error during Bedrock client initialization: {e}")
            raise ValueError(f"AWS SDK credentials error: {e}") from e
        except ClientError as e: # Catch other Boto3 client errors during init e.g. invalid region
            logger.error(f"Boto3 ClientError during Bedrock client initialization: {e}")
            raise RuntimeError(f"Failed to initialize Bedrock client due to Boto3 ClientError: {e}") from e
        except Exception as e: # Catch any other unexpected exceptions during init
            logger.error(f"Unexpected error during Bedrock client initialization: {e}")
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}") from e

    def _create_model_body(self, model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]) -> Dict[str, Any]:
        """Helper to create the request body based on model provider."""
        body_params: Dict[str, Any]

        # Check for Claude 3+ models (including Claude 4) that require Messages API
        if ("anthropic.claude-3" in model_id or 
            "claude-3" in model_id or 
            "claude-4" in model_id or 
            "claude-sonnet-4" in model_id or
            "claude-haiku-3" in model_id or
            "claude-opus-3" in model_id or
            "claude-opus-4" in model_id): # Handle Claude 3+ models with Messages API
            # Extract user content from the "Human: ... Assistant: ..." prompt format
            user_content = prompt
            if prompt.startswith("Human: "):
                user_content_start = len("Human: ")
                # Find the last occurrence of Assistant cue in case of complex prompts
                assistant_cue_index = prompt.rfind("\\n\\nAssistant:")
                if assistant_cue_index != -1:
                    user_content = prompt[user_content_start:assistant_cue_index].strip()
                else:
                    user_content = prompt[user_content_start:].strip()
            else:
                # If prompt doesn't follow Human:/Assistant: format, use as is, but log a warning.
                logger.warning(f"Prompt for Claude 3+ model '{model_id}' does not start with 'Human: '. Using the entire prompt as user content.")
                user_content = prompt.strip()
            
            body_params = {
                "anthropic_version": "bedrock-2023-05-31", # Required for Claude 3+
                "max_tokens": max_tokens, # Claude 3+ uses "max_tokens"
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_content}]
                    }
                ]
            }
            # Optional parameters for Claude 3+ Messages API
            if temperature is not None: body_params["temperature"] = temperature
            if top_p is not None: body_params["top_p"] = top_p
            if top_k is not None: body_params["top_k"] = top_k
            # System prompt can be added here if needed:
            # system_prompt_text = "You are a helpful coding assistant." # Example
            # if system_prompt_text:
            #     body_params["system"] = system_prompt_text

        elif "anthropic.claude" in model_id: # Older Claude models (v1, v2, instant)
            body_params = {
                "prompt": prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if top_k is not None: # Claude supports top_k
                body_params["top_k"] = top_k
            # "stop_sequences": ["\n\nHuman:"] # Example for Claude
        elif "amazon.titan" in model_id:
            body_params = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                    # "stopSequences": [] # Example for Titan
                }
            }
        elif "ai21.j2" in model_id: # Jurassic
            body_params = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
            if top_k is not None:
                 body_params["topKReturn"] = 0 # Not directly top_k, but related to sampling
                 body_params["numResults"] = 1 # Usually 1
        elif "cohere.command" in model_id: # Cohere
            body_params = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "p": top_p, # Cohere uses 'p' for top_p
            }
            if top_k is not None:
                body_params["k"] = top_k # Cohere uses 'k' for top_k
        elif "deepseek" in model_id.lower():
            # DeepSeek uses a specific prompt format and text completion style API
            # The prompt passed to this function should be the raw user query.
            # IMPORTANT: The prompt for DeepSeek needs to be stripped of "Human: " and "Assistant: " if present,
            # as those are part of its special token sequence.
            clean_prompt = prompt
            if prompt.startswith("Human: "):
                clean_prompt = prompt[len("Human: "):]
            if "\n\nAssistant:" in clean_prompt: # Check before stripping, to handle prompts not ending with it
                assistant_cue_index = clean_prompt.rfind("\n\nAssistant:")
                clean_prompt = clean_prompt[:assistant_cue_index]
            
            formatted_prompt = f"<｜begin of sentence｜> {clean_prompt.strip()} <｜Assistant｜><think>\n\n"
            body_params = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens, # DeepSeek uses "max_tokens"
                "temperature": temperature,
                "top_p": top_p,
            }
            # DeepSeek also supports a "stop" parameter (array of strings), not currently used here.
            # e.g. "stop": ["<｜end of sentence｜>"]
        elif "meta.llama" in model_id.lower(): # Llama2 also Claude-like (for text completion style)
            # Note: Llama 3 might prefer a messages-like API too. This might need further generalization.
            logger.warning(f"Using Anthropic Claude (legacy Text Completions) like structure for model: {model_id}. This may need adjustment.")
            body_params = {
                "prompt": prompt, 
                "max_tokens_to_sample": max_tokens, # Assumption
                "temperature": temperature,
                "top_p": top_p,
            }
            if top_k is not None:
                 body_params["top_k"] = top_k
        else:
            logger.warning(f"Model family for '{model_id}' not explicitly handled. Using a generic Claude-like body structure. This may fail or produce suboptimal results.")
            body_params = {
                "prompt": prompt,
                "max_tokens_to_sample": max_tokens, # Defaulting to Claude's token parameter name
                "temperature": temperature,
                "top_p": top_p,
            }
            if top_k is not None:
                 body_params["top_k"] = top_k
        return body_params

    def _extract_output_text(self, model_id: str, response_body_json: Dict[str, Any]) -> Optional[str]:
        """Helper to extract the generated text from the model's response body."""
        generated_text: Optional[str] = None

        # Check for Claude 3+ models (including Claude 4) that use Messages API response
        if ("anthropic.claude-3" in model_id or 
            "claude-3" in model_id or 
            "claude-4" in model_id or 
            "claude-sonnet-4" in model_id or
            "claude-haiku-3" in model_id or
            "claude-opus-3" in model_id): # Handle Claude 3+ Messages API response
            content_blocks = response_body_json.get('content', [])
            if content_blocks and isinstance(content_blocks, list) and len(content_blocks) > 0:
                # Assuming the first block is the text response we want
                first_block = content_blocks[0]
                if isinstance(first_block, dict) and first_block.get("type") == "text":
                    generated_text = first_block.get('text')
                else:
                    logger.warning(f"First content block for Claude 3+ model {model_id} is not a text block or is malformed: {first_block}")
            else:
                logger.warning(f"No content blocks found in Claude 3+ model {model_id} response or content is not a list: {response_body_json.get('content')}")

        elif "anthropic.claude" in model_id or \
           ("deepseek" in model_id.lower() and "completion" in response_body_json) or \
           ("meta.llama" in model_id.lower() and "generation" in response_body_json): # Llama2 specific
            if "completion" in response_body_json: # Claude (legacy), some older non-Claude with this key
                generated_text = response_body_json.get('completion')
            elif "generation" in response_body_json: # Llama2
                generated_text = response_body_json.get('generation')
        elif "amazon.titan" in model_id:
            # Titan response can have multiple results, typically we take the first.
            results = response_body_json.get('results', [])
            if results and isinstance(results, list) and len(results) > 0:
                generated_text = results[0].get('outputText')
            else: # Fallback for older Titan structures or if results is empty
                generated_text = response_body_json.get('outputText') 
        elif "ai21.j2" in model_id: # Jurassic
            completions = response_body_json.get('completions', [])
            if completions and isinstance(completions, list) and len(completions) > 0:
                generated_text = completions[0].get('data', {}).get('text')
        elif "cohere.command" in model_id: # Cohere
            generations = response_body_json.get('generations', [])
            if generations and isinstance(generations, list) and len(generations) > 0:
                generated_text = generations[0].get('text')
        elif "deepseek" in model_id.lower(): # Specific extraction for DeepSeek
            choices = response_body_json.get('choices', [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    generated_text = first_choice.get('text')
                else:
                    logger.warning(f"First choice in DeepSeek response is not a dict: {first_choice}")
            else:
                logger.warning(f"No 'choices' or empty/malformed 'choices' array found in DeepSeek model {model_id} response: {response_body_json}")
        else:
            # Attempt a generic extraction if model family is unknown or has multiple possible keys
            possible_keys = ['completion', 'generated_text', 'text', 'outputText', 'generation']
            for key in possible_keys:
                if response_body_json.get(key) and isinstance(response_body_json.get(key), str):
                    generated_text = response_body_json.get(key)
                    logger.info(f"Used generic key '{key}' to extract output for model {model_id}.")
                    break
            if not generated_text:
                logger.warning(f"Could not find a standard output key for model {model_id}. Full response body: {response_body_json}")
                # Fallback to string representation if no known key works, as a last resort.
                generated_text = str(response_body_json) if isinstance(response_body_json, dict) else str(response_body_json)

        return generated_text

    def invoke_model(self, model_id: str, prompt: str, max_tokens: Optional[int] = None,
                     temperature: float = 0.7, top_p: float = 1.0, top_k: Optional[int] = None,
                     max_retries: int = 3, initial_backoff_seconds: float = 1.0,
                     analysis_type: str = 'heavy_analysis') -> str:
        """
        Invokes a Bedrock model with the given prompt and parameters, with retry logic and rate limiting.

        Args:
            model_id: The ID of the Bedrock model to invoke.
            prompt: The prompt to send to the model.
            max_tokens: The maximum number of tokens to generate (calculated dynamically if None).
            temperature: Controls randomness. Lower for more factual, higher for creative.
            top_p: Nucleus sampling. Consider adjusting with temperature.
            top_k: Top-k sampling. Supported by some models (e.g. Claude, Cohere).
            max_retries: Maximum number of retry attempts for retryable errors.
            initial_backoff_seconds: Initial wait time for exponential backoff.
            analysis_type: Type of analysis for dynamic token calculation.

        Returns:
            The text generated by the model.

        Raises:
            ValueError: If model_id or prompt is empty, or for validation errors from Bedrock.
            RuntimeError: If the model invocation fails or returns an unexpected response.
        """
        if not model_id:
            logger.error("Model ID cannot be empty for invoke_model.")
            raise ValueError("Model ID cannot be empty.")
        if not prompt:
            logger.error("Prompt cannot be empty for invoke_model.")
            raise ValueError("Prompt cannot be empty.")

        # Dynamic token calculation if max_tokens not provided
        if max_tokens is None:
            context_window = TokenCalculator.get_model_context_limits(model_id)
            max_tokens = TokenCalculator.calculate_dynamic_max_tokens(
                input_content=prompt,
                analysis_type=analysis_type,
                model_context_window=context_window
            )
            logger.info(f"Dynamically calculated max_tokens for {analysis_type}: {max_tokens}")

        # Get rate limiter
        rate_limiter = get_global_rate_limiter()
        
        logger.info(f"Invoking Bedrock model: {model_id} with prompt (first 80 chars): {prompt[:80].replace('\n', ' ')}...")
        logger.debug(f"Full prompt for {model_id}:\n{prompt}")
        logger.debug(f"Invocation params for {model_id}: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")

        attempt = 0
        while attempt <= max_retries:
            try:
                # Acquire rate limit token before making request
                if not rate_limiter.acquire(timeout=30):  # 30 second timeout
                    logger.warning(f"Rate limiter timeout waiting for token for model {model_id}")
                    raise RuntimeError("Rate limiter timeout - too many requests")

                body_params = self._create_model_body(model_id, prompt, max_tokens, temperature, top_p, top_k)
                body = json.dumps(body_params)
                logger.debug(f"Request body for {model_id} (attempt {attempt + 1}): {body}")

                response = self.client.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )

                response_body_raw = response.get('body').read()
                try:
                    response_body_json = json.loads(response_body_raw)
                    logger.debug(f"Response body JSON for {model_id}: {response_body_json}")
                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to decode JSON response from {model_id}. Raw response: {response_body_raw[:500]}... Error: {json_e}")
                    # This is not a retryable Bedrock error, but a response format issue.
                    raise RuntimeError(f"Failed to decode JSON response from {model_id}. Check logs for details.") from json_e
                
                generated_text = self._extract_output_text(model_id, response_body_json)

                if generated_text is None or not isinstance(generated_text, str):
                    logger.error(f"Model {model_id} invocation succeeded but no valid string text found. Parsed as: {generated_text}. Full response JSON: {response_body_json}")
                    # This is not a retryable Bedrock error, but a response content issue.
                    raise RuntimeError(f"Model {model_id} returned no valid string text. Check logs. Full response: {response_body_json}")

                logger.info(f"Model {model_id} invocation successful. Output (first 80 chars): {generated_text[:80].replace('\n', ' ')}...")
                return generated_text.strip()

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                error_message = e.response.get('Error', {}).get('Message')
                
                retryable_errors = ['ThrottlingException', 'ModelTimeoutException', 'ServiceUnavailableException', 'InternalServerException']
                # 'ModelNotReadyException' is also mentioned by AWS SDK to be auto-retried, but good to list if we handle it manually.

                if error_code in retryable_errors and attempt < max_retries:
                    # Special handling for token-based throttling
                    if error_code == 'ThrottlingException' and 'tokens' in error_message.lower():
                        # Token-based throttling needs longer delays
                        wait_time = (initial_backoff_seconds * (3 ** attempt)) + (random.uniform(0, 2) * initial_backoff_seconds)
                        wait_time = min(wait_time, 60)  # Cap at 60 seconds
                        logger.warning(f"Bedrock API token-based {error_code} for model {model_id} (Attempt {attempt + 1}/{max_retries + 1}). Using extended backoff: {wait_time:.2f} seconds... Error: {error_message}")
                    else:
                        # Standard exponential backoff for other retryable errors
                        wait_time = (initial_backoff_seconds * (2 ** attempt)) + (random.uniform(0, 1) * initial_backoff_seconds)
                        logger.warning(f"Bedrock API {error_code} for model {model_id} (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_time:.2f} seconds... Error: {error_message}")
                    
                    time.sleep(wait_time)
                    attempt += 1
                    continue # Retry the while loop
                
                # Non-retryable ClientErrors or max_retries exceeded for retryable ones
                logger.error(f"Bedrock API ClientError invoking model {model_id}: {error_code} - {error_message}")
                if error_code == 'AccessDeniedException':
                    raise RuntimeError(f"Access denied for {model_id}. Check IAM permissions.") from e
                elif error_code == 'ResourceNotFoundException':
                    raise RuntimeError(f"Model {model_id} not found. Ensure it is enabled in the region.") from e
                elif error_code == 'ValidationException':
                     logger.error(f"Validation error for model {model_id}. Request body was: {body if 'body' in locals() else 'Body not yet generated'}. Error details: {error_message}")
                     raise ValueError(f"Invalid request parameters for model {model_id}. Details: {error_message}") from e
                else: # Other Boto3 ClientErrors or retryable ones that exhausted retries
                    # Make it clear if it was a retryable error that failed all attempts
                    failure_reason = f"after {max_retries + 1} attempts" if error_code in retryable_errors else "non-retryable error"
                    raise RuntimeError(f"Bedrock API error for {model_id} ({error_code} - {failure_reason}): {error_message}") from e
            
            except Exception as e:
                # Includes ValueErrors from _create_model_body or _extract_output_text, or any other unexpected error
                logger.error(f"An unexpected error occurred invoking model {model_id} (Attempt {attempt + 1}): {e}", exc_info=True) 
                # If it's an unexpected error, we might not want to retry unless we know it's transient.
                # For now, re-raising directly without retry for non-ClientError exceptions.
                raise RuntimeError(f"Unexpected error invoking {model_id}: {e}") from e
        
        # This part should ideally not be reached if max_retries leads to an exception above.
        # However, as a safeguard if the loop exits some other way:
        logger.error(f"Exhausted all {max_retries + 1} retry attempts for model {model_id}. Giving up.")
        raise RuntimeError(f"Failed to invoke model {model_id} after {max_retries + 1} attempts due to persistent retryable errors.")
