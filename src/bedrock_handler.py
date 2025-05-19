import json
import logging
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from src import config

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

        if "anthropic.claude" in model_id:
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
        # Basic check for deepseek or other models not explicitly handled
        # This assumes a Claude-like structure as a fallback, which might be incorrect.
        # It's better to have explicit handling or clear documentation for new models.
        elif "deepseek" in model_id.lower() or "meta.llama" in model_id.lower(): # Llama2 also Claude-like
            logger.warning(f"Using Anthropic Claude-like structure for model: {model_id}. This may need adjustment.")
            body_params = {
                "prompt": prompt, # This is an assumption
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

        if "anthropic.claude" in model_id or \
           ("deepseek" in model_id.lower() and "completion" in response_body_json) or \
           ("meta.llama" in model_id.lower() and "generation" in response_body_json): # Llama2 specific
            if "completion" in response_body_json: # Claude, some DeepSeek
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

    def invoke_model(self, model_id: str, prompt: str, max_tokens: int = 2048,
                     temperature: float = 0.7, top_p: float = 1.0, top_k: Optional[int] = None) -> str:
        """
        Invokes a Bedrock model with the given prompt and parameters.

        Args:
            model_id: The ID of the Bedrock model to invoke.
            prompt: The prompt to send to the model.
            max_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness. Lower for more factual, higher for creative.
            top_p: Nucleus sampling. Consider adjusting with temperature.
            top_k: Top-k sampling. Supported by some models (e.g. Claude, Cohere).

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

        logger.info(f"Invoking Bedrock model: {model_id} with prompt (first 80 chars): {prompt[:80].replace('\n', ' ')}...")
        logger.debug(f"Full prompt for {model_id}:\n{prompt}")
        logger.debug(f"Invocation params for {model_id}: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")

        try:
            body_params = self._create_model_body(model_id, prompt, max_tokens, temperature, top_p, top_k)
            body = json.dumps(body_params)
            logger.debug(f"Request body for {model_id}: {body}")

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
                raise RuntimeError(f"Failed to decode JSON response from {model_id}. Check logs for details.") from json_e
            
            generated_text = self._extract_output_text(model_id, response_body_json)

            if generated_text is None or not isinstance(generated_text, str):
                logger.error(f"Model {model_id} invocation succeeded but no valid string text found. Parsed as: {generated_text}. Full response JSON: {response_body_json}")
                raise RuntimeError(f"Model {model_id} returned no valid string text. Check logs. Full response: {response_body_json}")

            logger.info(f"Model {model_id} invocation successful. Output (first 80 chars): {generated_text[:80].replace('\n', ' ')}...")
            return generated_text.strip()

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            error_message = e.response.get('Error', {}).get('Message')
            logger.error(f"Bedrock API ClientError invoking model {model_id}: {error_code} - {error_message}")
            if error_code == 'AccessDeniedException':
                raise RuntimeError(f"Access denied for {model_id}. Check IAM permissions.") from e
            elif error_code == 'ResourceNotFoundException':
                raise RuntimeError(f"Model {model_id} not found. Ensure it is enabled in the region.") from e
            elif error_code == 'ThrottlingException':
                logger.warning(f"Throttling error for model {model_id}. Consider retries with backoff.")
                raise RuntimeError(f"Model invocation throttled for {model_id}.") from e # Made message more specific
            elif error_code == 'ModelTimeoutException':
                logger.warning(f"Model {model_id} timed out. Consider increasing timeout or reducing payload.")
                raise RuntimeError(f"Model invocation timed out for {model_id}.") from e # Made message more specific
            elif error_code == 'ValidationException':
                 logger.error(f"Validation error for model {model_id}. Request body was: {body}. Error details: {error_message}")
                 raise ValueError(f"Invalid request parameters for model {model_id}. Details: {error_message}") from e
            else: # Other Boto3 ClientErrors
                raise RuntimeError(f"Bedrock API error for {model_id} ({error_code}): {error_message}") from e
        except Exception as e:
            # Includes ValueErrors from _create_model_body or _extract_output_text, or any other unexpected error
            logger.error(f"An unexpected error occurred invoking model {model_id}: {e}", exc_info=True) # Add exc_info for stack trace
            # It's good to log details that might have caused it, but be careful with prompt size
            # logger.error(f"Prompt was (first 50 chars): {prompt[:50]}... Other params: max_tokens={max_tokens}, temp={temperature}")
            raise RuntimeError(f"Unexpected error invoking {model_id}: {e}") from e
