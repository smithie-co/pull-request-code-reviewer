import pytest
from unittest import mock
import json

from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Assuming src.config and src.bedrock_handler are structured as previously discussed
from src.bedrock_handler import BedrockHandler
from src import config

@pytest.fixture
def mock_config_values(monkeypatch):
    """Fixture to mock environment variables loaded by src.config."""
    monkeypatch.setattr(config, 'AWS_ACCESS_KEY_ID', "test_access_key_id")
    monkeypatch.setattr(config, 'AWS_SECRET_ACCESS_KEY', "test_secret_access_key")
    monkeypatch.setattr(config, 'AWS_REGION_NAME', "us-east-1")
    monkeypatch.setattr(config, 'HEAVY_MODEL_ID', "anthropic.claude-v2")
    monkeypatch.setattr(config, 'LIGHT_MODEL_ID', "anthropic.claude-instant-v1")
    monkeypatch.setattr(config, 'DEEPSEEK_MODEL_ID', "meta.llama2-13b-chat-v1") # Placeholder, actual DeepSeek may vary

@pytest.fixture
def mock_boto_client():
    """Fixture to mock boto3.client and yield the mock constructor."""
    with mock.patch('boto3.client') as mock_constructor:
        mock_client_instance = mock.MagicMock() # This is what the constructor will return
        mock_constructor.return_value = mock_client_instance
        yield mock_constructor # Yield the mock of boto3.client itself

# --- Initialization Tests ---

def test_bedrock_handler_initialization_success(mock_config_values, mock_boto_client):
    """Test successful BedrockHandler initialization."""
    # mock_boto_client is now the mock of boto3.client function
    handler = BedrockHandler() 
    # Get the instance that BedrockHandler should have stored
    # BedrockHandler stores the result of boto3.client() in self.client
    # The mock_boto_client fixture ensures boto3.client() returns a MagicMock (mock_client_instance)
    
    mock_boto_client.assert_called_once_with(
        service_name='bedrock-runtime',
        region_name="us-east-1", # From mock_config_values via config
        aws_access_key_id="test_access_key_id", # From mock_config_values via config
        aws_secret_access_key="test_secret_access_key" # From mock_config_values via config
    )
    # Ensure the handler has stored the instance that our mocked constructor returned
    assert isinstance(handler.client, mock.MagicMock)
    # If we need to check that handler.client is specifically the one created inside the fixture:
    # we'd need the fixture to yield both the constructor and the instance, or use a shared object.
    # For this test, checking it was called correctly and an instance is set is sufficient.

def test_bedrock_handler_initialization_missing_credentials(mock_config_values, monkeypatch):
    """Test BedrockHandler initialization fails if AWS credentials are not set."""
    monkeypatch.setattr(config, 'AWS_ACCESS_KEY_ID', None)
    with pytest.raises(ValueError, match="AWS credentials or region not fully configured"):
        BedrockHandler()

    monkeypatch.setattr(config, 'AWS_ACCESS_KEY_ID', "test_access_key_id") # Restore
    monkeypatch.setattr(config, 'AWS_SECRET_ACCESS_KEY', None)
    with pytest.raises(ValueError, match="AWS credentials or region not fully configured"):
        BedrockHandler()

    monkeypatch.setattr(config, 'AWS_SECRET_ACCESS_KEY', "test_secret_key") # Restore
    monkeypatch.setattr(config, 'AWS_REGION_NAME', None)
    with pytest.raises(ValueError, match="AWS credentials or region not fully configured"):
        BedrockHandler()

def test_bedrock_handler_init_boto_no_credentials_error(mock_config_values, monkeypatch):
    """Test BedrockHandler initialization handles NoCredentialsError from boto3.client."""
    with mock.patch('boto3.client', side_effect=NoCredentialsError()) as mock_boto_constructor:
        with pytest.raises(ValueError, match=r"AWS SDK credentials error: Unable to locate credentials"):
            BedrockHandler()
    mock_boto_constructor.assert_called_once()

def test_bedrock_handler_init_boto_partial_credentials_error(mock_config_values, monkeypatch):
    """Test BedrockHandler initialization handles PartialCredentialsError from boto3.client."""
    partial_error = PartialCredentialsError(provider='test', cred_var='test_var')
    with mock.patch('boto3.client', side_effect=partial_error) as mock_boto_constructor:
        with pytest.raises(ValueError, match=r"AWS SDK credentials error: Partial credentials found in test, missing: test_var"):
            BedrockHandler()
    mock_boto_constructor.assert_called_once()

def test_bedrock_handler_init_boto_generic_client_error(mock_config_values, monkeypatch):
    """Test BedrockHandler initialization handles generic ClientError from boto3.client."""
    error_response = {'Error': {'Code': 'SomeOtherException', 'Message': 'Details'}}
    with mock.patch('boto3.client', side_effect=ClientError(error_response, 'operation_name')) as mock_boto_constructor:
        with pytest.raises(RuntimeError, match=r"Failed to initialize Bedrock client due to Boto3 ClientError: An error occurred \(SomeOtherException\) when calling the operation_name operation: Details"):
            BedrockHandler()
    mock_boto_constructor.assert_called_once()

# --- invoke_model Tests ---

@pytest.mark.parametrize("model_id_key, model_id_type, prompt_template, output_payload, expected_output_key", [
    ("LIGHT_MODEL_ID", "anthropic.claude-instant-v1", "Human: {}\n\nAssistant:", {"completion": " Test output"}, "completion"),
    ("HEAVY_MODEL_ID", "anthropic.claude-v2", "Human: {}\n\nAssistant:", {"completion": "Deep analysis output "}, "completion"),
    ("amazon.titan-text-express-v1", "amazon.titan-text-express-v1", "{}", {"results": [{"outputText": "Titan response "}]}, "outputText"),
    # Placeholder for DeepSeek - assuming Claude-like for now based on bedrock_handler logic
    ("DEEPSEEK_MODEL_ID", "meta.llama2-13b-chat-v1", "User: {}\nAssistant:", {"completion": "DeepSeek output "}, "completion"), 
])
def test_invoke_model_success(mock_config_values, mock_boto_client, model_id_key, model_id_type, prompt_template, output_payload, expected_output_key):
    """Test successful model invocation for different model types."""
    handler = BedrockHandler()
    
    actual_model_id = getattr(config, model_id_key) if hasattr(config, model_id_key) else model_id_type

    mock_response_stream = mock.MagicMock()
    mock_response_stream.read.return_value = json.dumps(output_payload).encode('utf-8')
    handler.client.invoke_model.return_value = {
        'body': mock_response_stream,
        'contentType': 'application/json'
    }

    prompt_content = "Test prompt"
    prompt = prompt_template.format(prompt_content)
    response = handler.invoke_model(actual_model_id, prompt)

    handler.client.invoke_model.assert_called_once()
    called_args, called_kwargs = handler.client.invoke_model.call_args
    assert called_kwargs['modelId'] == actual_model_id
    
    body_sent = json.loads(called_kwargs['body'])    
    if "anthropic.claude" in actual_model_id or ("meta.llama2" in actual_model_id and hasattr(config, model_id_key) and model_id_key == "DEEPSEEK_MODEL_ID") :        
        assert body_sent["prompt"] == prompt        # Now uses dynamic token calculation, so check it's a reasonable value        
        assert isinstance(body_sent["max_tokens_to_sample"], int)        
        assert body_sent["max_tokens_to_sample"] > 0    
    elif "amazon.titan" in actual_model_id:        
        assert body_sent["inputText"] == prompt        # Now uses dynamic token calculation, so check it's a reasonable value        
        assert isinstance(body_sent["textGenerationConfig"]["maxTokenCount"], int)        
        assert body_sent["textGenerationConfig"]["maxTokenCount"] > 0

    if "amazon.titan" in actual_model_id:
        assert response == output_payload["results"][0][expected_output_key].strip()
    else:
        assert response == output_payload[expected_output_key].strip()

def test_invoke_model_empty_model_id(mock_config_values, mock_boto_client):
    """Test invoke_model raises ValueError for empty model_id."""
    handler = BedrockHandler()
    with pytest.raises(ValueError, match="Model ID cannot be empty."):
        handler.invoke_model("", "Test prompt")

def test_invoke_model_empty_prompt(mock_config_values, mock_boto_client):
    """Test invoke_model raises ValueError for empty prompt."""
    handler = BedrockHandler()
    with pytest.raises(ValueError, match="Prompt cannot be empty."):
        handler.invoke_model("some-model-id", "")

@pytest.mark.parametrize("error_code, error_message, expected_exception, expected_message_match", [
    ("AccessDeniedException", "User not authorized.", RuntimeError, "Access denied for anthropic.claude-v2. Check IAM permissions."),
    ("ResourceNotFoundException", "Model not found.", RuntimeError, "Model anthropic.claude-v2 not found. Ensure it is enabled in the region."),
    ("ThrottlingException", "Too many requests.", RuntimeError, "Model invocation throttled for anthropic.claude-v2."),
    ("ModelTimeoutException", "Model timed out.", RuntimeError, "Model invocation timed out for anthropic.claude-v2."),
    ("ValidationException", "Invalid request body.", ValueError, "Invalid request parameters for model anthropic.claude-v2. Details: Invalid request body."),
    ("InternalServerException", "Bedrock internal error.", RuntimeError, r"Bedrock API error for anthropic.claude-v2 \(InternalServerException\): Bedrock internal error.")
])
def test_invoke_model_client_errors(mock_config_values, mock_boto_client, error_code, error_message, expected_exception, expected_message_match):
    """Test handling of various Bedrock ClientErrors during model invocation."""
    handler = BedrockHandler()
    model_id = config.HEAVY_MODEL_ID

    error_response = {'Error': {'Code': error_code, 'Message': error_message}}
    handler.client.invoke_model.side_effect = ClientError(error_response, 'invoke_model')

    with pytest.raises(expected_exception, match=expected_message_match):
        handler.invoke_model(model_id, "Test prompt")
    handler.client.invoke_model.assert_called_once()

def test_invoke_model_unexpected_error(mock_config_values, mock_boto_client):
    """Test handling of an unexpected non-ClientError error during model invocation."""
    handler = BedrockHandler()
    model_id = config.LIGHT_MODEL_ID
    handler.client.invoke_model.side_effect = Exception("Some_unexpected_error_details")

    with pytest.raises(RuntimeError, match=r"Unexpected error invoking anthropic.claude-instant-v1: Some_unexpected_error_details"):
        handler.invoke_model(model_id, "Test prompt")

def test_invoke_model_malformed_json_response(mock_config_values, mock_boto_client):
    """Test model invocation when Bedrock returns malformed JSON."""
    handler = BedrockHandler()
    model_id = config.LIGHT_MODEL_ID

    mock_response_stream = mock.MagicMock()
    mock_response_stream.read.return_value = b"this is not json"
    handler.client.invoke_model.return_value = {'body': mock_response_stream}

    with pytest.raises(RuntimeError, match=f"Failed to decode JSON response from {model_id}"):
        handler.invoke_model(model_id, "Test prompt")

def test_invoke_model_missing_output_key(mock_config_values, mock_boto_client):
    """Test model invocation when the expected output key is missing from the response."""
    handler = BedrockHandler()
    model_id = config.LIGHT_MODEL_ID 
    
    mock_response_stream = mock.MagicMock()
    mock_response_stream.read.return_value = json.dumps({"some_other_key": "some_value"}).encode('utf-8')
    handler.client.invoke_model.return_value = {'body': mock_response_stream}

    with pytest.raises(RuntimeError, match=f"Model {model_id} returned no valid string text."):
        handler.invoke_model(model_id, "Human: Test prompt\n\nAssistant:")

# Example test for a model type not explicitly listed in bedrock_handler's _get_body_and_output_key initially,
# to ensure the fallback logic is covered.
def test_invoke_model_unhandled_model_family_fallback(mock_config_values, mock_boto_client):
    """Test invoke_model with a model ID not explicitly handled, using fallback logic."""
    handler = BedrockHandler()
    unhandled_model_id = "some-other-provider.some-model-v1"
    
    mock_response_stream = mock.MagicMock()
    output_payload = {"generated_text": "Fallback output "}
    mock_response_stream.read.return_value = json.dumps(output_payload).encode('utf-8')
    handler.client.invoke_model.return_value = {
        'body': mock_response_stream,
        'contentType': 'application/json'
    }

    prompt = "Human: Test prompt for fallback\n\nAssistant:"
    response = handler.invoke_model(unhandled_model_id, prompt)

    assert response == "Fallback output"
    handler.client.invoke_model.assert_called_once()
    called_args, called_kwargs = handler.client.invoke_model.call_args
    body_sent = json.loads(called_kwargs['body'])
    assert body_sent["prompt"] == prompt
    assert "max_tokens_to_sample" in body_sent # Defaulting to Claude's token parameter name 