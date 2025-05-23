"""
Integration tests to validate the updated API signatures work correctly.
These tests focus on the key changes from our optimization improvements.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

class TestAPISignatureUpdates(unittest.TestCase):
    """Tests to validate that the API signature updates work correctly."""
    
    @patch('src.bedrock_handler.boto3.client')
    @patch('src.rate_limiter.get_global_rate_limiter')
    @patch('src.token_calculator.TokenCalculator.calculate_dynamic_max_tokens')
    def test_bedrock_handler_uses_analysis_type(self, mock_calc_tokens, mock_get_limiter, mock_boto_client):
        """Test that BedrockHandler correctly uses analysis_type parameter for dynamic tokens."""
        from src.bedrock_handler import BedrockHandler
        
        # Setup mocks
        mock_limiter = Mock()
        mock_limiter.acquire.return_value = True
        mock_get_limiter.return_value = mock_limiter
        mock_calc_tokens.return_value = 1500  # Dynamic calculation result
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.get.return_value.read.return_value = '{"content": [{"type": "text", "text": "Test response"}]}'
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        # Test handler
        handler = BedrockHandler(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret", 
            aws_region_name="us-east-1"
        )
        
        # Test the new API signature with analysis_type
        result = handler.invoke_model(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt="Test prompt",
            analysis_type='heavy_analysis'  # This is the new parameter
        )
        
        # Verify rate limiter was used
        mock_get_limiter.assert_called_once()
        mock_limiter.acquire.assert_called_once_with(timeout=30)
        
        # Verify dynamic token calculation was used with correct analysis_type
        mock_calc_tokens.assert_called_once_with(
            input_content="Test prompt",
            analysis_type='heavy_analysis',
            model_context_window=unittest.mock.ANY
        )
        
        # Verify the API call used the calculated tokens
        call_args = mock_client.invoke_model.call_args
        body = eval(call_args[1]['body'])  # Convert JSON string back to dict
        self.assertEqual(body['max_tokens_to_sample'], 1500)
        
        self.assertEqual(result, "Test response")
    
    @patch('src.analysis_service.BedrockHandler')
    def test_analysis_service_passes_analysis_type(self, mock_bedrock_handler_class):
        """Test that AnalysisService methods pass the correct analysis_type."""
        from src.analysis_service import AnalysisService
        
        # Setup mock
        mock_handler = Mock()
        mock_handler.invoke_model.return_value = "Mock analysis result"
        mock_bedrock_handler_class.return_value = mock_handler
        
        # Test service
        service = AnalysisService(mock_handler)
        
        # Test analyze_code_changes passes 'heavy_analysis'
        service.analyze_code_changes("diff content")
        
        # Verify the call includes analysis_type
        call_kwargs = mock_handler.invoke_model.call_args[1]
        self.assertEqual(call_kwargs['analysis_type'], 'heavy_analysis')
        self.assertEqual(call_kwargs['temperature'], 0.5)
        
        # Reset mock for next test
        mock_handler.invoke_model.reset_mock()
        
        # Test summarize_changes passes 'summary'
        service.summarize_changes("diff content")
        
        call_kwargs = mock_handler.invoke_model.call_args[1]
        self.assertEqual(call_kwargs['analysis_type'], 'summary')
        self.assertEqual(call_kwargs['temperature'], 0.7)
    
    @patch('src.analysis_service.BedrockHandler')
    def test_analysis_service_individual_file_analysis(self, mock_bedrock_handler_class):
        """Test that analyze_individual_file_diff passes correct analysis_type."""
        from src.analysis_service import AnalysisService
        
        mock_handler = Mock()
        mock_handler.invoke_model.return_value = "Individual file analysis"
        mock_bedrock_handler_class.return_value = mock_handler
        
        service = AnalysisService(mock_handler)
        
        # Test individual file analysis
        service.analyze_individual_file_diff("file patch", "test.py")
        
        call_kwargs = mock_handler.invoke_model.call_args[1]
        self.assertEqual(call_kwargs['analysis_type'], 'individual_file')
    
    @patch('src.analysis_service.BedrockHandler')
    def test_analysis_service_structured_extraction(self, mock_bedrock_handler_class):
        """Test that analyze_heavy_model_output passes correct analysis_type."""
        from src.analysis_service import AnalysisService
        
        mock_handler = Mock()
        mock_handler.invoke_model.return_value = '[]'  # Empty JSON array
        mock_bedrock_handler_class.return_value = mock_handler
        
        service = AnalysisService(mock_handler)
        
        # Test structured extraction
        service.analyze_heavy_model_output("heavy output", "diff content")
        
        call_kwargs = mock_handler.invoke_model.call_args[1]
        self.assertEqual(call_kwargs['analysis_type'], 'structured_extraction')
    
    @patch('src.analysis_service.BedrockHandler')
    def test_analysis_service_release_notes(self, mock_bedrock_handler_class):
        """Test that generate_release_notes passes correct analysis_type."""
        from src.analysis_service import AnalysisService
        
        mock_handler = Mock()
        mock_handler.invoke_model.return_value = "Release notes content"
        mock_bedrock_handler_class.return_value = mock_handler
        
        service = AnalysisService(mock_handler)
        
        # Test release notes generation
        service.generate_release_notes("diff content")
        
        call_kwargs = mock_handler.invoke_model.call_args[1]
        self.assertEqual(call_kwargs['analysis_type'], 'release_notes')


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the new API maintains backward compatibility where needed."""
    
    @patch('src.bedrock_handler.boto3.client')
    @patch('src.rate_limiter.get_global_rate_limiter')
    @patch('src.token_calculator.TokenCalculator.calculate_dynamic_max_tokens')
    def test_bedrock_handler_max_tokens_override(self, mock_calc_tokens, mock_get_limiter, mock_boto_client):
        """Test that explicit max_tokens parameter still works and overrides dynamic calculation."""
        from src.bedrock_handler import BedrockHandler
        
        # Setup mocks
        mock_limiter = Mock()
        mock_limiter.acquire.return_value = True
        mock_get_limiter.return_value = mock_limiter
        mock_calc_tokens.return_value = 1500  # This should be ignored
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.get.return_value.read.return_value = '{"content": [{"type": "text", "text": "Test response"}]}'
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        handler = BedrockHandler(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret", 
            aws_region_name="us-east-1"
        )
        
        # Test with explicit max_tokens - should NOT use dynamic calculation
        result = handler.invoke_model(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt="Test prompt",
            max_tokens=3000,  # Explicit override
            analysis_type='heavy_analysis'
        )
        
        # Verify dynamic calculation was NOT called when max_tokens is provided
        mock_calc_tokens.assert_not_called()
        
        # Verify the API call used the explicit tokens
        call_args = mock_client.invoke_model.call_args
        body = eval(call_args[1]['body'])
        self.assertEqual(body['max_tokens_to_sample'], 3000)


class TestRateLimitingIntegration(unittest.TestCase):
    """Test rate limiting integration with the updated system."""
    
    @patch('src.bedrock_handler.boto3.client')
    @patch('src.rate_limiter.get_global_rate_limiter')
    def test_rate_limiting_applied_to_all_calls(self, mock_get_limiter, mock_boto_client):
        """Test that rate limiting is applied to all BedrockHandler calls."""
        from src.bedrock_handler import BedrockHandler
        
        # Setup rate limiter mock
        mock_limiter = Mock()
        mock_limiter.acquire.return_value = True
        mock_get_limiter.return_value = mock_limiter
        
        # Setup bedrock client mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.get.return_value.read.return_value = '{"completion": "response"}'
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        handler = BedrockHandler(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret", 
            aws_region_name="us-east-1"
        )
        
        # Make multiple calls
        handler.invoke_model("test-model", "prompt 1", analysis_type='summary')
        handler.invoke_model("test-model", "prompt 2", analysis_type='heavy_analysis')
        
        # Verify rate limiter was called for each request
        self.assertEqual(mock_limiter.acquire.call_count, 2)
        mock_limiter.acquire.assert_called_with(timeout=30)


if __name__ == '__main__':
    unittest.main() 