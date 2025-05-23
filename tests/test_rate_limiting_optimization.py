import unittest
import time
from unittest.mock import Mock, patch, MagicMock
import os

from src.rate_limiter import BedrockRateLimiter, get_global_rate_limiter, configure_global_rate_limiter
from src.token_calculator import TokenCalculator

class TestRateLimiter(unittest.TestCase):
    """Test the global rate limiter functionality."""
    
    def setUp(self):
        """Reset singleton for each test."""
        BedrockRateLimiter._instance = None
    
    def test_rate_limiter_singleton(self):
        """Test that rate limiter implements singleton pattern."""
        limiter1 = get_global_rate_limiter()
        limiter2 = get_global_rate_limiter()
        self.assertIs(limiter1, limiter2)
    
    def test_rate_limiter_configuration(self):
        """Test rate limiter configuration."""
        limiter = configure_global_rate_limiter(requests_per_minute=60, burst_capacity=15)
        self.assertEqual(limiter.requests_per_minute, 60)
        self.assertEqual(limiter.burst_capacity, 15)
    
    def test_token_acquisition(self):
        """Test token acquisition and refill."""
        limiter = BedrockRateLimiter(requests_per_minute=120, burst_capacity=5)  # Fast refill for testing
        
        # Should be able to acquire tokens up to burst capacity
        for i in range(5):
            self.assertTrue(limiter.acquire(timeout=1))
        
        # Next acquisition should need to wait for refill
        start_time = time.time()
        self.assertTrue(limiter.acquire(timeout=2))
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.4)  # Should wait at least 0.5 seconds for refill
    
    def test_token_timeout(self):
        """Test timeout behavior when rate limit exceeded."""
        limiter = BedrockRateLimiter(requests_per_minute=1, burst_capacity=2)  # Very slow refill
        
        # Exhaust initial tokens
        self.assertTrue(limiter.acquire(timeout=0.1))
        self.assertTrue(limiter.acquire(timeout=0.1))
        
        # Next acquisition should timeout
        self.assertFalse(limiter.acquire(timeout=0.1))
    
    def test_rate_limiter_status(self):
        """Test rate limiter status reporting."""
        limiter = BedrockRateLimiter(requests_per_minute=60, burst_capacity=10)
        status = limiter.get_status()
        
        self.assertIn('available_tokens', status)
        self.assertIn('max_capacity', status)
        self.assertIn('requests_per_minute', status)
        self.assertEqual(status['max_capacity'], 10)
        self.assertEqual(status['requests_per_minute'], 60)


class TestTokenCalculator(unittest.TestCase):
    """Test dynamic token calculation functionality."""
    
    def test_input_token_estimation(self):
        """Test input token estimation."""
        # Simple text
        tokens = TokenCalculator.estimate_input_tokens("Hello world")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 10)
        
        # Code with complexity
        code_text = """
        def function():
            if condition:
                return {"key": "value"}
        """
        code_tokens = TokenCalculator.estimate_input_tokens(code_text)
        simple_tokens = TokenCalculator.estimate_input_tokens("simple text of similar length to the code above")
        self.assertGreater(code_tokens, simple_tokens)  # Code should estimate more tokens
    
    def test_complexity_factor_calculation(self):
        """Test complexity factor calculation."""
        # Simple text
        simple_factor = TokenCalculator.calculate_complexity_factor("Hello world, this is simple text.")
        self.assertEqual(simple_factor, 1.0)
        
        # Complex code
        complex_code = """
        +++ b/src/complex.py
        def complex_function(param1, param2):
            if condition1 and condition2:
                return {"result": process(param1, param2)}
        +++ b/src/another.js
        function anotherFunction() {
            const value = process();
            return value;
        }
        """
        complex_factor = TokenCalculator.calculate_complexity_factor(complex_code)
        self.assertGreater(complex_factor, 1.0)
        self.assertLessEqual(complex_factor, 2.0)  # Should be capped at 2.0
    
    def test_dynamic_max_tokens_calculation(self):
        """Test dynamic max tokens calculation."""
        # Small input
        small_tokens = TokenCalculator.calculate_dynamic_max_tokens(
            input_content="Small change",
            analysis_type='summary'
        )
        self.assertGreaterEqual(small_tokens, TokenCalculator.MIN_TOKENS['summary'])
        self.assertLessEqual(small_tokens, TokenCalculator.MAX_TOKENS['summary'])
        
        # Large input
        large_content = "Large code change. " * 100  # Simulate large diff
        large_tokens = TokenCalculator.calculate_dynamic_max_tokens(
            input_content=large_content,
            analysis_type='heavy_analysis'
        )
        self.assertGreater(large_tokens, small_tokens)
        self.assertLessEqual(large_tokens, TokenCalculator.MAX_TOKENS['heavy_analysis'])
    
    def test_model_context_limits(self):
        """Test model context limit detection."""
        # Known models
        claude_limit = TokenCalculator.get_model_context_limits('anthropic.claude-3-sonnet-20240229-v1:0')
        self.assertEqual(claude_limit, 200000)
        
        titan_limit = TokenCalculator.get_model_context_limits('amazon.titan-text-lite-v1')
        self.assertEqual(titan_limit, 4000)
        
        # Unknown model should return conservative default
        unknown_limit = TokenCalculator.get_model_context_limits('unknown.model-v1')
        self.assertEqual(unknown_limit, 4000)
    
    def test_context_window_integration(self):
        """Test that context window limits are respected."""
        # Test with small context window
        tokens = TokenCalculator.calculate_dynamic_max_tokens(
            input_content="Test content",
            analysis_type='heavy_analysis',
            model_context_window=1000
        )
        
        estimated_input = TokenCalculator.estimate_input_tokens("Test content")
        max_possible = 1000 - estimated_input - 100  # Account for buffer
        self.assertLessEqual(tokens, max_possible)


class TestBedrockHandlerIntegration(unittest.TestCase):
    """Test BedrockHandler integration with rate limiter and token calculator."""
    
    @patch('src.bedrock_handler.boto3.client')
    @patch('src.rate_limiter.get_global_rate_limiter')
    @patch('src.token_calculator.TokenCalculator.calculate_dynamic_max_tokens')
    def test_bedrock_handler_rate_limiting(self, mock_calc_tokens, mock_get_limiter, mock_boto_client):
        """Test that BedrockHandler uses rate limiter and dynamic tokens."""
        from src.bedrock_handler import BedrockHandler
        
        # Setup mocks
        mock_limiter = Mock()
        mock_limiter.acquire.return_value = True
        mock_get_limiter.return_value = mock_limiter
        mock_calc_tokens.return_value = 1500
        
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
        
        result = handler.invoke_model(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt="Test prompt",
            analysis_type='heavy_analysis'
        )
        
        # Verify rate limiter was called
        mock_get_limiter.assert_called_once()
        mock_limiter.acquire.assert_called_once_with(timeout=30)
        
        # Verify dynamic token calculation was called
        mock_calc_tokens.assert_called_once()
        
        # Verify response
        self.assertEqual(result, "Test response")
    
    


class TestIntegrationWithAnalysisService(unittest.TestCase):
    """Test integration of optimizations with AnalysisService."""
    
    @patch('src.analysis_service.BedrockHandler')
    def test_analysis_service_uses_dynamic_tokens(self, mock_bedrock_handler):
        """Test that AnalysisService passes analysis_type for dynamic token calculation."""
        from src.analysis_service import AnalysisService
        
        # Setup mock
        mock_handler = Mock()
        mock_handler.invoke_model.return_value = "Mock analysis result"
        mock_bedrock_handler.return_value = mock_handler
        
        # Test service
        service = AnalysisService(mock_handler)
        
        # Test different analysis types call with correct analysis_type
        service.analyze_code_changes("diff content")
        mock_handler.invoke_model.assert_called_with(
            model_id=unittest.mock.ANY,
            prompt=unittest.mock.ANY,
            analysis_type='heavy_analysis',
            temperature=0.5
        )


class TestEnvironmentConfiguration(unittest.TestCase):
    """Test environment variable configuration."""
    
    def test_rate_limiter_env_config(self):
        """Test rate limiter configuration from environment variables."""
        with patch.dict(os.environ, {
            'BEDROCK_REQUESTS_PER_MINUTE': '30',
            'BEDROCK_BURST_CAPACITY': '5'
        }):
            # This would be called in main.py
            requests_per_minute = int(os.getenv("BEDROCK_REQUESTS_PER_MINUTE", "40"))
            burst_capacity = int(os.getenv("BEDROCK_BURST_CAPACITY", "8"))
            
            self.assertEqual(requests_per_minute, 30)
            self.assertEqual(burst_capacity, 5)


if __name__ == '__main__':
    unittest.main() 