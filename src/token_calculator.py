import re
import logging
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)

class TokenCalculator:
    """
    Utility class for dynamically calculating appropriate max_tokens values
    based on input content size and complexity.
    """
    
    # Rough approximation: 1 token ≈ 4 characters for most models
    CHARS_PER_TOKEN = 4
    
    # Base token allocations for different types of analysis
    # Significantly reduced based on AWS Bedrock tokens-per-minute quotas
    BASE_TOKENS = {
        'summary': 200,           # Reduced from 300
        'heavy_analysis': 800,    # Reduced from 1200  
        'individual_file': 300,   # Significantly reduced from 600
        'structured_extraction': 400,  # Reduced from 600
        'release_notes': 150      # Reduced from 200
    }
    
    # Maximum tokens to allow for different analysis types
    # Much more conservative to stay within AWS quotas
    MAX_TOKENS = {
        'summary': 400,           # Reduced from 600
        'heavy_analysis': 1500,   # Reduced from 3000
        'individual_file': 800,   # Significantly reduced from 1500  
        'structured_extraction': 1000,  # Reduced from 1500
        'release_notes': 300      # Reduced from 400
    }
    
    # Minimum tokens to ensure meaningful output
    MIN_TOKENS = {
        'summary': 100,
        'heavy_analysis': 500,
        'individual_file': 300,
        'structured_extraction': 200,
        'release_notes': 50
    }
    
    @classmethod
    def estimate_input_tokens(cls, text: str) -> int:
        """
        Estimate the number of tokens in the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        # Basic token estimation
        char_count = len(text)
        estimated_tokens = char_count // cls.CHARS_PER_TOKEN
        
        # Adjust for code complexity (more tokens needed for structured content)
        code_indicators = len(re.findall(r'[{}();]', text))
        if code_indicators > 0:
            # Code tends to be more dense, adjust token estimate
            estimated_tokens = int(estimated_tokens * 1.2)
        
        return estimated_tokens
    
    @classmethod
    def calculate_complexity_factor(cls, content: str) -> float:
        """
        Calculate a complexity factor based on content characteristics.
        
        Args:
            content: Content to analyze
            
        Returns:
            Complexity factor (1.0 = normal, >1.0 = more complex)
        """
        if not content:
            return 1.0
        
        complexity_factor = 1.0
        
        # Check for various indicators of complexity
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return 1.0
        
        # Factor 1: Code density (brackets, keywords, etc.)
        code_chars = len(re.findall(r'[{}();,\[\]+=\-*/<>]', content))
        code_density = code_chars / len(content) if content else 0
        if code_density > 0.1:  # More than 10% special characters
            complexity_factor *= 1.3
        
        # Factor 2: Number of different files in diff
        file_count = len(re.findall(r'\+\+\+ b/', content))
        if file_count > 1:
            complexity_factor *= (1.0 + (file_count - 1) * 0.1)
        
        # Factor 3: Large changes (many additions/deletions)
        added_lines = len(re.findall(r'^\+[^+]', content, re.MULTILINE))
        deleted_lines = len(re.findall(r'^-[^-]', content, re.MULTILINE))
        change_ratio = (added_lines + deleted_lines) / total_lines if total_lines > 0 else 0
        
        if change_ratio > 0.5:  # More than 50% of lines are changes
            complexity_factor *= 1.2
        
        # Factor 4: Multiple programming languages
        language_indicators = {
            'python': len(re.findall(r'\bdef\b|\bclass\b|\bimport\b', content)),
            'javascript': len(re.findall(r'\bfunction\b|\bconst\b|\blet\b|\bvar\b', content)),
            'java': len(re.findall(r'\bpublic\b|\bprivate\b|\bclass\b|\binterface\b', content)),
            'cpp': len(re.findall(r'\b#include\b|\bnamespace\b|\bstruct\b', content)),
        }
        
        languages_detected = sum(1 for count in language_indicators.values() if count > 0)
        if languages_detected > 1:
            complexity_factor *= 1.15
        
        return min(complexity_factor, 2.0)  # Cap at 2x complexity
    
    @classmethod
    def calculate_dynamic_max_tokens(
        cls, 
        input_content: str, 
        analysis_type: str = 'heavy_analysis',
        model_context_window: Optional[int] = None
    ) -> int:
        """
        Calculate optimal max_tokens based on input content and analysis type.
        
        Args:
            input_content: The content to be analyzed
            analysis_type: Type of analysis ('summary', 'heavy_analysis', etc.)
            model_context_window: Maximum context window of the model (optional)
            
        Returns:
            Optimal max_tokens value
        """
        if not input_content:
            return cls.MIN_TOKENS.get(analysis_type, 300)
        
        # Get base configuration for this analysis type
        base_tokens = cls.BASE_TOKENS.get(analysis_type, 1000)
        max_tokens = cls.MAX_TOKENS.get(analysis_type, 4000)
        min_tokens = cls.MIN_TOKENS.get(analysis_type, 300)
        
        # Estimate input tokens
        input_tokens = cls.estimate_input_tokens(input_content)
        
        # Calculate complexity factor
        complexity_factor = cls.calculate_complexity_factor(input_content)
        
        # Dynamic calculation based on input size and complexity
        if input_tokens < 100:
            # Small input - use minimum tokens
            calculated_tokens = min_tokens
        elif input_tokens < 500:
            # Medium input - use base tokens adjusted by complexity
            calculated_tokens = int(base_tokens * complexity_factor)
        else:
            # Large input - scale tokens based on input size and complexity
            # Use a logarithmic scale to avoid excessive token allocation
            import math
            scale_factor = 1 + math.log10(input_tokens / 500) * 0.5
            calculated_tokens = int(base_tokens * scale_factor * complexity_factor)
        
        # Apply bounds
        calculated_tokens = max(min_tokens, min(max_tokens, calculated_tokens))
        
        # If model context window is specified, ensure we don't exceed it
        # Reserve space for input tokens
        if model_context_window:
            max_output_tokens = model_context_window - input_tokens - 100  # 100 token buffer
            calculated_tokens = min(calculated_tokens, max(min_tokens, max_output_tokens))
        
        logger.debug(f"Token calculation for {analysis_type}: "
                    f"input_tokens={input_tokens}, "
                    f"complexity_factor={complexity_factor:.2f}, "
                    f"calculated_tokens={calculated_tokens}")
        
        return calculated_tokens
    
    @classmethod
    def get_model_context_limits(cls, model_id: str) -> Optional[int]:
        """
        Get known context window limits for different models.
        
        Args:
            model_id: The Bedrock model ID
            
        Returns:
            Context window size in tokens, or None if unknown
        """
        # Common Bedrock model context limits
        context_limits = {
            # Claude models
            'anthropic.claude-v2': 100000,
            'anthropic.claude-v2:1': 100000,
            'anthropic.claude-3-sonnet-20240229-v1:0': 200000,
            'anthropic.claude-3-haiku-20240307-v1:0': 200000,
            'anthropic.claude-3-opus-20240229-v1:0': 200000,
            'anthropic.claude-instant-v1': 100000,
            
            # Claude 4 models (adding these)
            'claude-sonnet-4-20250514': 200000,
            'anthropic.claude-4-sonnet': 200000,
            'claude-4-sonnet': 200000,
            'claude-4-haiku': 200000,
            'claude-4-opus': 200000,
            
            # Titan models
            'amazon.titan-text-lite-v1': 4000,
            'amazon.titan-text-express-v1': 8000,
            'amazon.titan-text-large-v1': 8000,
            
            # AI21 models
            'ai21.j2-ultra-v1': 8192,
            'ai21.j2-mid-v1': 8192,
            
            # Cohere models
            'cohere.command-text-v14': 4096,
            'cohere.command-light-text-v14': 4096,
        }
        
        # Try exact match first
        if model_id in context_limits:
            return context_limits[model_id]
        
        # Try partial matches for model families
        for pattern, limit in context_limits.items():
            if pattern.split('-')[0] in model_id:  # Match model family
                return limit
        
        # Default conservative limit if unknown
        logger.warning(f"Unknown context limit for model {model_id}, using conservative default")
        return 4000 

class TokenBudgetManager:
    """
    Manages token budget to stay within AWS Bedrock tokens-per-minute quotas.
    """
    
    def __init__(self):
        self.token_usage_log: Dict[str, list] = {}  # model_id -> [(timestamp, tokens_used)]
        self.quota_cache: Dict[str, int] = {}  # Cached quotas to avoid repeated API calls
        self.quota_cache_time: Optional[float] = None
        self.quota_cache_ttl = 300  # Cache for 5 minutes
        
        # Conservative fallback estimates (used if Service Quotas API fails)
        self.fallback_quotas = {
            'claude-3': 10000,     # Conservative estimate for Claude 3 models
            'claude-4': 8000,      # Even more conservative for Claude 4
            'anthropic.claude': 15000,  # Legacy Claude models may have higher limits
            'amazon.titan': 5000,   # Titan models typically have lower limits
            'default': 5000        # Very conservative default
        }
        
        # Try to discover actual quotas from AWS
        self._discover_quotas()
    
    def _discover_quotas(self):
        """Discover actual quotas from AWS Service Quotas API."""
        # Check if quota discovery is disabled
        if os.getenv("DISABLE_QUOTA_DISCOVERY", "false").lower() == "true":
            logger.info("Quota discovery disabled via DISABLE_QUOTA_DISCOVERY=true. Using fallback estimates.")
            return
            
        try:
            quota_client = ServiceQuotaClient()
            discovered_quotas = quota_client.get_all_bedrock_quotas()
            
            if discovered_quotas:
                logger.info(f"Discovered {len(discovered_quotas)} Bedrock token quotas from AWS Service Quotas")
                self.quota_cache = discovered_quotas
                self.quota_cache_time = time.time()
                
                # Update fallback quotas with discovered values
                for pattern, quota in discovered_quotas.items():
                    self.fallback_quotas[pattern] = quota
                    
                logger.info(f"Updated quotas: {self.fallback_quotas}")
            else:
                logger.info("No quotas discovered from Service Quotas API, using fallback estimates")
                
        except Exception as e:
            logger.warning(f"Failed to discover quotas from Service Quotas API: {e}. Using fallback estimates.")
    
    def _is_quota_cache_valid(self) -> bool:
        """Check if the quota cache is still valid."""
        if not self.quota_cache_time:
            return False
        return (time.time() - self.quota_cache_time) < self.quota_cache_ttl
    
    def get_quota_for_model(self, model_id: str) -> int:
        """Get the tokens-per-minute quota for a specific model."""
        # Refresh quota cache if needed
        if not self._is_quota_cache_valid():
            logger.debug("Quota cache expired, refreshing...")
            self._discover_quotas()
        
        # First try cached discovered quotas
        if self.quota_cache:
            for pattern, quota in self.quota_cache.items():
                if pattern in model_id.lower():
                    return quota
        
        # Fall back to hardcoded estimates
        for pattern, quota in self.fallback_quotas.items():
            if pattern in model_id.lower():
                return quota
        
        return self.fallback_quotas['default']
    
    def clean_old_usage(self, model_id: str, current_time: float):
        """Remove usage records older than 1 minute."""
        if model_id not in self.token_usage_log:
            self.token_usage_log[model_id] = []
        
        # Keep only records from the last 60 seconds
        cutoff_time = current_time - 60
        self.token_usage_log[model_id] = [
            (timestamp, tokens) for timestamp, tokens in self.token_usage_log[model_id]
            if timestamp > cutoff_time
        ]
    
    def get_current_usage(self, model_id: str) -> int:
        """Get current token usage in the last minute for a model."""
        current_time = time.time()
        self.clean_old_usage(model_id, current_time)
        
        return sum(tokens for _, tokens in self.token_usage_log[model_id])
    
    def can_use_tokens(self, model_id: str, requested_tokens: int) -> tuple[bool, int]:
        """
        Check if we can use the requested tokens without exceeding quota.
        
        Returns:
            (can_proceed, available_tokens)
        """
        current_usage = self.get_current_usage(model_id)
        quota = self.get_quota_for_model(model_id)
        available = quota - current_usage
        
        if requested_tokens <= available:
            return True, available
        else:
            return False, available
    
    def record_usage(self, model_id: str, tokens_used: int):
        """Record token usage for a model."""
        current_time = time.time()
        if model_id not in self.token_usage_log:
            self.token_usage_log[model_id] = []
        
        self.token_usage_log[model_id].append((current_time, tokens_used))
        logger.debug(f"Recorded {tokens_used} tokens for {model_id}. Current usage: {self.get_current_usage(model_id)}")

# Global token budget manager instance
_global_token_budget_manager = None

def get_global_token_budget_manager() -> TokenBudgetManager:
    """Get the global token budget manager instance."""
    global _global_token_budget_manager
    if _global_token_budget_manager is None:
        _global_token_budget_manager = TokenBudgetManager()
    return _global_token_budget_manager 

class ServiceQuotaClient:
    """
    Client to query AWS Service Quotas for Bedrock token limits.
    """
    
    def __init__(self, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None,
                 aws_region_name: Optional[str] = None):
        """Initialize the Service Quotas client."""
        try:
            # Import config here to avoid circular imports
            from src import config
            
            # Use provided credentials or fall back to config
            _aws_access_key_id = aws_access_key_id or config.AWS_ACCESS_KEY_ID
            _aws_secret_access_key = aws_secret_access_key or config.AWS_SECRET_ACCESS_KEY  
            _aws_region_name = aws_region_name or config.AWS_REGION_NAME
            
            self.client = boto3.client(
                service_name="service-quotas",
                region_name=_aws_region_name,
                aws_access_key_id=_aws_access_key_id,
                aws_secret_access_key=_aws_secret_access_key
            )
            self.region = _aws_region_name
            logger.info(f"Service Quotas client initialized for region: {_aws_region_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Service Quotas client: {e}. Will use fallback quotas.")
            self.client = None
            self.region = None
    
    def get_bedrock_token_quota(self, model_family: str) -> Optional[int]:
        """
        Query AWS Service Quotas for Bedrock token limits.
        
        Args:
            model_family: Model family like 'claude-4', 'claude-3', etc.
            
        Returns:
            Tokens per minute quota, or None if not found/failed
        """
        if not self.client:
            return None
            
        try:
            # Get all Bedrock quotas
            service_code = "bedrock"
            response = self.client.list_service_quotas(ServiceCode=service_code)
            
            # Look for token-related quotas
            for quota in response.get('Quotas', []):
                quota_name = quota.get('QuotaName', '').lower()
                quota_code = quota.get('QuotaCode')
                quota_value = quota.get('Value')
                
                # Check if this quota matches our model family and is for tokens
                if ('tokens per minute' in quota_name and 
                    'on-demand' in quota_name and
                    model_family.lower() in quota_name):
                    
                    logger.info(f"Found token quota for {model_family}: {quota_value} tokens/min (Code: {quota_code})")
                    return int(quota_value)
            
            # If we didn't find a specific quota, try to get a general one
            logger.debug(f"No specific token quota found for {model_family}, checking for general quotas")
            return None
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'AccessDeniedException':
                logger.warning("No permission to query Service Quotas. Add 'servicequotas:ListServiceQuotas' permission for dynamic quota discovery.")
            else:
                logger.warning(f"Error querying Service Quotas: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error querying Service Quotas: {e}")
            return None
    
    def get_all_bedrock_quotas(self) -> Dict[str, int]:
        """
        Get all available Bedrock token quotas.
        
        Returns:
            Dictionary mapping model patterns to token quotas
        """
        quotas = {}
        if not self.client:
            return quotas
            
        try:
            service_code = "bedrock"
            response = self.client.list_service_quotas(ServiceCode=service_code)
            
            for quota in response.get('Quotas', []):
                quota_name = quota.get('QuotaName', '').lower()
                quota_value = quota.get('Value')
                
                if 'tokens per minute' in quota_name and 'on-demand' in quota_name:
                    # Extract model name from quota name
                    # Example: "On-demand InvokeModel tokens per minute for Claude 3 Sonnet"
                    if 'claude 3' in quota_name or 'claude-3' in quota_name:
                        quotas['claude-3'] = int(quota_value)
                    elif 'claude 4' in quota_name or 'claude-4' in quota_name:
                        quotas['claude-4'] = int(quota_value)
                    elif 'claude' in quota_name:
                        quotas['anthropic.claude'] = int(quota_value)
                    elif 'titan' in quota_name:
                        quotas['amazon.titan'] = int(quota_value)
                    
                    logger.info(f"Found quota: {quota_name} = {quota_value} tokens/min")
            
            return quotas
            
        except Exception as e:
            logger.warning(f"Error getting all Bedrock quotas: {e}")
            return quotas 