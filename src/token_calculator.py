import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TokenCalculator:
    """
    Utility class for dynamically calculating appropriate max_tokens values
    based on input content size and complexity.
    """
    
    # Rough approximation: 1 token ≈ 4 characters for most models
    CHARS_PER_TOKEN = 4
    
    # Base token allocations for different types of analysis
    BASE_TOKENS = {
        'summary': 300,
        'heavy_analysis': 1500,
        'individual_file': 1000,
        'structured_extraction': 800,
        'release_notes': 200
    }
    
    # Maximum tokens to allow for different analysis types
    MAX_TOKENS = {
        'summary': 800,
        'heavy_analysis': 6000,
        'individual_file': 5000,
        'structured_extraction': 3000,
        'release_notes': 500
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