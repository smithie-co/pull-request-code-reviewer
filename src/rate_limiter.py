import time
import threading
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BedrockRateLimiter:
    """
    A singleton-style global rate limiter for AWS Bedrock API calls.
    
    This implements a token bucket algorithm to control the rate of API calls
    across the entire application.
    """
    
    _instance: Optional['BedrockRateLimiter'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BedrockRateLimiter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, requests_per_minute: int = 50, burst_capacity: int = 10):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            burst_capacity: Number of tokens that can be used in a burst
        """
        # Only initialize once (singleton pattern)
        if hasattr(self, '_initialized'):
            return
            
        self.requests_per_minute = requests_per_minute
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity  # Start with full capacity
        self.last_refill_time = time.time()
        self._token_lock = threading.Lock()
        self._initialized = True
        
        logger.info(f"BedrockRateLimiter initialized: {requests_per_minute} req/min, burst capacity: {burst_capacity}")
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed since last refill."""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time
        
        # Calculate tokens to add based on rate (requests per minute)
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60.0)
        
        if tokens_to_add >= 1:  # Only add if at least 1 token worth of time has passed
            self.tokens = min(self.burst_capacity, self.tokens + int(tokens_to_add))
            self.last_refill_time = current_time
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token for making a request.
        
        Args:
            timeout: Maximum time to wait for a token (None for no timeout)
            
        Returns:
            True if token acquired, False if timeout exceeded
        """
        start_time = time.time()
        
        while True:
            with self._token_lock:
                self._refill_tokens()
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    logger.debug(f"Rate limiter: Token acquired. Remaining tokens: {self.tokens}")
                    return True
                
                # Calculate wait time until next token
                time_until_next_token = (1.0 / (self.requests_per_minute / 60.0))
                
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning(f"Rate limiter: Timeout exceeded waiting for token")
                        return False
                    
                    # Don't wait longer than the remaining timeout
                    time_until_next_token = min(time_until_next_token, timeout - elapsed)
            
            # Wait before checking again
            wait_time = min(time_until_next_token, 1.0)  # Cap at 1 second intervals
            logger.debug(f"Rate limiter: Waiting {wait_time:.2f}s for next token")
            time.sleep(wait_time)
    
    def get_status(self) -> dict:
        """Get current rate limiter status for debugging."""
        with self._token_lock:
            self._refill_tokens()
            return {
                'available_tokens': self.tokens,
                'max_capacity': self.burst_capacity,
                'requests_per_minute': self.requests_per_minute,
                'last_refill_time': self.last_refill_time
            }

# Global instance
_global_rate_limiter: Optional[BedrockRateLimiter] = None

def get_global_rate_limiter() -> BedrockRateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = BedrockRateLimiter()
    return _global_rate_limiter

def configure_global_rate_limiter(requests_per_minute: int = 50, burst_capacity: int = 10) -> BedrockRateLimiter:
    """Configure the global rate limiter with custom settings."""
    global _global_rate_limiter
    _global_rate_limiter = BedrockRateLimiter(requests_per_minute, burst_capacity)
    return _global_rate_limiter 