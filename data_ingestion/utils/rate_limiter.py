"""
Rate limiter utility for managing API request rates.
"""

import asyncio
import time
import logging
from typing import Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for controlling API request rates.
    Supports both per-second and per-minute rate limiting.
    """
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        requests_per_second: int = 5,
        burst_size: int = 10
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size for requests
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        
        # Track request timestamps
        self.request_times = deque()
        self.last_request_time = None
        
        # Rate limiting windows
        self.minute_window = timedelta(minutes=1)
        self.second_window = timedelta(seconds=1)
        
        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0
        self.last_reset_time = datetime.utcnow()
    
    async def wait(self) -> None:
        """
        Wait if necessary to respect rate limits.
        """
        now = datetime.utcnow()
        
        # Clean up old request times
        self._cleanup_old_requests(now)
        
        # Check if we need to wait
        wait_time = self._calculate_wait_time(now)
        
        if wait_time > 0:
            self.throttled_requests += 1
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(now)
        self.last_request_time = now
        self.total_requests += 1
    
    def _cleanup_old_requests(self, now: datetime) -> None:
        """
        Remove old request timestamps that are no longer relevant.
        """
        # Remove requests older than 1 minute
        while self.request_times and (now - self.request_times[0]) > self.minute_window:
            self.request_times.popleft()
    
    def _calculate_wait_time(self, now: datetime) -> float:
        """
        Calculate how long to wait before making the next request.
        
        Returns:
            Wait time in seconds
        """
        wait_time = 0.0
        
        # Check per-minute rate limit
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = self.request_times[0]
            time_since_oldest = (now - oldest_request).total_seconds()
            
            if time_since_oldest < 60:
                wait_time = max(wait_time, 60 - time_since_oldest)
        
        # Check per-second rate limit
        recent_requests = sum(1 for req_time in self.request_times 
                            if (now - req_time).total_seconds() < 1)
        
        if recent_requests >= self.requests_per_second:
            wait_time = max(wait_time, 1.0)
        
        # Check burst limit
        if len(self.request_times) >= self.burst_size:
            wait_time = max(wait_time, 1.0)
        
        return wait_time
    
    def can_make_request(self) -> bool:
        """
        Check if a request can be made immediately without waiting.
        
        Returns:
            True if request can be made immediately
        """
        now = datetime.utcnow()
        self._cleanup_old_requests(now)
        
        # Check all rate limits
        if len(self.request_times) >= self.requests_per_minute:
            return False
        
        recent_requests = sum(1 for req_time in self.request_times 
                            if (now - req_time).total_seconds() < 1)
        
        if recent_requests >= self.requests_per_second:
            return False
        
        if len(self.request_times) >= self.burst_size:
            return False
        
        return True
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with rate limiter statistics
        """
        now = datetime.utcnow()
        self._cleanup_old_requests(now)
        
        recent_requests = sum(1 for req_time in self.request_times 
                            if (now - req_time).total_seconds() < 1)
        
        return {
            'total_requests': self.total_requests,
            'throttled_requests': self.throttled_requests,
            'current_queue_size': len(self.request_times),
            'requests_in_last_minute': len(self.request_times),
            'requests_in_last_second': recent_requests,
            'throttle_rate': (self.throttled_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'last_request_time': self.last_request_time,
            'can_make_request': self.can_make_request()
        }
    
    def reset_stats(self) -> None:
        """Reset rate limiter statistics."""
        self.total_requests = 0
        self.throttled_requests = 0
        self.last_reset_time = datetime.utcnow()
        logger.info("Rate limiter statistics reset")


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on response times and errors.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_second: int = 5,
        burst_size: int = 10,
        min_requests_per_minute: int = 10,
        max_requests_per_minute: int = 120
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            requests_per_minute: Initial requests per minute
            requests_per_second: Initial requests per second
            burst_size: Maximum burst size
            min_requests_per_minute: Minimum requests per minute
            max_requests_per_minute: Maximum requests per minute
        """
        super().__init__(requests_per_minute, requests_per_second, burst_size)
        
        self.min_requests_per_minute = min_requests_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        
        # Response time tracking
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        
        # Adaptive parameters
        self.target_response_time = 1.0  # seconds
        self.error_threshold = 0.1  # 10% error rate
        self.adjustment_factor = 0.1  # 10% adjustment per cycle
    
    def record_response(self, response_time: float, success: bool) -> None:
        """
        Record a response for adaptive rate limiting.
        
        Args:
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Adjust rates periodically
        if len(self.response_times) >= 10:
            self._adjust_rates()
    
    def _adjust_rates(self) -> None:
        """
        Adjust rate limits based on performance metrics.
        """
        if len(self.response_times) < 10:
            return
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        total_requests = self.success_count + self.error_count
        error_rate = self.error_count / total_requests if total_requests > 0 else 0
        
        # Calculate adjustment
        adjustment = 0
        
        # Adjust based on response time
        if avg_response_time > self.target_response_time * 1.5:
            # Response time too high, reduce rate
            adjustment -= self.adjustment_factor
        elif avg_response_time < self.target_response_time * 0.5:
            # Response time good, increase rate
            adjustment += self.adjustment_factor * 0.5
        
        # Adjust based on error rate
        if error_rate > self.error_threshold:
            # Error rate too high, reduce rate
            adjustment -= self.adjustment_factor
        elif error_rate < self.error_threshold * 0.5:
            # Error rate good, increase rate slightly
            adjustment += self.adjustment_factor * 0.25
        
        # Apply adjustment
        if adjustment != 0:
            old_rate = self.requests_per_minute
            self.requests_per_minute = max(
                self.min_requests_per_minute,
                min(self.max_requests_per_minute, int(self.requests_per_minute * (1 + adjustment)))
            )
            
            if old_rate != self.requests_per_minute:
                logger.info(f"Adjusted rate limit from {old_rate} to {self.requests_per_minute} requests/minute")
        
        # Reset counters
        self.response_times.clear()
        self.error_count = 0
        self.success_count = 0
    
    def get_stats(self) -> dict:
        """
        Get adaptive rate limiter statistics.
        
        Returns:
            Dictionary with rate limiter statistics
        """
        base_stats = super().get_stats()
        
        total_requests = self.success_count + self.error_count
        error_rate = self.error_count / total_requests if total_requests > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        adaptive_stats = {
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'min_requests_per_minute': self.min_requests_per_minute,
            'max_requests_per_minute': self.max_requests_per_minute,
            'target_response_time': self.target_response_time,
            'error_threshold': self.error_threshold
        }
        
        base_stats.update(adaptive_stats)
        return base_stats


# Example usage
async def test_rate_limiter():
    """Test the rate limiter functionality."""
    # Create a rate limiter
    rate_limiter = RateLimiter(requests_per_minute=60, requests_per_second=5)
    
    print("Testing rate limiter...")
    
    # Make some requests
    for i in range(10):
        start_time = time.time()
        await rate_limiter.wait()
        end_time = time.time()
        
        print(f"Request {i+1}: waited {end_time - start_time:.2f}s")
    
    # Get stats
    stats = rate_limiter.get_stats()
    print(f"Rate limiter stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_rate_limiter())