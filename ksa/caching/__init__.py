from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class CacheConfig:
    def __init__(self, 
                 semantic_threshold: float = 0.7,
                 cache_ttl: int = 3600,  # 1 hour
                 max_cache_size: int = 10000):
        self.semantic_threshold = semantic_threshold
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size

class CacheManager:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}  # Simple in-memory cache
        self.timestamps = {}  # Track when items were added
        
    def get_from_cache(self, key: str, cache_type: str = "keyword") -> Optional[Any]:
        """Get item from cache using specified strategy"""
        try:
            if key not in self.cache:
                return None
                
            # Check if item has expired
            if time.time() - self.timestamps[key] > self.config.cache_ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
                
            return self.cache[key]
                
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
            
    def store_in_cache(self, key: str, value: Any, 
                      cache_type: str = "keyword") -> bool:
        """Store item in cache using specified strategy"""
        try:
            # Enforce cache size limit
            if len(self.cache) >= self.config.max_cache_size:
                # Remove oldest item
                oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            return True
                
        except Exception as e:
            logger.error(f"Cache store error: {str(e)}")
            return False 