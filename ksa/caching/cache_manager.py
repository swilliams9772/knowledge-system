from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from gptcache import cache
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from redis import Redis
from azure.cosmos import CosmosClient
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    semantic_threshold: float = 0.7
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 10000
    vector_dimension: int = 768

class CacheManager:
    """Manages different caching strategies for the KSA system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize different cache backends
        self.redis_client = Redis(host='localhost', port=6379, db=0)
        self.cosmos_client = CosmosClient.from_connection_string(
            os.getenv("COSMOS_CONNECTION_STRING")
        )
        
        # Initialize embedding model
        self.onnx = Onnx()
        
        # Setup GPTCache with multiple backends
        self._setup_gptcache()
        
    def _setup_gptcache(self):
        """Initialize GPTCache with multiple storage backends"""
        # SQLite for cache keys
        cache_base = CacheBase('sqlite')
        
        # Milvus for vector storage
        vector_base = VectorBase(
            'milvus',
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
            dimension=self.onnx.dimension,
            collection_name='ksa_cache'
        )
        
        # Create data manager
        data_manager = get_data_manager(cache_base, vector_base)
        
        # Initialize cache
        cache.init(
            pre_embedding_func=self._get_content,
            embedding_func=self.onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )
        
    def _get_content(self, data: Dict[str, Any]) -> str:
        """Extract content for embedding from data"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return str(data.get('content', str(data)))
        return str(data)
        
    def get_from_cache(self, key: str, cache_type: str = "keyword") -> Optional[Any]:
        """Get item from cache using specified strategy"""
        try:
            if cache_type == "keyword":
                return self._keyword_cache_get(key)
            elif cache_type == "semantic":
                return self._semantic_cache_get(key)
            elif cache_type == "hierarchical":
                return self._hierarchical_cache_get(key)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
                
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
            
    def store_in_cache(self, key: str, value: Any, 
                      cache_type: str = "keyword") -> bool:
        """Store item in cache using specified strategy"""
        try:
            if cache_type == "keyword":
                return self._keyword_cache_store(key, value)
            elif cache_type == "semantic":
                return self._semantic_cache_store(key, value)
            elif cache_type == "hierarchical":
                return self._hierarchical_cache_store(key, value)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
                
        except Exception as e:
            logger.error(f"Cache store error: {str(e)}")
            return False
            
    def _keyword_cache_get(self, key: str) -> Optional[Any]:
        """Get from keyword-based cache"""
        # Try Redis first for fast retrieval
        value = self.redis_client.get(key)
        if value:
            return value
            
        # Fallback to Cosmos DB
        container = self.cosmos_client.get_database_client("ksa")\
                       .get_container_client("cache")
        try:
            item = container.read_item(item=key, partition_key=key)
            # Store in Redis for future fast access
            self.redis_client.setex(
                key, 
                self.config.cache_ttl,
                item['value']
            )
            return item['value']
        except:
            return None
            
    def _semantic_cache_get(self, query: str) -> Optional[Any]:
        """Get from semantic cache using similarity search"""
        # Get embedding for query
        query_embedding = self.onnx.to_embeddings(query)
        
        # Search in vector store
        results = cache.data_manager.search_data(
            query_embedding,
            self.config.semantic_threshold
        )
        
        if results:
            return results[0].value
        return None
        
    def _hierarchical_cache_get(self, key: str) -> Optional[Any]:
        """Get from hierarchical cache with fallback strategy"""
        # Try each cache level
        value = self._keyword_cache_get(key)
        if value:
            return value
            
        value = self._semantic_cache_get(key)
        if value:
            return value
            
        return None
        
    def _keyword_cache_store(self, key: str, value: Any) -> bool:
        """Store in keyword-based cache"""
        # Store in Redis with TTL
        self.redis_client.setex(key, self.config.cache_ttl, value)
        
        # Store in Cosmos DB for persistence
        container = self.cosmos_client.get_database_client("ksa")\
                       .get_container_client("cache")
        container.upsert_item({
            'id': key,
            'value': value,
            'timestamp': time.time()
        })
        
        return True
        
    def _semantic_cache_store(self, key: str, value: Any) -> bool:
        """Store in semantic cache"""
        # Get embedding
        embedding = self.onnx.to_embeddings(key)
        
        # Store in vector store
        cache.data_manager.save(key, embedding, value)
        return True
        
    def _hierarchical_cache_store(self, key: str, value: Any) -> bool:
        """Store in all cache levels"""
        keyword_success = self._keyword_cache_store(key, value)
        semantic_success = self._semantic_cache_store(key, value)
        
        return keyword_success and semantic_success 