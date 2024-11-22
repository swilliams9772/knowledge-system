from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

class SensoryMemory:
    """Handles immediate sensory inputs with very short retention"""
    def __init__(self, retention_seconds: int = 30):
        self.retention_seconds = retention_seconds
        self.buffer = []
        self.timestamps = []
        
    def add(self, input_data: Dict[str, Any]):
        """Add new sensory input with timestamp"""
        self.buffer.append(input_data)
        self.timestamps.append(datetime.now())
        self._cleanup_old_inputs()
        
    def get_recent(self) -> List[Dict[str, Any]]:
        """Get recent sensory inputs within retention period"""
        self._cleanup_old_inputs()
        return self.buffer
        
    def _cleanup_old_inputs(self):
        """Remove inputs older than retention period"""
        now = datetime.now()
        valid_indices = [
            i for i, ts in enumerate(self.timestamps)
            if (now - ts).total_seconds() <= self.retention_seconds
        ]
        self.buffer = [self.buffer[i] for i in valid_indices]
        self.timestamps = [self.timestamps[i] for i in valid_indices]

class WorkingMemory:
    """Manages active processing and temporary information storage"""
    def __init__(self, max_items: int = 7):
        self.max_items = max_items
        self.items = []
        self.importance_scores = []
        
    def add(self, item: Any, importance: float):
        """Add item with importance score"""
        if len(self.items) >= self.max_items:
            # Remove least important item if full
            min_idx = np.argmin(self.importance_scores)
            self.items.pop(min_idx)
            self.importance_scores.pop(min_idx)
            
        self.items.append(item)
        self.importance_scores.append(importance)
        
    def get_active_items(self) -> List[Any]:
        """Get currently active items"""
        return self.items
        
    def clear(self):
        """Clear working memory"""
        self.items = []
        self.importance_scores = []

class EpisodicMemory:
    """Stores experiences and events with temporal context"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(
            ["Initial memory"], self.embeddings
        )
        self.conversation_memory = ConversationBufferMemory()
        
    def store_episode(self, 
                     episode: Dict[str, Any],
                     metadata: Dict[str, Any] = None):
        """Store an episode with its metadata"""
        # Convert episode to string representation
        episode_text = str(episode)
        
        # Store in vector database with metadata
        self.vector_store.add_texts(
            [episode_text],
            metadatas=[metadata] if metadata else None
        )
        
        # Store in conversation memory if it's a dialogue
        if 'dialogue' in episode:
            self.conversation_memory.save_context(
                {"input": episode['dialogue']['input']},
                {"output": episode['dialogue']['output']}
            )
            
    def retrieve_similar(self, 
                        query: str,
                        k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve k most similar episodes"""
        results = self.vector_store.similarity_search(query, k=k)
        return results
        
    def get_conversation_history(self) -> str:
        """Get recent conversation history"""
        return self.conversation_memory.load_memory_variables({}) 