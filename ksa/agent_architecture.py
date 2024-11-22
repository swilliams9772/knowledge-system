from ksa.memory_system import HierarchicalMemory
from ksa.planning import ExperienceAugmentedPlanner
from ksa.retrieval import PerplexicaRetrieval
from ksa.reasoning import MultiModalReasoner
from ksa.interface import AgentComputerInterface
from ksa.monitoring import PerformanceMonitor
from ksa.caching import CacheManager, CacheConfig
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class KnowledgeSynthesisAgent:
    def __init__(self):
        self.memory_system = HierarchicalMemory()
        self.planning_system = ExperienceAugmentedPlanner()
        self.retrieval_system = PerplexicaRetrieval()
        self.reasoning_engine = MultiModalReasoner()
        self.action_executor = AgentComputerInterface()
        self.monitor = PerformanceMonitor()
        
        # Initialize cache manager
        self.cache_manager = CacheManager(CacheConfig())

    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process user query with caching and monitoring"""
        try:
            # Check cache first
            cached_result = self.cache_manager.get_from_cache(
                user_input, 
                cache_type="hierarchical"
            )
            if cached_result:
                logger.info("Cache hit for query")
                return cached_result
                
            # Record query start
            self.monitor.record_query("standard")
            
            # Get relevant context through retrieval
            context = self.retrieval_system.search(user_input)
            
            # Generate plan using experience and context
            plan = self.planning_system.create_plan(
                query=user_input,
                context=context,
                past_experience=self.memory_system.get_relevant_experiences()
            )
            
            results = []
            # Execute plan through reasoning and actions
            for step in plan:
                # Check step cache
                step_cache_key = f"{user_input}_{step['task']}"
                cached_step = self.cache_manager.get_from_cache(
                    step_cache_key,
                    cache_type="semantic"
                )
                if cached_step:
                    results.append(cached_step)
                    continue
                    
                reasoning = self.reasoning_engine.analyze(step)
                result = self.action_executor.execute(reasoning)
                
                # Cache step result
                self.cache_manager.store_in_cache(
                    step_cache_key,
                    result,
                    cache_type="semantic"
                )
                
                self.memory_system.store(step, reasoning, result)
                results.append(result)
            
            final_result = {
                "results": results,
                "plan": plan,
                "context": context
            }
            
            # Cache final result
            self.cache_manager.store_in_cache(
                user_input,
                final_result,
                cache_type="hierarchical"
            )
            
            return final_result
            
        except Exception as e:
            logger.exception("Error processing query")
            raise