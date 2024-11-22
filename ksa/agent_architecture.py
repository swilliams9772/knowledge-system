from ksa.memory_system import HierarchicalMemory
from ksa.planning import ExperienceAugmentedPlanner
from ksa.retrieval import PerplexicaRetrieval
from ksa.reasoning import MultiModalReasoner
from ksa.interface import AgentComputerInterface
from ksa.monitoring import PerformanceMonitor
from ksa.caching import CacheManager, CacheConfig

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