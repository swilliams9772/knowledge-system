import pytest
from ksa import KnowledgeSynthesisAgent
from knowledge_graph import KnowledgeGraph

@pytest.fixture
def agent():
    return KnowledgeSynthesisAgent()

@pytest.fixture
def knowledge_graph():
    return KnowledgeGraph()

@pytest.fixture
def sample_data():
    return {
        "query": "Analyze climate change trends",
        "context": {"timeframe": "2000-2024"},
        "expected_tools": ["pandas", "wolfram_alpha"]
    } 