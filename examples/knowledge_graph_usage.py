from knowledge_graph import KnowledgeGraph, KnowledgeTriple

# Initialize graph
kg = KnowledgeGraph()

# Add knowledge
kg.add_triple(KnowledgeTriple(
    subject="neural_network",
    predicate="is_a",
    object="machine_learning_model",
    confidence=1.0,
    metadata={"source": "textbook", "year": 2024}
))

# Semantic query
results = kg.query_graph("""
    SELECT ?model
    WHERE {
        ?model <http://knowledge.base/relation/is_a> <http://knowledge.base/concept/machine_learning_model>
    }
""", method="semantic")

# Similarity query
similar = kg.query_graph(
    "deep learning model",
    method="similarity",
    k=5,
    threshold=0.7
)

# Path query
paths = kg.query_graph(
    start_node="neural_network",
    end_node="artificial_intelligence",
    method="path",
    max_length=3
)

# Get local subgraph
subgraph = kg.get_subgraph("neural_network", depth=2) 