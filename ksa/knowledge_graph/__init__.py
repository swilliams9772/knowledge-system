from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import torch
from transformers import AutoTokenizer, AutoModel

@dataclass
class KnowledgeTriple:
    """Represents a knowledge triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeGraph:
    """Enhanced knowledge graph with multiple representation capabilities"""
    
    def __init__(self):
        # Graph backends
        self.nx_graph = nx.MultiDiGraph()  # NetworkX for algorithms
        self.rdf_graph = Graph()  # RDF for semantic queries
        
        # Node and edge embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.node_embeddings = {}
        self.edge_embeddings = {}
        
        # Namespaces for RDF
        self.ns = {
            'base': Namespace("http://knowledge.base/"),
            'concept': Namespace("http://knowledge.base/concept/"),
            'relation': Namespace("http://knowledge.base/relation/"),
            'property': Namespace("http://knowledge.base/property/")
        }
        
        # Hierarchical structure
        self.concept_hierarchy = {}
        self.relation_hierarchy = {}
        
    def add_triple(self, triple: KnowledgeTriple):
        """Add a knowledge triple to the graph"""
        # Add to NetworkX graph
        self.nx_graph.add_edge(
            triple.subject,
            triple.object,
            predicate=triple.predicate,
            confidence=triple.confidence,
            metadata=triple.metadata
        )
        
        # Add to RDF graph
        subj = self.ns['concept'][triple.subject]
        pred = self.ns['relation'][triple.predicate]
        obj = self.ns['concept'][triple.object]
        
        self.rdf_graph.add((subj, pred, obj))
        if triple.metadata:
            for key, value in triple.metadata.items():
                self.rdf_graph.add((
                    subj, 
                    self.ns['property'][key],
                    Literal(value)
                )) 