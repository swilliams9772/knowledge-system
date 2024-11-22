from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import torch
from torch_geometric.data import Data
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
                
        # Update embeddings
        self._update_embeddings(triple)
        
        # Update hierarchies
        self._update_hierarchies(triple)
        
    def query_graph(self, 
                   query: str, 
                   method: str = "semantic",
                   **kwargs) -> List[Dict[str, Any]]:
        """Query the knowledge graph using different methods"""
        if method == "semantic":
            return self._semantic_query(query)
        elif method == "similarity":
            return self._similarity_query(query, **kwargs)
        elif method == "path":
            return self._path_query(query, **kwargs)
        else:
            raise ValueError(f"Unknown query method: {method}")
            
    def _semantic_query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query on RDF graph"""
        results = self.rdf_graph.query(sparql_query)
        return [
            {str(var): str(value) for var, value in result.items()}
            for result in results
        ]
        
    def _similarity_query(self, 
                         query: str,
                         k: int = 5,
                         threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find nodes/edges similar to query using embeddings"""
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Calculate similarities with nodes
        similarities = {}
        for node, emb in self.node_embeddings.items():
            sim = torch.cosine_similarity(query_embedding, emb, dim=0)
            if sim > threshold:
                similarities[node] = sim.item()
                
        # Sort and return top k
        return sorted(
            [{"node": node, "similarity": sim} 
             for node, sim in similarities.items()],
            key=lambda x: x["similarity"],
            reverse=True
        )[:k]
        
    def _path_query(self,
                    start_node: str,
                    end_node: str,
                    max_length: int = 3) -> List[List[str]]:
        """Find paths between nodes"""
        return list(nx.all_simple_paths(
            self.nx_graph,
            start_node,
            end_node,
            cutoff=max_length
        ))
        
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using transformer model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
        
    def _update_embeddings(self, triple: KnowledgeTriple):
        """Update node and edge embeddings"""
        # Update node embeddings
        for node in [triple.subject, triple.object]:
            if node not in self.node_embeddings:
                self.node_embeddings[node] = self._encode_text(node)
                
        # Update edge embedding
        edge_text = f"{triple.subject} {triple.predicate} {triple.object}"
        edge_key = (triple.subject, triple.predicate, triple.object)
        self.edge_embeddings[edge_key] = self._encode_text(edge_text)
        
    def _update_hierarchies(self, triple: KnowledgeTriple):
        """Update concept and relation hierarchies"""
        # Update concept hierarchy
        if triple.predicate == "is_a" or triple.predicate == "subclass_of":
            if triple.object not in self.concept_hierarchy:
                self.concept_hierarchy[triple.object] = set()
            self.concept_hierarchy[triple.object].add(triple.subject)
            
        # Update relation hierarchy
        if triple.predicate == "subproperty_of":
            if triple.object not in self.relation_hierarchy:
                self.relation_hierarchy[triple.object] = set()
            self.relation_hierarchy[triple.object].add(triple.subject)
            
    def get_subgraph(self, 
                     center_node: str,
                     depth: int = 2,
                     max_nodes: int = 50) -> nx.MultiDiGraph:
        """Extract local subgraph around a node"""
        nodes = {center_node}
        current_nodes = {center_node}
        
        for _ in range(depth):
            if len(nodes) >= max_nodes:
                break
                
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.nx_graph.predecessors(node))
                neighbors.update(self.nx_graph.successors(node))
                next_nodes.update(neighbors)
                
            nodes.update(next_nodes)
            current_nodes = next_nodes
            
        return self.nx_graph.subgraph(nodes)
        
    def merge_graphs(self, other_graph: 'KnowledgeGraph'):
        """Merge another knowledge graph into this one"""
        # Merge NetworkX graphs
        self.nx_graph = nx.compose(self.nx_graph, other_graph.nx_graph)
        
        # Merge RDF graphs
        self.rdf_graph += other_graph.rdf_graph
        
        # Merge embeddings
        self.node_embeddings.update(other_graph.node_embeddings)
        self.edge_embeddings.update(other_graph.edge_embeddings)
        
        # Merge hierarchies
        for concept, subconcepts in other_graph.concept_hierarchy.items():
            if concept not in self.concept_hierarchy:
                self.concept_hierarchy[concept] = set()
            self.concept_hierarchy[concept].update(subconcepts)
            
        for relation, subrelations in other_graph.relation_hierarchy.items():
            if relation not in self.relation_hierarchy:
                self.relation_hierarchy[relation] = set()
            self.relation_hierarchy[relation].update(subrelations) 