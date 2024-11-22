from typing import Dict, Any, List, Optional
import requests
from dataclasses import dataclass
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
import wolframalpha
import json
import os

@dataclass 
class ToolResponse:
    """Standardized response format for external tools"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ExternalToolRegistry:
    """Registry and interface for external API tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._setup_default_tools()
        
    def _setup_default_tools(self):
        """Initialize default set of external tools"""
        # Search Tools
        if os.getenv("SEARXNG_URL"):
            self.register_tool(
                "searxng",
                SearxNGSearch(base_url=os.getenv("SEARXNG_URL"))
            )
            
        if os.getenv("WOLFRAM_APP_ID"):
            self.register_tool(
                "wolfram_alpha",
                WolframAlphaAPI(app_id=os.getenv("WOLFRAM_APP_ID"))
            )
            
        # Knowledge Base Tools
        self.register_tool("wikipedia", WikipediaAPI())
        self.register_tool("wikidata", WikidataAPI())
        
        # Data Analysis Tools
        self.register_tool("pandas", PandasAnalyzer())
        self.register_tool("numpy", NumPyProcessor())
        
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new external tool"""
        self.tools[name] = tool
        
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())

class SearxNGSearch:
    """Integration with SearxNG self-hosted search engine"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def search(self, query: str, **kwargs) -> ToolResponse:
        """Perform search query"""
        try:
            params = {
                "q": query,
                "format": "json",
                **kwargs
            }
            response = requests.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            
            return ToolResponse(
                success=True,
                data=response.json()["results"],
                metadata={"source": "searxng"}
            )
            
        except Exception as e:
            return ToolResponse(
                success=False,
                data=None,
                error=str(e),
                metadata={"source": "searxng"}
            )

class WolframAlphaAPI:
    """Integration with Wolfram Alpha API for computational knowledge"""
    
    def __init__(self, app_id: str):
        self.client = wolframalpha.Client(app_id)
        
    def query(self, query: str) -> ToolResponse:
        """Send query to Wolfram Alpha"""
        try:
            result = self.client.query(query)
            
            # Extract the plaintext results
            answers = []
            for pod in result.pods:
                for sub in pod.subpods:
                    if sub.plaintext:
                        answers.append({
                            "title": pod.title,
                            "text": sub.plaintext
                        })
            
            return ToolResponse(
                success=True,
                data=answers,
                metadata={
                    "source": "wolfram_alpha",
                    "query_url": f"https://www.wolframalpha.com/input?i={query}"
                }
            )
            
        except Exception as e:
            return ToolResponse(
                success=False,
                data=None, 
                error=str(e),
                metadata={"source": "wolfram_alpha"}
            )

class WikidataAPI:
    """Integration with Wikidata for structured knowledge"""
    
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        
    def query(self, sparql_query: str) -> ToolResponse:
        """Execute SPARQL query against Wikidata"""
        try:
            response = requests.get(
                self.endpoint,
                params={
                    "query": sparql_query,
                    "format": "json"
                },
                headers={
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            return ToolResponse(
                success=True,
                data=response.json()["results"]["bindings"],
                metadata={"source": "wikidata"}
            )
            
        except Exception as e:
            return ToolResponse(
                success=False,
                data=None,
                error=str(e),
                metadata={"source": "wikidata"}
            )

class PandasAnalyzer:
    """Data analysis capabilities using pandas"""
    
    def analyze_data(self, data: Any, operations: List[Dict[str, Any]]) -> ToolResponse:
        """Perform pandas operations on data"""
        try:
            import pandas as pd
            
            # Convert input data to DataFrame if needed
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            results = {}
            for op in operations:
                method = op["method"]
                args = op.get("args", [])
                kwargs = op.get("kwargs", {})
                
                if hasattr(df, method):
                    results[method] = getattr(df, method)(*args, **kwargs)
                    
            return ToolResponse(
                success=True,
                data=results,
                metadata={"source": "pandas"}
            )
            
        except Exception as e:
            return ToolResponse(
                success=False,
                data=None,
                error=str(e),
                metadata={"source": "pandas"}
            )

class NumPyProcessor:
    """Scientific computing capabilities using NumPy"""
    
    def process_array(self, data: Any, operations: List[Dict[str, Any]]) -> ToolResponse:
        """Perform NumPy operations on array data"""
        try:
            import numpy as np
            
            # Convert input to NumPy array
            arr = np.array(data)
            
            results = {}
            for op in operations:
                method = op["method"]
                args = op.get("args", [])
                kwargs = op.get("kwargs", {})
                
                if hasattr(np, method):
                    results[method] = getattr(np, method)(arr, *args, **kwargs)
                    
            return ToolResponse(
                success=True,
                data=results,
                metadata={"source": "numpy"}
            )
            
        except Exception as e:
            return ToolResponse(
                success=False,
                data=None,
                error=str(e),
                metadata={"source": "numpy"}
            ) 