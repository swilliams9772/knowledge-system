from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
import numpy as np

class ToolType(str, Enum):
    SEARCH = "search"
    COMPUTE = "compute"
    KNOWLEDGE = "knowledge"
    ANALYSIS = "analysis"

class ConfidenceScore(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    
    @validator('value')
    def validate_confidence(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Confidence must be numeric')
        return float(v)

class KnowledgeTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    confidence: ConfidenceScore
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReasoningStep(BaseModel):
    task: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)
    confidence: ConfidenceScore
    tools: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tools')
    def validate_tools(cls, v):
        if not all(isinstance(tool, str) for tool in v):
            raise ValueError('All tools must be strings')
        return v

class QueryResult(BaseModel):
    query: str = Field(..., min_length=1)
    steps: List[ReasoningStep]
    final_result: Dict[str, Any]
    execution_time: float
    confidence: ConfidenceScore
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryItem(BaseModel):
    content: Union[str, Dict[str, Any]]
    memory_type: str = Field(..., pattern='^(sensory|working|episodic|semantic)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: float = Field(..., ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_content(self) -> 'MemoryItem':
        content = self.content
        if isinstance(content, dict):
            if not all(isinstance(k, str) for k in content.keys()):
                raise ValueError('All dictionary keys must be strings')
        return self

class PlanningStep(BaseModel):
    task: str = Field(..., min_length=1)
    dependencies: List[str] = Field(default_factory=list)
    estimated_time: float = Field(..., gt=0)
    required_tools: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('dependencies')
    def validate_dependencies(cls, v, values):
        if 'task' in values and values['task'] in v:
            raise ValueError('Task cannot depend on itself')
        return v

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, details: Dict[str, Any]):
        self.message = message
        self.details = details
        super().__init__(self.message) 