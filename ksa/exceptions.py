class KSAError(Exception):
    """Base exception for KSA errors."""
    pass

class MemoryError(KSAError):
    """Raised when memory operations fail."""
    pass

class PlanningError(KSAError):
    """Raised when planning operations fail."""
    pass

class ReasoningError(KSAError):
    """Raised when reasoning operations fail."""
    pass

class ToolError(KSAError):
    """Raised when external tool operations fail."""
    pass 