from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# Initialize tracer
tracer_provider = TracerProvider()
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Initialize metrics
prometheus_reader = PrometheusMetricReader()
meter_provider = MeterProvider(metric_readers=[prometheus_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Define metrics
query_counter = meter.create_counter(
    name="ksa_queries_total",
    description="Total number of queries processed",
    unit="1"
)

query_duration = meter.create_histogram(
    name="ksa_query_duration_seconds",
    description="Duration of query processing",
    unit="s"
)

memory_usage = meter.create_histogram(
    name="ksa_memory_usage_bytes",
    description="Memory usage per operation",
    unit="bytes"
)

tool_calls = meter.create_counter(
    name="ksa_tool_calls_total",
    description="Number of external tool calls",
    unit="1"
)

def trace_method(name: str = None):
    """Decorator to add tracing to methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = name or func.__name__
            with tracer.start_as_current_span(operation_name) as span:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Add metrics
                    query_duration.record(duration)
                    
                    # Add span attributes
                    span.set_attribute("duration_seconds", duration)
                    if hasattr(result, "success"):
                        span.set_attribute("success", result.success)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    logger.exception(f"Error in {operation_name}")
                    raise
                    
        return wrapper
    return decorator

class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def record_memory(self, operation: str):
        """Record memory usage for operation"""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        memory_usage.record(
            mem_info.rss,
            {"operation": operation}
        )
        
    def record_tool_call(self, tool_name: str, success: bool):
        """Record external tool usage"""
        tool_calls.add(
            1,
            {
                "tool": tool_name,
                "success": str(success)
            }
        )
        
    def record_query(self, query_type: str):
        """Record query execution"""
        query_counter.add(
            1,
            {"type": query_type}
        ) 