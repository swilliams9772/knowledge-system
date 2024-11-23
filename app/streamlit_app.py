import streamlit as st
import asyncio
from typing import Dict, Any
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from ksa import KnowledgeSynthesisAgent
from ksa.knowledge_graph import KnowledgeGraph, KnowledgeTriple
from ksa.external_tools import ExternalToolRegistry, ToolType
from ksa.validation.schemas import (
    KnowledgeTriple, QueryResult, ConfidenceScore,
    ToolType, ValidationError
)
from datetime import datetime

# Initialize agent and tools
@st.cache_resource
def init_resources():
    agent = KnowledgeSynthesisAgent()
    tools = ExternalToolRegistry()
    kg = KnowledgeGraph()
    return agent, tools, kg

agent, tools, kg = init_resources()

# Streamlit UI
st.title("Knowledge Synthesis Agent")
st.sidebar.header("Controls")

# Sidebar options
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Query Processing", "Knowledge Graph", "Data Analysis", "Tool Integration"]
)

def validate_query_input(query: str) -> bool:
    """Validate query input"""
    if not query or len(query.strip()) < 3:
        st.error("Query must be at least 3 characters long")
        return False
    return True

def validate_knowledge_triple(subject: str, predicate: str, object_: str) -> bool:
    """Validate knowledge triple input"""
    try:
        KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=ConfidenceScore(
                value=1.0,
                reasoning="User input"
            )
        )
        return True
    except ValidationError as e:
        st.error(f"Invalid knowledge triple: {e.message}")
        return False

def validate_file_upload(file) -> pd.DataFrame:
    """Validate uploaded file"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("File contains no data")
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Main content area
if mode == "Query Processing":
    st.header("Query Processing")
    
    # Query input
    query = st.text_area("Enter your query:", height=100)
    
    if st.button("Process Query"):
        if validate_query_input(query):
            with st.spinner("Processing query..."):
                try:
                    result = agent.process_query(query)
                    
                    # Validate result
                    validated_result = QueryResult(
                        query=query,
                        steps=result["steps"],
                        final_result=result["results"],
                        execution_time=result["execution_time"],
                        confidence=result["confidence"]
                    )
                    
                    # Display results
                    st.subheader("Results")
                    st.json(validated_result.dict())
                    
                except ValidationError as e:
                    st.error(f"Validation error: {e.message}")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

elif mode == "Knowledge Graph":
    st.header("Knowledge Graph Explorer")
    
    # Knowledge input with validation
    st.subheader("Add Knowledge")
    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.text_input("Subject:")
    with col2:
        predicate = st.text_input("Predicate:")
    with col3:
        object_ = st.text_input("Object:")
        
    if st.button("Add Triple"):
        if validate_knowledge_triple(subject, predicate, object_):
            triple = KnowledgeTriple(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=ConfidenceScore(
                    value=1.0,
                    reasoning="User input"
                )
            )
            kg.add_triple(triple)
            st.success("Knowledge added successfully!")
    
    # Graph visualization
    st.subheader("Knowledge Graph Visualization")
    
    # Convert NetworkX graph to Plotly figure
    G = kg.nx_graph
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'),
        hoverinfo='none', mode='lines')
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text',
        hoverinfo='text', textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
        ))
    
    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Add nodes to trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    st.plotly_chart(fig)

elif mode == "Data Analysis":
    st.header("Data Analysis Tools")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        df = validate_file_upload(uploaded_file)
        if df is not None:
            # Analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Statistical Summary", "Time Series Analysis", "Correlation Analysis"]
            )
            
            if analysis_type == "Statistical Summary":
                st.dataframe(df.describe())
                
            elif analysis_type == "Time Series Analysis":
                if st.button("Analyze Time Series"):
                    with st.spinner("Analyzing..."):
                        result = tools.get_tool("pandas").analyze_data(
                            df,
                            operations=[
                                {"method": "rolling", "kwargs": {"window": 12}},
                                {"method": "mean"}
                            ]
                        )
                        st.line_chart(result.data["rolling"])
                        
            elif analysis_type == "Correlation Analysis":
                st.heatmap(df.corr())

elif mode == "Tool Integration":
    st.header("External Tool Integration")
    
    try:
        tool_type = ToolType(st.selectbox(
            "Select Tool",
            [t.value for t in ToolType]
        ))
        
        if tool_type == ToolType.SEARCH:
            query = st.text_input("Search Query:")
            if st.button("Search"):
                with st.spinner("Searching..."):
                    # Create event loop and run search
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(tools.get_tool("searxng").search(query))
                    loop.close()
                    
                    # Display results
                    for result in results.data[:5]:
                        st.write(f"- [{result['title']}]({result['url']})")
                        
        elif tool_type == ToolType.COMPUTE:
            st.subheader("NumPy Operations")
            data = st.text_area("Enter array (comma-separated):")
            operation = st.selectbox(
                "Select Operation",
                ["mean", "std", "max", "min"]
            )
            
            if st.button("Calculate"):
                data_array = [float(x.strip()) for x in data.split(",")]
                result = tools.get_tool("numpy").process_array(
                    data_array,
                    operations=[{"method": operation}]
                )
                st.write(f"Result: {result.data[operation]}")
                
        elif tool_type == ToolType.KNOWLEDGE:
            st.subheader("Wikidata Query")
            query = st.text_area("Enter SPARQL Query:")
            if st.button("Execute Query"):
                with st.spinner("Querying..."):
                    results = tools.get_tool("wikidata").query(query)
                    st.json(results.data)

    except ValidationError as e:
        st.error(f"Tool validation error: {e.message}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io) • "
    "[View Source](https://github.com/yourusername/ksa)"
) 