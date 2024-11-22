from external_tools import ExternalToolRegistry

# Initialize tools
tools = ExternalToolRegistry()

# Example: Complex query combining multiple tools
async def analyze_climate_data():
    # 1. Search for climate data
    search_results = await tools.get_tool("searxng").search(
        "global temperature dataset csv"
    )
    
    # 2. Get dataset URL from search results
    dataset_url = search_results.data[0]["url"]
    
    # 3. Load and analyze data with pandas
    analysis_results = tools.get_tool("pandas").analyze_data(
        dataset_url,
        operations=[
            {"method": "describe"},
            {"method": "rolling", "kwargs": {"window": 12}},
            {"method": "mean"}
        ]
    )
    
    # 4. Get scientific context from Wolfram Alpha
    context = tools.get_tool("wolfram_alpha").query(
        "global temperature trends last 100 years"
    )
    
    return {
        "analysis": analysis_results.data,
        "context": context.data
    } 