from ksa import KnowledgeSynthesisAgent

agent = KnowledgeSynthesisAgent()

# Interactive loop
while True:
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break
        
    result = agent.process_query(query)
    print(f"Agent response: {result}") 