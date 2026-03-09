import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.tools import tool


### REFERENCES ###
# https://neo4j.com/labs/genai-ecosystem/langchain/
# https://medium.com/data-science/integrating-neo4j-into-the-langchain-ecosystem-df0e988344d2
# https://medium.com/data-science/langchain-has-added-cypher-search-cb9d821120d5


load_dotenv()

graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URL"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    sanitize=True, # Used to remove embedding-like properties (long lists) to not exceed the token limit
)
graph.refresh_schema()


model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0,
)

chain = GraphCypherQAChain.from_llm(
    model, graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    # graph_schema=graph.get_schema,
    return_intermediate_steps=True,
    # cypher_query_corrector=None,
)

@tool
def query_neo4j(query: str):
    "Use this tool to query the Neo4j database with natural language questions."
    try:
        result = chain.invoke(query)
        print("Result", result)
        return result.get("result", "No result found")
    except Exception as e:
        return f"Error: {str(e)}"

system_message = SystemMessage(content=f"""
You are a Neo4j database expert. Use the {query_neo4j.name} tool to answer questions.

Database Schema context: 
{graph.get_schema}

Guidelines:
- Always check the schema carefully before querying
- Ensure queries return non-null values when relevant (and possible)

If a query fails or returns unexpected results:
1. Analyze the error message carefully
2. Verify property names and relationships exist in the schema
3. Try simpler queries first, then build complexity
4. Use modern Cypher syntax (e.g., COUNT() instead of SIZE() for aggregations)
""")
# TODO: Make tests to see if keeping the schema here improves performance
# (Since it is quite large, it slows down the model a lot, and the Chain should already have access to it)

agent = create_agent(model, tools=[query_neo4j], system_prompt=system_message)

# TODO: Add FewShot templates (examples)

def get_tool_calls(model_response):
    tool_calls = []
    for msg in model_response["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    return tool_calls

### EXAMPLES ###
questions = [
    "What are the 5 movies with the highest ratings?",
    "What are the top 10 movies with highest number of connections?", # fails because of outdated syntax -> use COUNT instead of SIZE
    "What are 6 movies with Leonardo DiCaprio? Only return the names",
    "Give me the title and length (not null) of the 5 longest movies in the database."
]

# TODO: make it interactive
for question in questions[:1]:
    print(f"Question: {question}")
    # chain.invoke(question)
    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Response: {response["messages"][-1].content}")
    print(f"Tool calls: {len(get_tool_calls(response))}")
    print("-"*50+"\n")