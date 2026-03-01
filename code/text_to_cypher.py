import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

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

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0,
)

chain = GraphCypherQAChain.from_llm(
    model, graph=graph, verbose=True, allow_dangerous_requests=True
)

### EXAMPLES ###
# chain.invoke("What are the 5 movies with the highest ratings?")
# chain.invoke("What are the top 10 movies with highest number of connections?") # fails because of outdated syntax -> use COUNT instead of SIZE
# chain.invoke("What are 6 movies with Leonardo DiCaprio? Only return the names")
# chain.invoke("Give me the title and length (not null) of the 5 longest movies in the database.")