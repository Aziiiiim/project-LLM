import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain.chains import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URL"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
)

print("here")