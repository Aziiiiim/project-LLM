from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json, os
from dotenv import load_dotenv

load_dotenv()

with open("schema.json", "r") as f:
    schema = json.load(f)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URL"), 
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
kg_builder = SimpleKGPipeline(
    driver=driver,
    schema={
        "node_types": schema["node_types"],
        "relationship_types": schema["relationship_types"],
        "patterns": schema["patterns"],
        "additional_node_types": False,
    },
    llm=ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    ),
    embedder=OpenAIEmbeddings(
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    ),
    from_pdf=False,
    prompt_template=(
        "Given the following text, "
        "extract entities and relationships "
        "according to the provided schema. ""Return the results in JSON format with 'nodes' and 'relationships' as keys. Text: {text}"),
)

async def ingest(file_path: str):
    with open(file_path, "r") as f:
        text = f.read()
    await kg_builder.run_async(text=text)
