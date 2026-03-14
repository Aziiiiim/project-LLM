from typing import Any
import asyncio
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
import json, os
from dotenv import load_dotenv

load_dotenv()

def create_kg_builder() -> SimpleKGPipeline:
    """Creates and configures the knowledge graph builder pipeline.
    This function reads the graph schema from a JSON file called 'graph_schema.json'.
    """
    with open("graph_schema.json", "r") as f:
        schema : dict[str, list[Any]] = json.load(f)
        if not all(key in schema for key in ["node_types", "relationship_types", "patterns"]):
            raise ValueError("Schema must contain 'node_types', 'relationship_types' and 'patterns' keys")

    driver = GraphDatabase.driver(
        "bolt://neo4j:7687", 
        auth=(os.getenv("NEO4J_USERNAME",""), os.getenv("NEO4J_PASSWORD",""))
    )
    return SimpleKGPipeline(
        driver=driver,
        schema={
            "node_types": schema["node_types"],
            "relationship_types": schema["relationship_types"],
            "patterns": schema["patterns"],
            "additional_node_types": False,         # Only allow labels in NODE_TYPES
            "additional_relationship_types": False, # Only allow types in REL_TYPES
            "additional_patterns": False
        },
        llm=OpenAILLM(
            model_name=os.getenv("AI_MODEL",""),
            base_url=os.getenv("AI_ENDPOINT",""),
            api_key=os.getenv("AI_API_KEY",""),
            model_params={
                "response_format": {
                    "type": "json_object"
                }, # Forces adherence
                "temperature": 0 # Keep it deterministic
            }
        ),
        embedder=OpenAIEmbeddings(
            base_url=os.getenv("AI_ENDPOINT",""),
            api_key=os.getenv("AI_API_KEY",""),
            model="text-embedding-3-small"
        ),
        from_pdf=False,
        on_error="RAISE",
        # prompt_template=(
        #     "Given the following text, "
        #     "extract entities and relationships "
        #     "according to the provided schema. ""Return the results in JSON format with 'nodes' and 'relationships' as keys. Text: {text}"),
    )

async def ingest(kg_builder: SimpleKGPipeline, file_path: str):
    with open(file_path, "r") as f:
        text = f.read()
    await kg_builder.run_async(text=text)

if __name__ == "__main__":
    # Tutorial : test with a simple recipe text file. 
    # You can replace this with any text file you'd like to ingest, 
    # just make sure it follows the expected format for your schema.
    kg_builder = create_kg_builder()
    asyncio.run(
        ingest(
            kg_builder, 
            "recipes/cheesecake.txt"
        )
    )