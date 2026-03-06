from typing import Any
import asyncio
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
import json, os
from dotenv import load_dotenv

import logging

from sqlalchemy import true
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

with open("graph_schema.json", "r") as f:
    schema : dict[str, list[Any]] = json.load(f)
    if not all(key in schema for key in ["node_types", "relationship_types", "patterns"]):
        raise ValueError("Schema must contain 'node_types', 'relationship_types' and 'patterns' keys")

print(schema)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URL",""), 
    auth=(os.getenv("NEO4J_USERNAME",""), os.getenv("NEO4J_PASSWORD",""))
)
kg_builder = SimpleKGPipeline(
    driver=driver,
    schema={
        "node_types": schema["node_types"],
        "relationship_types": schema["relationship_types"],
        "patterns": [
            ("Ingredient", "INGREDIENT_OF", "Recipe"),
            ("Author", "AUTHOR_OF", "Recipe")
        ],
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

async def ingest(file_path: str):
    with open(file_path, "r") as f:
        text = f.read()
    await kg_builder.run_async(text=text)

asyncio.run(ingest("recipes/cheesecake.txt"))