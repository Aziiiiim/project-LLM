import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.tools import tool
from langchain_community.callbacks import get_openai_callback


### REFERENCES ###
# https://neo4j.com/labs/genai-ecosystem/langchain/
# https://medium.com/data-science/integrating-neo4j-into-the-langchain-ecosystem-df0e988344d2
# https://medium.com/data-science/langchain-has-added-cypher-search-cb9d821120d5

### INITIALIZATIONS ###

load_dotenv()

def _build_graph():
    graph = Neo4jGraph(
        url="bolt://neo4j:7687",
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        sanitize=True, # Used to remove embedding-like properties (long lists) to not exceed the token limit
    )
    graph.refresh_schema()
    return graph

def _build_model():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        temperature=0,
    )
    return model

def _build_chain(model, graph):
    chain = GraphCypherQAChain.from_llm(
        model,
        graph=graph,
        verbose=False,
        allow_dangerous_requests=True,
        return_intermediate_steps=False,
        # cypher_query_corrector=None,
    )
    return chain


### PROMPT ###

# After some testing, it turned out including the database schema in this context was not efficient, as
# - the schema is already included in the chain
# - it requires a lot more tokens for each query (about 600+), plus slows down the request
# - the results were pretty much the same

def _get_system_message(tool_name: str) -> SystemMessage:
    return SystemMessage(
        content=f"""
    You are a Neo4j database expert. Use the {tool_name} tool to answer questions.
    
    Guidelines:
    - Always check the schema carefully before querying
    - Ensure queries return non-null values when relevant (and possible)
    
    If a query fails or returns unexpected results:
    1. Analyze the error message carefully
    2. Verify property names and relationships exist in the schema
    3. Try simpler queries first, then build complexity
    4. Use modern Cypher syntax (e.g., COUNT() instead of SIZE() for aggregations)
    """
    )
# I case we need few shot examples
examples = [
    {
        "question": "What are the 5 movies with the highest ratings?",
        "cypher": "MATCH (m:Movie) WHERE m.imdbRating IS NOT NULL RETURN m.title AS title, m.imdbRating AS rating ORDER BY m.imdbRating DESC LIMIT 5",
    },
]

def get_tool_calls(model_response):
    tool_calls = []
    for msg in model_response["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    return tool_calls

### TESTS ###
questions = [
    "What are the 5 movies with the highest ratings?",
    "What are the top 10 movies with highest number of connections?",
    "What are 6 movies with Leonardo DiCaprio? Only return the names",
    "Give me the title and length (not null) of the 5 longest movies in the database.",
]

### TRACK CONVERSATION HISTORY ###
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )


# def interactive_mode():
#     print("Neo4j Chat Agent (type 'exit' to quit)")
#     agent = create_agent(model, tools=[query_neo4j], system_prompt=system_message)
#
#     while True:
#         question = input("\nYou: ").strip()
#         if question.lower() in ["exit", "quit"]:
#             break
#
#         invoke_agent(agent, question)
#
#         # # Optional: Show generated Cypher
#         # if input("Show Cypher? (y/n): ").lower() == "y":
#         #     pass

# interactive_mode()

class AgentQuery:
    def __init__(self):
        load_dotenv()
        graph = _build_graph()
        model = _build_model()
        self.query_tool = self._create_tool()
        system_message = _get_system_message(self.query_tool.name)
        self.chain = _build_chain(model, graph)
        self.agent = create_agent(model, tools=[self.query_tool], system_prompt=system_message)

    ### TOOLS ###

    def _create_tool(self):
        """Creates the tool with access to the chain"""

        @tool
        def query_neo4j(query: str):
            """
            Use this tool to query the Neo4j database with natural language questions.
            """
            try:
                result = self.chain.invoke(query)
                # query = result.get("query", "No query found")
                # print("Result: ", result)
                return result.get("result", "No result found")
            except Exception as e:
                return f"Error: {str(e)}"

        return query_neo4j

    def invoke(self, question: str):
        print(f"Question: {question}")
        with get_openai_callback() as cb:
            response = self.agent.invoke({"messages": [HumanMessage(content=question)]})
            print(f"Total tokens used: {cb.total_tokens}")
        print(f"Tool calls: {len(get_tool_calls(response))}")
        print(f"Response: {response['messages'][-1].content}")
        print("-" * 50 + "\n")

