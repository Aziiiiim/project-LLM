# 1. Import required modules
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# 2. Define your search tool using @tool decorator:

class WebSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(description="The search query to find factual information")

@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Performs a web search for factual information."""
    search_data = {
        "population of tokyo": "Tokyo has a population of approximately 14 million people.",
        "capital of france": "The capital of France is Paris.",
        "capital of japan": "The capital of Japan is Tokyo.",
        "population of new york": "New York City has a population of approximately 8.3 million people.",
        "population of london": "London has a population of approximately 9.1 million people.",
        "capital of germany": "The capital of Germany is Berlin.",
        # Add more entries as needed
    }

    query_lower = query.lower()
    for key, result in search_data.items():
        if key in query_lower or query_lower in key:
            return result
    return "No results found for the query."

# 3. Define your calculator tool using @tool decorator:

class CalculatorInput(BaseModel):
    """Input for calculator."""
    expression: str = Field(description="The mathematical expression to evaluate")

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Use this for arithmetic operations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"

# 4. Create the ChatOpenAI model with your environment variables

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
)

# 5. Create agent using create_agent():
agent = create_agent(model, tools=[web_search, calculator])

# 6. Test with multi-step queries in a loop:
queries = \
    ["What is the population of Tokyo multiplied by 2?",
     "Search for the capital of France and tell me how many letters are in its name",
     "What is the population of New York divided by 2?",
     ]
for query in queries:
    print("\nUser query: ", query)
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    last_message = response["messages"][-1]

    # 7. Optional: Display which tools were used:
    tool_calls = []
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend([tc["name"] for tc in msg.tool_calls])

    print("Model response: ", last_message.content)
    print(f"Tools used: {', '.join(set(tool_calls))}")
