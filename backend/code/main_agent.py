from functools import partial
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

@tool("query", description="Queries the database and returns the most accurate subgraph to the question")
def call_query_agent(agent, query: str):
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


def main(messages: list[BaseMessage], model_name:str):
    model = ChatOpenAI(
        model=model_name, 
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )
    query_agent = partial(
        call_query_agent,
        agent=create_agent(model=model, tools=[])
    )
    main_agent = create_agent(
        model=model,
        tools=[
            query_agent,
        ]
    )
    
    result = main_agent.invoke({"messages": messages})

    return result["messages"][-1].content