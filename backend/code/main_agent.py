import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from .text_to_cypher import AgentQuery
class MultiAgent:
    def __init__(self, model_name:str):
        self.model = ChatOpenAI(
            model=model_name, 
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )
        self.query_agent = AgentQuery()

        #### TOOLS ####
        @tool("query", description="Queries the database and returns the most accurate subgraph to the question")
        def call_query_agent(query: str):
            result = self.query_agent.invoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        ##############

        self.main_agent = create_agent(
            model=self.model,
            tools=[
                call_query_agent,
            ]
        )

    def run(self, messages: list[BaseMessage]):
        result = self.main_agent.invoke({"messages": messages[-1]})
        return result["messages"][-1].content
