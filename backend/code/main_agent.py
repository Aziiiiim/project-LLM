import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from .text_to_cypher import AgentQuery
from .scraping_agent import AgentScraping
class MultiAgent:
    def __init__(self, model_name:str):
        self.model = ChatOpenAI(
            model=model_name, 
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )
        self.query_agent = AgentQuery()
        self.scraping_agent = AgentScraping()

        #### TOOLS ####
        @tool("query", description="Queries the database and returns the most accurate subgraph to the question")
        def call_query_agent(query: str):
            result = self.query_agent.invoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        @tool("scrape_recipe",description="Searches and scrapes a recipe from JoCooks based on a cooking request, ingredients, or a recipe URL")
        def call_scraping_agent(query: str) -> str:
            result = self.scraping_agent.invoke(query)
            return result["messages"][-1].content

        ##############

        self.main_agent = create_agent(
            model=self.model,
            tools=[
                call_query_agent,
                call_scraping_agent,
            ],
            system_prompt=(
                "You are a router agent.\n"
                "Use the query tool for graph/database questions.\n"
                "Use the scrape_recipe tool for recipe, cooking, ingredient, or recipe URL requests.\n"
                "If the request is unrelated to available tools, answer normally if possible.\n"
            ),
        )

    def run(self, messages: list[BaseMessage]):
        result = self.main_agent.invoke({"messages": messages[-1]})
        return result["messages"][-1].content
