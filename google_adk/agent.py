import asyncio
import os
import json
from google import genai
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from tavily import TavilyClient

# Clients
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Tool function
def search_company_news(company: str) -> str:
    """Search for recent news about a company."""
    results = tavily.search(f"{company} latest news 2025", max_results=3)
    return str(results["results"])

# Wrap as ADK tool
search_tool = FunctionTool(func=search_company_news)

# Agent
agent = Agent(
    model="gemini-3.1-pro-preview",
    name="research_agent",
    instruction="""You are a company research agent. When given a company name:
    1. Use the search tool to find recent news
    2. Summarize the findings in 3 sentences
    3. Return a JSON object with:
       - company: company name
       - summary: one sentence summary
       - sentiment: positive/negative/neutral
       - top_facts: list of 3 key facts
    Return only raw JSON, no markdown.""",
    tools=[search_tool]
)

async def run_agent(company: str):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="research_agent",
        user_id="user_1",
        session_id="session_1"
    )
    
    runner = Runner(
        agent=agent,
        app_name="research_agent",
        session_service=session_service
    )
    
    message = Content(parts=[Part(text=company)])
    
    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message=message
    ):
        if event.is_final_response():
            response = event.content.parts[0].text
            # Strip markdown if present
            import re
            response = re.sub(r'^```json\s*', '', response.strip())
            response = re.sub(r'^```\s*', '', response)
            response = re.sub(r'\s*```$', '', response).strip()
            print(json.loads(response))

if __name__ == "__main__":
    asyncio.run(run_agent("Anthropic"))