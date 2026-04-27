import asyncio
import time
import json
import os
import statistics
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from google import genai
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from tavily import TavilyClient
import re

# ── 200 company inputs ──────────────────────────────────────────────
COMPANIES = [
    "Anthropic", "OpenAI", "Google", "Microsoft", "Apple",
    "Amazon", "Meta", "Tesla", "Nvidia", "Stripe",
    "Airbnb", "Uber", "Lyft", "DoorDash", "Instacart",
    "Spotify", "Netflix", "Adobe", "Salesforce", "Snowflake",
    "Databricks", "Palantir", "Cloudflare", "Twilio", "Okta",
    "Zoom", "Slack", "Notion", "Figma", "Canva",
    "Shopify", "Square", "PayPal", "Robinhood", "Coinbase",
    "SpaceX", "Rivian", "Lucid", "Waymo", "Cruise",
    "DeepMind", "Mistral", "Cohere", "Hugging Face", "Scale AI",
    "Runway", "Midjourney", "Stability AI", "Inflection AI", "xAI",
    "Samsung", "Sony", "LG", "Intel", "AMD",
    "Qualcomm", "TSMC", "ASML", "Arm", "Broadcom",
    "Oracle", "SAP", "IBM", "Dell", "HP",
    "Cisco", "VMware", "ServiceNow", "Workday", "Zendesk",
    "HubSpot", "Intercom", "Atlassian", "GitHub", "GitLab",
    "MongoDB", "Redis", "Elastic", "HashiCorp", "Confluent",
    "Datadog", "New Relic", "Splunk", "PagerDuty", "Dynatrace",
    "Vercel", "Netlify", "Supabase", "PlanetScale", "Neon",
    "Brex", "Ramp", "Plaid", "Chime", "Nubank",
    "Klarna", "Affirm", "Afterpay", "Toast", "Faire",
    "Rippling", "Gusto", "ADP", "Workday", "BambooHR",
    "Duolingo", "Coursera", "Udemy", "Chegg", "Khan Academy",
    "Peloton", "Calm", "Headspace", "Noom", "Hims",
    "Oscar Health", "Ro", "Forward", "Carbon Health", "Teladoc",
    "Epic Systems", "Veeva", "Medidata", "Flatiron Health", "Tempus",
    "Palantir", "C3.ai", "UiPath", "Automation Anywhere", "Blue Prism",
    "Twitch", "Discord", "Reddit", "Pinterest", "Snap",
    "TikTok", "ByteDance", "Baidu", "Alibaba", "Tencent",
    "Grab", "Gojek", "Sea Limited", "Coupang", "Mercado Libre",
    "Rappi", "Nubank", "PagSeguro", "Stone", "Totvs",
    "Arm Holdings", "SoftBank", "Rakuten", "Line", "Kakao",
    "Naver", "SK Hynix", "MediaTek", "Foxconn", "CATL",
    "BYD", "NIO", "Li Auto", "Xpeng", "Zeekr",
    "Northvolt", "Lilium", "Volocopter", "Joby Aviation", "Archer",
    "Relativity Space", "Rocket Lab", "Planet Labs", "Maxar", "Spire",
    "Samsara", "Mobileye", "Luminar", "Innoviz", "Ouster",
    "Lemonade", "Root", "Hippo", "Kin Insurance", "Branch",
    "Carta", "Angellist", "Forge", "Nasdaq Private Market", "EquityZen",
    "Airtable", "Coda", "Monday.com", "Asana", "Linear",
    "Miro", "Loom", "Descript", "Grain", "Otter.ai",
    "Grammarly", "Jasper", "Copy.ai", "Writer", "Wordtune",
    "Synthesia", "HeyGen", "D-ID", "ElevenLabs", "Murf",
    "Pinecone", "Weaviate", "Qdrant", "Chroma", "Milvus",
    "LangChain", "LlamaIndex", "Weights and Biases", "MLflow", "Comet"
]

# ── LangGraph setup ──────────────────────────────────────────────────
class AgentState(TypedDict):
    company: str
    search_results: str
    summary: str
    output: dict

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
search = TavilySearch(
    max_results=3,
    api_key=os.environ["TAVILY_API_KEY"]
)

def search_node(state):
    results = search.invoke(f"{state['company']} latest news 2025")
    state["search_results"] = str(results)
    return state

def summarize_node(state):
    response = llm.invoke(f"Summarize this news about {state['company']} in 3 sentences: {state['search_results']}")
    state["summary"] = response.content
    return state

def format_node(state):
    response = llm.invoke(f"""
    Return a JSON object with: company, summary (1 sentence), sentiment (positive/negative/neutral), top_facts (3 strings).
    Summary: {state['summary']}
    Return only raw JSON. No markdown.
    """)
    content = re.sub(r'^```json\s*|^```\s*|\s*```$', '', response.content.strip()).strip()
    state["output"] = json.loads(content)
    return state

graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)
graph.add_node("format", format_node)
graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", "format")
graph.add_edge("format", END)
lg_app = graph.compile()

def run_langgraph(company):
    start = time.time()
    try:
        result = lg_app.invoke({"company": company})
        return time.time() - start, True
    except Exception as e:
        return time.time() - start, False

# ── ADK setup ────────────────────────────────────────────────────────
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search_company_news(company: str) -> str:
    """Search for recent news about a company."""
    results = tavily_client.search(f"{company} latest news 2025", max_results=3)
    return str(results["results"])

search_tool = FunctionTool(func=search_company_news)

adk_agent = Agent(
    model="gemini-3.1-pro-preview",
    name="research_agent",
    instruction="""Research the given company and return a JSON object with:
    - company: company name
    - summary: one sentence summary
    - sentiment: positive/negative/neutral
    - top_facts: list of 3 key facts
    Use the search tool. Return only raw JSON.""",
    tools=[search_tool]
)

async def run_adk(company):
    start = time.time()
    try:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="research_agent",
            user_id="user_1",
            session_id=f"session_{company}"
        )
        runner = Runner(
            agent=adk_agent,
            app_name="research_agent",
            session_service=session_service
        )
        message = Content(parts=[Part(text=company)])
        async for event in runner.run_async(
            user_id="user_1",
            session_id=f"session_{company}",
            new_message=message
        ):
            if event.is_final_response():
                return time.time() - start, True
        return time.time() - start, False
    except Exception:
        return time.time() - start, False

# ── Benchmark runner ─────────────────────────────────────────────────
def run_benchmark(n=200):
    companies = COMPANIES[:n]
    
    print(f"\n{'='*50}")
    print(f"Benchmarking {n} companies")
    print(f"{'='*50}\n")

    # LangGraph benchmark
    print("Running LangGraph...")
    lg_times = []
    lg_errors = 0
    for i, company in enumerate(companies):
        t, success = run_langgraph(company)
        lg_times.append(t)
        if not success:
            lg_errors += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n} done | avg: {statistics.mean(lg_times):.2f}s")

    # ADK benchmark
    print("\nRunning Google ADK...")
    adk_times = []
    adk_errors = 0
    for i, company in enumerate(companies):
        t, success = asyncio.run(run_adk(company))
        adk_times.append(t)
        if not success:
            adk_errors += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n} done | avg: {statistics.mean(adk_times):.2f}s")

    # Results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"\nLangGraph ({n} runs):")
    print(f"  Mean:   {statistics.mean(lg_times):.2f}s")
    print(f"  Median: {statistics.median(lg_times):.2f}s")
    print(f"  Stdev:  {statistics.stdev(lg_times):.2f}s")
    print(f"  Errors: {lg_errors}/{n}")

    print(f"\nGoogle ADK ({n} runs):")
    print(f"  Mean:   {statistics.mean(adk_times):.2f}s")
    print(f"  Median: {statistics.median(adk_times):.2f}s")
    print(f"  Stdev:  {statistics.stdev(adk_times):.2f}s")
    print(f"  Errors: {adk_errors}/{n}")

    print(f"\nSpeedup: LangGraph is {statistics.mean(adk_times)/statistics.mean(lg_times):.1f}x faster")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "n": n,
            "langgraph": {"times": lg_times, "errors": lg_errors},
            "adk": {"times": adk_times, "errors": adk_errors}
        }, f)
    print("\nSaved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark(n=30)
    