from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
import operator
import os

# State definition
class AgentState(TypedDict):
    company: str
    search_results: str
    summary: str
    output: dict

# LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# Tool
search = TavilySearch(
    max_results=3,
    api_key=os.environ["TAVILY_API_KEY"]
)

# Nodes
def search_node(state: AgentState) -> AgentState:
    results = search.invoke(f"{state['company']} latest news 2025")
    state["search_results"] = str(results)
    return state

def summarize_node(state: AgentState) -> AgentState:
    response = llm.invoke(f"""
    Summarize this news about {state['company']} in 3 sentences:
    {state['search_results']}
    """)
    state["summary"] = response.content
    return state

def format_node(state: AgentState) -> AgentState:
    response = llm.invoke(f"""
    Based on this summary, return a JSON object with exactly these fields:
    - company: company name as string
    - summary: one sentence summary as string
    - sentiment: one of "positive", "negative", or "neutral"
    - top_facts: list of exactly 3 strings
    
    Summary: {state['summary']}
    
    Return only the raw JSON object. No markdown, no backticks, no explanation.
    Example format: {{"company": "X", "summary": "Y", "sentiment": "positive", "top_facts": ["a", "b", "c"]}}
    """)
    
    import json
    import re
    
    # Strip markdown code blocks if present
    content = response.content.strip()
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'^```\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()
    
    state["output"] = json.loads(content)
    return state

# Build graph
graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)
graph.add_node("format", format_node)

graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", "format")
graph.add_edge("format", END)

app = graph.compile()

# Run
if __name__ == "__main__":
    result = app.invoke({"company": "Anthropic"})
    print(result["output"])