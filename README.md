# LangGraph vs Google ADK — Same Agent, Two Frameworks

Built the same research agent twice. Same task, same tools, different frameworks.

**Task:** Given a company name → search the web → summarize findings → return structured JSON output.

---

## The Key Difference in One Slide

| | LangGraph | Google ADK |
|---|---|---|
| **Control model** | You define every node and edge | Agent decides tool usage from instructions |
| **State management** | Explicit `TypedDict` state | Handled internally by the session |
| **Async** | Optional | Native |
| **Tool definition** | LangChain tool wrappers | `FunctionTool(func=...)` |
| **Debugging** | Step-by-step via graph | Event stream |
| **Best for** | Deterministic multi-step pipelines | Fast prototyping in Google ecosystem |
| **LLM used** | Claude (via `langchain-anthropic`) | Gemini 3.1 Pro (via ADK) |
| **Lines of code** | ~75 | ~70 |

---

## LangGraph — You Wire Everything

```python
# State is explicit — you define what flows between nodes
class AgentState(TypedDict):
    company: str
    search_results: str
    summary: str
    output: dict

# You build the graph manually
graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)
graph.add_node("format", format_node)

graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", "format")
graph.add_edge("format", END)

app = graph.compile()
result = app.invoke({"company": "Anthropic"})
```

Every connection is explicit. You know exactly what runs when and in what order.

---

## Google ADK — Agent Decides

```python
# You describe behavior in natural language
agent = Agent(
    model="gemini-3.1-pro-preview",
    name="research_agent",
    instruction="""You are a company research agent. When given a company name:
    1. Use the search tool to find recent news
    2. Summarize in 3 sentences
    3. Return structured JSON with company, summary, sentiment, top_facts""",
    tools=[search_tool]  # Agent decides when and how to call this
)

# Async-native runner
async for event in runner.run_async(...):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

No graph. No edges. The agent reads the instruction and figures out the sequence.

---

## Output — Both Return the Same Thing

```json
{
  "company": "Anthropic",
  "summary": "Anthropic continues to expand through partnerships with Amazon and Google Cloud while addressing AI safety risks.",
  "sentiment": "positive",
  "top_facts": [
    "Partnered with Google Cloud to integrate Claude on Vertex AI",
    "Released report warning about agentic AI misuse in cybercrime",
    "Major Amazon deal significantly boosted company valuation"
  ]
}
```

---

## When to Use Which

**Use LangGraph when:**
- You need deterministic, auditable pipelines
- Failure modes matter (production ML systems)
- You're building multi-agent orchestration with conditional branching
- You want full control over retry logic and state

**Use Google ADK when:**
- You're prototyping fast
- You're already in the Google/Gemini/GCP ecosystem
- The agent task is well-defined enough to describe in a paragraph
- You want built-in session management without extra setup

---

## Stack

- **LangGraph version:** `langgraph` + `langchain-anthropic` + `langchain-tavily`
- **ADK version:** `google-adk` + `google-generativeai` + `tavily-python`
- **Search:** Tavily API (free tier)
- **LLMs:** Claude Haiku (LangGraph) · Gemini 3.1 Pro Preview (ADK)

---

## Run It

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install langgraph langchain langchain-anthropic langchain-tavily google-adk tavily-python

# Set keys
export ANTHROPIC_API_KEY=your_key
export GEMINI_API_KEY=your_key
export TAVILY_API_KEY=your_key

# LangGraph version
python langgraph/agent.py

# Google ADK version
python google_adk/agent.py
```

---

## What I Learned

LangGraph forces you to think in graphs — which is actually useful when your pipeline has conditional branches or parallel steps. The verbosity pays off in debuggability.

ADK is genuinely faster to get running. The instruction-based approach means less boilerplate. The tradeoff is less predictability — the agent can surprise you with how it sequences tool calls.

Neither is better. They solve different problems.
