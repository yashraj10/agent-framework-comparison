## Benchmark Results

30 runs each · Same task · Same output · 0 errors

| Metric | LangGraph | Google ADK |
|--------|-----------|------------|
| Mean | 4.42s | 10.17s |
| Median | 4.20s | 10.03s |
| Stdev | 1.01s | 1.54s |
| Speedup | 2.3x faster | baseline |
| Est. cost/run | ~$0.005 | ~$0.008 (+53%) |

**Why the gap exists:** LangGraph executes a predefined graph — no reasoning overhead. ADK reads the instruction and decides how to use tools at runtime, which means more LLM calls per task.

Run the benchmark yourself:
```bash
python benchmark.py
```
Raw results in `benchmark_results.json`.