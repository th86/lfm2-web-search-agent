# LFM2.5 Web Search Agent

A standalone agent that answers questions by searching the web and summarizing
the results — running the **[LiquidAI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)**
language model **entirely on your local machine** (no LLM API key required).

It searches DuckDuckGo, extracts the readable text of the top results, and uses
the local model to write a concise, source-cited answer.

## How it works

A 230M-parameter model is too small to reliably drive an open-ended
tool-calling loop, so the agent uses a fixed, dependable pipeline:

1. **Query rewrite** — the LFM turns your question into a focused search query.
2. **Search** — DuckDuckGo returns the top results (no API key needed).
3. **Fetch & extract** — each result page's main text is pulled with `trafilatura`.
4. **Summarize** — the LFM writes a grounded answer with inline `[1]`, `[2]` citations.

## Requirements

- [uv](https://docs.astral.sh/uv/) (handles Python + dependencies)
- ~1 GB disk for the model weights (downloaded automatically on first run)
- Runs on CPU; uses a GPU automatically if available

## Quick start

```bash
# from this directory
uv run agent.py "Who won the 2024 Nobel Prize in Physics?"
```

Or run it interactively:

```bash
uv run agent.py
```

### Options

```bash
uv run agent.py -n 6 "latest news on the Artemis program"   # more sources
uv run agent.py --model LiquidAI/LFM2.5-230M "your question" # pick a model
```

## Notes

- The first run downloads the model from Hugging Face and caches it locally;
  subsequent runs are offline for the model itself (web search still needs a
  connection).
- Output quality is bounded by a 230M model — answers are short and best for
  factual lookups. Swap `--model` for a larger LFM2 variant for richer answers.

## License

MIT
