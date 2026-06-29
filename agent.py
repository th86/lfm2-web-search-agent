#!/usr/bin/env python3
"""Standalone web-search + summarize agent powered by a locally hosted LFM2.5-230M.

The model runs entirely on your machine via 🤗 transformers (no API key needed).
Web search uses DuckDuckGo (no API key); page text is extracted with trafilatura.

Usage:
    uv run agent.py "What happened in the latest SpaceX launch?"
    uv run agent.py            # interactive prompt

Pipeline (a small model can't reliably drive a tool-calling loop, so we use a
fixed agentic pipeline instead):
    1. The LFM rewrites the question into a focused web-search query.
    2. DuckDuckGo returns the top results.
    3. We fetch and extract the readable text of each result page.
    4. The LFM writes a grounded, cited answer from that context.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

MODEL_ID = "LiquidAI/LFM2.5-230M"

# How many search results to fetch and feed to the model as context.
DEFAULT_NUM_RESULTS = 4
# Cap per-page text so the tiny context window isn't blown out.
MAX_CHARS_PER_PAGE = 1500


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LocalLFM:
    """Thin wrapper around the locally hosted LFM2.5 model."""

    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[agent] loading {model_id} (first run downloads weights)...",
              file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
            # attn_implementation="flash_attention_2"  # uncomment on a compatible GPU
        )
        print(f"[agent] model ready on {self.model.device}.", file=sys.stderr)

    def chat(self, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.3, stream: bool = False) -> str:
        """Run one user turn through the chat template and return the reply."""
        from transformers import TextStreamer

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )["input_ids"].to(self.model.device)

        streamer = (TextStreamer(self.tokenizer, skip_prompt=True,
                                 skip_special_tokens=True) if stream else None)

        output = self.model.generate(
            input_ids,
            do_sample=temperature > 0,
            temperature=temperature,
            top_k=50,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def web_search(query: str, num_results: int = DEFAULT_NUM_RESULTS) -> list[dict]:
    """Return a list of {title, href, body} dicts from DuckDuckGo."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=num_results))


def fetch_page_text(url: str, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    """Download a URL and extract its main readable text."""
    import requests
    import trafilatura

    try:
        resp = requests.get(
            url, timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; lfm2-agent/1.0)"},
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network is best-effort
        return f"[could not fetch page: {exc}]"

    text = trafilatura.extract(resp.text) or ""
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return text or "[no extractable text]"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
def make_search_query(llm: LocalLFM, question: str) -> str:
    """Ask the model to turn a natural-language question into a search query."""
    prompt = (
        "Rewrite the following question into a short, focused web search query. "
        "Reply with ONLY the query text, no quotes or explanation.\n\n"
        f"Question: {question}"
    )
    query = llm.chat(prompt, max_new_tokens=40, temperature=0.1)
    # Keep the first line only; strip echoed prefixes/quotes the small model
    # sometimes emits. Fall back to the original question if empty.
    query = query.splitlines()[0].strip().strip('"').strip()
    for prefix in ("Query:", "Search query:", "Question:", "Q:"):
        if query.lower().startswith(prefix.lower()):
            query = query[len(prefix):].strip()
    return query or question


def build_context(results: list[dict]) -> str:
    """Format fetched sources into a numbered context block for the model."""
    blocks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(untitled)")
        url = r.get("href", "")
        body = r.get("page_text") or r.get("body", "")
        blocks.append(f"[{i}] {title}\nURL: {url}\n{body}")
    return "\n\n".join(blocks)


def summarize(llm: LocalLFM, question: str, context: str, stream: bool = True) -> str:
    """Produce a grounded answer citing the numbered sources."""
    prompt = (
        "You are a research assistant. Using ONLY the sources below, answer the "
        "user's question concisely. Cite sources inline like [1], [2]. If the "
        "sources do not contain the answer, say so.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return llm.chat(prompt, max_new_tokens=512, temperature=0.3, stream=stream)


def run(question: str, num_results: int = DEFAULT_NUM_RESULTS,
        model_id: str = MODEL_ID) -> None:
    llm = LocalLFM(model_id)

    query = make_search_query(llm, question)
    print(f"\n🔎 Search query: {query}\n", file=sys.stderr)

    results = web_search(query, num_results=num_results)
    if not results:
        print("No search results found.", file=sys.stderr)
        return

    print("📄 Fetching sources:", file=sys.stderr)
    for r in results:
        url = r.get("href", "")
        print(f"   - {r.get('title', '')}\n     {url}", file=sys.stderr)
        r["page_text"] = fetch_page_text(url)

    context = build_context(results)

    print("\n💡 Answer:\n", file=sys.stderr)
    answer = summarize(llm, question, context, stream=True)
    if not answer:  # streamer already printed; this is a fallback
        print(answer)

    print("\n\nSources:", file=sys.stderr)
    for i, r in enumerate(results, start=1):
        print(f"  [{i}] {r.get('href', '')}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locally hosted LFM2.5-230M web-search + summary agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              uv run agent.py "Who won the 2024 Nobel Prize in Physics?"
              uv run agent.py -n 6 "latest news on the Artemis program"
        """),
    )
    parser.add_argument("question", nargs="*", help="the question to research")
    parser.add_argument("-n", "--num-results", type=int,
                        default=DEFAULT_NUM_RESULTS,
                        help=f"number of web results (default {DEFAULT_NUM_RESULTS})")
    parser.add_argument("--model", default=MODEL_ID, help="HF model id to load")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("Ask me anything: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not question:
        parser.error("no question provided")

    run(question, num_results=args.num_results, model_id=args.model)


if __name__ == "__main__":
    main()
