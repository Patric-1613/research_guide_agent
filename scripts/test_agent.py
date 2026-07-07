#!/usr/bin/env python3
"""Phase 4 sanity check: run the LangChain agent on a topic and print every
tool call / decision it makes as it makes them, not just the final answer.

Defaults to an acronym-heavy topic ("PEFT methods for LLMs") to see whether
the agent reformulates it (e.g. expands PEFT to "parameter-efficient
fine-tuning") before hitting arXiv/Semantic Scholar's literal keyword search.

Usage:
    python scripts/test_agent.py ["<topic>"]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_agent.agent import run_research_agent

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "PEFT methods for LLMs"


def log_step(message) -> None:
    if isinstance(message, HumanMessage):
        print(f"\n[USER] {message.content}")
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"\n[AGENT DECISION] call {call['name']}({call['args']})")
        if message.content:
            print(f"\n[AGENT] {message.content}")
    elif isinstance(message, ToolMessage):
        preview = message.content if len(message.content) < 500 else message.content[:500] + "..."
        print(f"[TOOL RESULT <- {message.name}]\n{preview}")


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    print(f"{'=' * 80}\nTopic: {topic!r}\n{'=' * 80}")
    session = run_research_agent(topic, s2_api_key=s2_key, on_step=log_step)

    print(f"\n{'=' * 80}\nFinal working pool: {len(session.papers)} paper(s)")
    print(f"Final ranking: {len(session.ranked)} paper(s)\n{'=' * 80}")
    for i, (p, score) in enumerate(session.ranked, 1):
        print(f"[{i}] ({score:.3f}) {p.title}  [{p.source}]")


if __name__ == "__main__":
    main()
