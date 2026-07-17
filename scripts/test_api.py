#!/usr/bin/env python3
"""Phase 7 sanity check: drive the real FastAPI app (no mocks) through the
full flow — /search -> /summarize -> /chat -> /export -> /library — to prove
the wiring genuinely works end to end, not just each piece in isolation.

Uses FastAPI's TestClient in-process (no need to run a separate uvicorn
server), but hits every real function: the live agent, live embeddings,
live OpenAI summarization/chat calls, and the real SQLite file at
data/history.sqlite.

Usage:
    python scripts/test_api.py
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import research_agent.api as api_module

TOPIC = "in-context learning for large language models"


def main() -> None:
    with TestClient(api_module.app) as client:
        print(f"{'=' * 80}\nPOST /search {{'topic': {TOPIC!r}}}\n{'=' * 80}")
        resp = client.post("/search", json={"topic": TOPIC})
        print("status:", resp.status_code)
        assert resp.status_code == 200, f"/search failed: {resp.status_code} {resp.text}"
        body = resp.json()
        search_id = body["search_id"]
        assert body["papers"], "/search returned zero papers for a well-known topic"
        print(f"search_id={search_id}, {len(body['papers'])} papers, top result: {body['papers'][0]['title']}")

        print(f"\n{'=' * 80}\nPOST /summarize {{'search_id': {search_id}}}\n{'=' * 80}")
        resp = client.post("/summarize", json={"search_id": search_id})
        print("status:", resp.status_code)
        assert resp.status_code == 200, f"/summarize failed: {resp.status_code} {resp.text}"
        summary = resp.json()
        assert summary["themes"], "/summarize returned zero themes"
        print(f"{len(summary['themes'])} theme(s):")
        for theme in summary["themes"]:
            print(f"  - {theme['theme_name']} ({len(theme['papers'])} paper(s))")

        print(f"\n{'=' * 80}\nPOST /chat (first question)\n{'=' * 80}")
        resp = client.post("/chat", json={"search_id": search_id, "question": "What is the main idea discussed here?"})
        print("status:", resp.status_code)
        assert resp.status_code == 200, f"/chat failed: {resp.status_code} {resp.text}"
        chat1 = resp.json()
        assert chat1["answer"].strip(), "/chat returned an empty answer"
        assert len(chat1["history"]) == 2, f"expected 2 history entries after one turn, got {len(chat1['history'])}"
        print("answer:", chat1["answer"][:300])
        print("cited:", [p["title"] for p in chat1["cited_papers"]])

        print(f"\n{'=' * 80}\nPOST /chat (follow-up, carrying history forward)\n{'=' * 80}")
        resp = client.post("/chat", json={
            "search_id": search_id,
            "question": "Can you say more about that?",
            "history": chat1["history"],
        })
        print("status:", resp.status_code)
        assert resp.status_code == 200, f"/chat (follow-up) failed: {resp.status_code} {resp.text}"
        chat2 = resp.json()
        assert chat2["answer"].strip(), "/chat follow-up returned an empty answer"
        assert len(chat2["history"]) == 4, f"expected history to carry forward + grow to 4 entries, got {len(chat2['history'])}"
        print("answer:", chat2["answer"][:300])

        print(f"\n{'=' * 80}\nGET /export/{search_id}\n{'=' * 80}")
        resp = client.get(f"/export/{search_id}")
        print("status:", resp.status_code, "| content-type:", resp.headers["content-type"])
        assert resp.status_code == 200, f"/export failed: {resp.status_code} {resp.text}"
        assert f"# Literature Summary: {TOPIC}" in resp.text, "/export markdown is missing the expected title header"
        print(resp.text[:500], "...")

        print(f"\n{'=' * 80}\nGET /library\n{'=' * 80}")
        resp = client.get("/library")
        print("status:", resp.status_code)
        assert resp.status_code == 200, f"/library failed: {resp.status_code} {resp.text}"
        library_items = resp.json()
        assert any(item["search_id"] == search_id for item in library_items), (
            f"/library did not include the search we just created (search_id={search_id})"
        )
        for item in library_items:
            print(f"  [{item['search_id']}] {item['topic']} ({item['paper_count']} papers, summary={item['has_summary']})")

        print(f"\n{'=' * 80}\nGET /library/{search_id}\n{'=' * 80}")
        resp = client.get(f"/library/{search_id}")
        print("status:", resp.status_code, "| papers:", len(resp.json()["papers"]))
        assert resp.status_code == 200, f"/library/{{search_id}} failed: {resp.status_code} {resp.text}"
        assert resp.json()["papers"], "/library/{search_id} returned zero papers for a search that has papers"

        print(f"\n{'=' * 80}\nPASS: full /search -> /summarize -> /chat -> /export -> /library flow succeeded end to end.\n{'=' * 80}")


if __name__ == "__main__":
    main()
