# app.py
"""
God Level Scraper v3 — Focused platforms (drop-in)
This variant restricts scraping to only coding/resource platforms:
- GitHub, GeeksforGeeks, LeetCode, StackOverflow, W3Schools

Behavior changes from previous:
- Only queries the above platforms (less noisy)
- Smaller per-platform result limits and lower concurrency
- Keeps extractive + optional OpenRouter abstractive summarization
- Keeps caching hooks (in-memory + optional Redis) and Playwright fallback (optional)

IMPORTANT:
- You already hardcoded an OpenRouter key below. Keep it secure.
- If you want to change platforms later, edit the `wanted` parsing in /fetch.

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import os
import re
import math
import json
import heapq
import asyncio
from typing import List, Dict, Any, Optional, Iterable
from urllib.parse import quote

from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import httpx
from bs4 import BeautifulSoup

# Optional libs (not required)
try:
    from readability import Document as ReadabilityDocument
    READABILITY_AVAILABLE = True
except Exception:
    READABILITY_AVAILABLE = False

try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except Exception:
    CACHE_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# ------------------------------
# OpenRouter Summarizer Config
# ------------------------------
# NOTE: you already placed the key here per your earlier message.
# If you prefer env var, replace the string below with os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "gpt-4o-mini"
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

app = FastAPI(title="God Level Smart Scraper v3 (focused)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# In-memory cache (LRU TTL) if cachetools installed
if CACHE_AVAILABLE:
    LRU_CACHE = TTLCache(maxsize=1024, ttl=60 * 30)  # 30 min default
else:
    LRU_CACHE = {}

# Redis config (optional)
redis = None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


async def init_redis():
    global redis
    if REDIS_AVAILABLE and redis is None:
        redis = await aioredis.from_url(REDIS_URL)


def _choose_ua():
    import random
    return random.choice(USER_AGENTS)


async def fetch_url(client: httpx.AsyncClient, url: str, use_js: bool = False, timeout: int = 12) -> Optional[httpx.Response]:
    """
    Fetch URL with retries; optionally fallback to Playwright JS render.
    Returns an object with .status_code and .text/.content on success, otherwise None.
    """
    headers = DEFAULT_HEADERS.copy()
    headers["User-Agent"] = _choose_ua()
    max_retries = 3
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = await client.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and (resp.text or resp.content):
                return resp
            if resp.status_code in (429, 503):
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                await asyncio.sleep(0.3)
        except (httpx.RequestError, httpx.ReadTimeout):
            await asyncio.sleep(backoff)
            backoff *= 2

    # JS rendering fallback
    if use_js and PLAYWRIGHT_AVAILABLE:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=_choose_ua(), extra_http_headers=DEFAULT_HEADERS)
                await page.goto(url, timeout=15000)
                await page.wait_for_timeout(800)
                content = await page.content()
                await browser.close()

                class Dummy:
                    def __init__(self, text):
                        self.status_code = 200
                        self.text = text
                        self.content = text.encode("utf8", errors="ignore")
                return Dummy(content)
        except Exception:
            return None

    return None


def extract_metadata(html: str, url: str) -> Dict[str, Any]:
    """Extract title, meta description, OpenGraph, JSON-LD, canonical."""
    soup = BeautifulSoup(html, "html.parser")
    meta: Dict[str, Any] = {}
    if soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()
    for tag in soup.find_all("meta"):
        prop = tag.get("property") or tag.get("name") or ""
        content = tag.get("content")
        if not content:
            continue
        if prop.startswith("og:") or prop.startswith("twitter:"):
            meta[prop] = content
        if prop.lower() == "description":
            meta["description"] = content
    ld_jsons = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(script.string or "{}")
            ld_jsons.append(ld)
        except Exception:
            continue
    if ld_jsons:
        meta["json_ld"] = ld_jsons
    can = soup.find("link", rel="canonical")
    if can and can.get("href"):
        meta["canonical"] = can.get("href")
    meta["final_url"] = url
    return meta


def extract_main_text(html: str) -> str:
    """Try readability; fallback to <article> or largest <div> heuristic; else body text."""
    if READABILITY_AVAILABLE:
        try:
            doc = ReadabilityDocument(html)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text and len(text.split()) > 20:
                return text
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
        if len(text.split()) > 30:
            return text

    candidates = soup.find_all(["div", "main", "section"], recursive=True)
    best = ""
    for c in candidates:
        txt = c.get_text(separator="\n", strip=True)
        if len(txt) > len(best):
            best = txt
    if best:
        return best
    return soup.get_text(separator="\n", strip=True)


def simple_summarize(text: str, max_sentences: int = 3) -> str:
    """Extractive per-document summary: return top sentences (heuristic)."""
    if not text:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', text.replace("\n", " ").strip())
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) <= max_sentences:
        return " ".join(sents).strip()
    scored = sorted(sents, key=lambda s: len(s), reverse=True)[:max_sentences]
    scored_sorted = sorted(scored, key=lambda s: sents.index(s))
    return " ".join(scored_sorted)


async def fetch_and_process(client: httpx.AsyncClient, url: str, use_js: bool = False, summary_sentences: int = 3) -> Dict[str, Any]:
    """Fetch URL and extract metadata, main text, snippet, and summary."""
    resp = await fetch_url(client, url, use_js=use_js)
    if not resp:
        return {"url": url, "error": "failed_fetch"}
    html = resp.text if hasattr(resp, "text") else resp.content.decode("utf8", errors="ignore")
    meta = extract_metadata(html, url)
    main_text = extract_main_text(html)
    short_summary = simple_summarize(main_text, max_sentences=summary_sentences)
    snippet = (main_text[:300] + "...") if len(main_text) > 300 else main_text
    return {
        "url": url,
        "meta": meta,
        "snippet": snippet,
        "summary": short_summary,
        "word_count": len(main_text.split())
    }


# -----------------------
# Platform search helpers (focused)
# -----------------------
async def search_geeksforgeeks(client: httpx.AsyncClient, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    results = []
    slug = query.lower().replace(" ", "-")
    direct = f"https://www.geeksforgeeks.org/{slug}/"
    resp = await fetch_url(client, direct)
    if resp:
        results.append({"title": f"{query.title()} - GeeksforGeeks", "url": direct, "type": "GFG"})
    search_url = f"https://www.geeksforgeeks.org/search/?q={quote(query)}"
    resp = await fetch_url(client, search_url)
    if resp:
        soup = BeautifulSoup(resp.content, "html.parser")
        links = soup.select("div.gsc-webResult a.gs-title") or soup.select("div.search-post a") or soup.select("a.gs-title")
        for a in links[:limit]:
            href = a.get("href")
            title = a.get_text(strip=True)
            if href and "geeksforgeeks.org" in href:
                results.append({"title": title or "GFG article", "url": href, "type": "GFG"})
    await asyncio.sleep(0.6)
    return results


async def search_stackoverflow(client: httpx.AsyncClient, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    results = []
    url = f"https://stackoverflow.com/search?q={quote(query)}"
    resp = await fetch_url(client, url)
    if resp:
        soup = BeautifulSoup(resp.content, "html.parser")
        questions = soup.select("div.s-post-summary") or soup.select("div.question-summary")
        for q in questions[:limit]:
            link_tag = q.select_one("h3 a.s-link") or q.select_one("a.question-hyperlink")
            if link_tag:
                href = link_tag.get("href")
                title = link_tag.get_text(strip=True)
                full = f"https://stackoverflow.com{href}" if href.startswith("/") else href
                results.append({"title": title, "url": full, "type": "StackOverflow"})
    await asyncio.sleep(0.4)
    return results


async def search_github(client: httpx.AsyncClient, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Lightweight GitHub repo search by HTML fallback. For production prefer GitHub API.
    """
    results = []
    url = f"https://github.com/search?q={quote(query)}&type=repositories"
    resp = await fetch_url(client, url)
    if resp:
        soup = BeautifulSoup(resp.content, "html.parser")
        repos = soup.select("a.v-align-middle") or soup.select("ul.repo-list li a.v-align-middle")
        for repo in repos[:limit]:
            href = repo.get("href")
            title = repo.get_text(strip=True)
            if href:
                full = f"https://github.com{href}" if href.startswith("/") else href
                results.append({"title": title or "GitHub Repo", "url": full, "type": "GitHub"})
    await asyncio.sleep(0.4)
    return results


async def search_leetcode(client: httpx.AsyncClient, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Curated LeetCode mapping + search link fallback.
    """
    results = []
    # small curated mapping for common keywords
    leet_map = {
        "binary search": [
            ("Binary Search", "https://leetcode.com/problems/binary-search/"),
            ("Search Insert Position", "https://leetcode.com/problems/search-insert-position/")
        ],
        "two sum": [("Two Sum", "https://leetcode.com/problems/two-sum/")],
        "tree": [("Binary Tree Inorder Traversal", "https://leetcode.com/problems/binary-tree-inorder-traversal/")],
    }
    ql = query.lower()
    for k, items in leet_map.items():
        if k in ql:
            for title, url in items[:limit]:
                results.append({"title": f"{title} - LeetCode", "url": url, "type": "LeetCode"})
    # always add search page link
    search_url = f"https://leetcode.com/problemset/all/?search={quote(query)}"
    results.append({"title": f"Search LeetCode for '{query}'", "url": search_url, "type": "LeetCode"})
    await asyncio.sleep(0.3)
    return results


async def search_w3schools(client: httpx.AsyncClient, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Search W3Schools search page and return results.
    """
    results = []
    search_url = f"https://www.w3schools.com/search/?q={quote(query)}"
    resp = await fetch_url(client, search_url)
    if resp:
        soup = BeautifulSoup(resp.content, "html.parser")
        # W3Schools search uses anchors to lessons; look for links containing w3schools.com/tags/ or /css/ etc.
        anchors = soup.select("a[href*='w3schools.com']") or soup.select("a")
        seen = set()
        for a in anchors:
            href = a.get("href")
            if not href:
                continue
            # make absolute if needed
            if href.startswith("/"):
                href = "https://www.w3schools.com" + href
            if "w3schools.com" in href and href not in seen:
                seen.add(href)
                title = a.get_text(strip=True) or href
                results.append({"title": title, "url": href, "type": "W3Schools"})
            if len(results) >= limit:
                break
    await asyncio.sleep(0.3)
    return results


# -----------------------
# Topic summarization helpers
# -----------------------
def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.replace("\n", " ").strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]


def tokenize(sentence: str) -> List[str]:
    return re.findall(r"\w+", sentence.lower())


def build_tf(sentences: Iterable[str]) -> Dict[str, int]:
    tf = {}
    for s in sentences:
        for t in tokenize(s):
            tf[t] = tf.get(t, 0) + 1
    return tf


def score_sentences(sentences: Iterable[str], tf: Dict[str, int]) -> List[tuple]:
    scores = []
    for i, s in enumerate(sentences):
        tokens = tokenize(s)
        if not tokens:
            scores.append((0.0, i))
            continue
        raw = sum(tf.get(t, 0) for t in tokens)
        norm = raw / math.sqrt(len(tokens))
        scores.append((norm, i))
    return scores


def aggregate_topic_summary(texts: Iterable[str], max_sentences: int = 5) -> str:
    all_sentences: List[str] = []
    for t in texts:
        all_sentences.extend(split_sentences(t))
    if not all_sentences:
        return ""
    tf = build_tf(all_sentences)
    scored = score_sentences(all_sentences, tf)
    top_n = min(max_sentences, len(all_sentences))
    top = heapq.nlargest(top_n, scored, key=lambda x: x[0])
    top_idxs = sorted([i for (_, i) in top])
    chosen = [all_sentences[i] for i in top_idxs]
    return " ".join(chosen)


async def openrouter_summarize(text: str, max_tokens: int = 250) -> str:
    """
    Abstractive summarization using OpenRouter Chat Completions endpoint.
    Requires OPENROUTER_API_KEY.
    """
    if not USE_OPENROUTER:
        return ""
    messages = [
        {"role": "system", "content": "You are an assistant that writes a concise technical summary."},
        {"role": "user", "content": f"Summarize the following content into a clear concise paragraph (technical terms allowed):\n\n{text}"}
    ]
    body = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                print("OpenRouter summarize error status:", resp.status_code, resp.text[:500])
                return ""
            data = resp.json()
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    first = choices[0]
                    if "message" in first and isinstance(first["message"], dict):
                        return first["message"].get("content", "").strip()
                    if "text" in first:
                        return first.get("text", "").strip()
            return ""
    except Exception as e:
        print("OpenRouter summarize exception:", e)
        return ""


# -----------------------
# Main /fetch endpoint (focused platforms)
# -----------------------
@app.post("/fetch")
async def fetch_resources(
    payload: Dict[str, Any] = Body(...),
    platforms: Optional[str] = Query(None, description="Comma list (gfg,so,gh,leetcode,w3)"),
    limit: int = Query(4, ge=1, le=8),
    use_js: bool = Query(False),
    summary_sentences: int = Query(3, ge=1, le=6),
    use_cache: bool = Query(True),
):
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in payload")

    if use_cache and REDIS_AVAILABLE:
        await init_redis()

    # default to only coding platforms
    wanted = set([p.strip().lower() for p in (platforms or "gfg,so,gh,leetcode,w3").split(",") if p.strip()])

    async with httpx.AsyncClient(follow_redirects=True, timeout=18) as client:
        tasks = []
        if "gfg" in wanted:
            tasks.append(search_geeksforgeeks(client, query, limit=limit))
        if "so" in wanted or "stackoverflow" in wanted:
            tasks.append(search_stackoverflow(client, query, limit=limit))
        if "gh" in wanted or "github" in wanted:
            tasks.append(search_github(client, query, limit=limit))
        if "leetcode" in wanted:
            tasks.append(search_leetcode(client, query, limit=limit))
        if "w3" in wanted or "w3schools" in wanted:
            tasks.append(search_w3schools(client, query, limit=limit))

        platform_results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        candidate_entries: List[Dict[str, Any]] = []
        for pl_res in platform_results_lists:
            if isinstance(pl_res, Exception):
                continue
            for e in pl_res:
                url = e.get("url")
                if url:
                    candidate_entries.append(e)

        # dedupe preserving order
        seen = set()
        unique_entries = []
        for e in candidate_entries:
            if e["url"] not in seen:
                seen.add(e["url"])
                unique_entries.append(e)

        # For each unique entry, check cache then fetch+process if needed
        fetch_jobs = []
        # limit number of entries to fetch to avoid jumble
        max_fetch = min(len(unique_entries), limit * len(tasks) if tasks else limit * 3)
        for e in unique_entries[:max_fetch]:
            url = e["url"]
            cache_key = f"scrape:{url}"
            cached_hit = None
            if use_cache:
                if CACHE_AVAILABLE:
                    try:
                        cached_hit = LRU_CACHE.get(cache_key)
                    except Exception:
                        cached_hit = None
                if not cached_hit and REDIS_AVAILABLE and redis:
                    try:
                        v = await redis.get(cache_key)
                        if v:
                            cached_hit = json.loads(v)
                    except Exception:
                        cached_hit = None
            if cached_hit:
                e["scraped"] = cached_hit
            else:
                fetch_jobs.append((e, fetch_and_process(client, url, use_js=use_js, summary_sentences=summary_sentences)))

        results: List[Dict[str, Any]] = []
        if fetch_jobs:
            sem = asyncio.Semaphore(6)  # lower concurrency for focused scraping

            async def run_with_sem(entry, coro):
                async with sem:
                    try:
                        return entry, await coro
                    except Exception as ex:
                        return entry, {"url": entry.get("url"), "error": str(ex)}

            run_coros = [run_with_sem(entry, coro) for entry, coro in fetch_jobs]
            done = await asyncio.gather(*run_coros, return_exceptions=False)
            for entry, scraped in done:
                entry["scraped"] = scraped
                # cache
                if use_cache:
                    cache_key = f"scrape:{entry['url']}"
                    if CACHE_AVAILABLE:
                        try:
                            LRU_CACHE[cache_key] = scraped
                        except Exception:
                            pass
                    if REDIS_AVAILABLE and redis:
                        try:
                            await redis.set(cache_key, json.dumps(scraped), ex=60 * 60)
                        except Exception:
                            pass
                results.append(entry)

        # include ones that were cached earlier and not in results list
        for e in unique_entries:
            if "scraped" in e and e not in results:
                results.append(e)

    # Group results (keeps only focused platforms)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(r.get("type", "Other"), []).append(r)

    # --------------------------
    # Aggregate texts for topic summary
    # --------------------------
    collected_texts: List[str] = []
    for entry in results:
        scraped = entry.get("scraped") or {}
        if scraped.get("summary"):
            collected_texts.append(scraped["summary"])
        elif scraped.get("snippet"):
            if len(scraped["snippet"]) > 40:
                collected_texts.append(scraped["snippet"])
        else:
            meta = scraped.get("meta") or {}
            if meta.get("description"):
                collected_texts.append(meta["description"])

    # Filter very short texts
    collected_texts = [t for t in collected_texts if t and len(t) > 30]

    topic_summary_extractive = aggregate_topic_summary(collected_texts, max_sentences=5)
    topic_summary_abstractive = ""
    if USE_OPENROUTER and collected_texts:
        big_text = " ".join(collected_texts)
        if len(big_text) > 30000:
            big_text = big_text[:30000]
        try:
            topic_summary_abstractive = await openrouter_summarize(big_text, max_tokens=300)
        except Exception as e:
            print("OpenRouter summarization failed:", e)
            topic_summary_abstractive = ""

    topic_summary_final = topic_summary_abstractive or topic_summary_extractive or "No summaryable content found."

    response = {
        "query": query,
        "count": len(results),
        "results": grouped,
        "topic_summary": topic_summary_final,
        "meta": {
            "engines_queried": list(wanted),
            "use_js": use_js,
            "cache_used": use_cache,
            "summary_type": "abstractive" if topic_summary_abstractive else "extractive",
        }
    }

    return response


@app.get("/")
async def home():
    return {"status": "ok", "message": "God Level Scraper v3 (focused) Running 🚀"}
# should be false yaar why is this

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
