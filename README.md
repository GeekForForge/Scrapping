# God Level Scraper v3 🚀

> **Focused Platforms Web Scraper** - A powerful FastAPI-based scraper designed for extracting resources from major coding and learning platforms.

## Overview

God Level Scraper v3 is an intelligent web scraper built with **FastAPI** and **Python** that focuses on scraping content from coding resource platforms. This variant restricts scraping to only well-established, trusted platforms including GitHub, GeeksforGeeks, LeetCode, StackOverflow, and W3Schools.

## Features ✨

- **Multi-Platform Support**: Simultaneously search across multiple platforms
  - GitHub (Repository search)
  - GeeksforGeeks (Tutorials & Articles)
  - LeetCode (Coding Problems)
  - StackOverflow (Q&A)
  - W3Schools (Web Development Resources)

- **Smart Fetching & Processing**:
  - Automatic retry logic with exponential backoff
  - Rotating user agents to avoid detection
  - JavaScript rendering support via Playwright (optional)
  - Timeout handling and resilient error recovery

- **Content Extraction**:
  - Intelligent metadata extraction (title, description, OpenGraph tags)
  - Main text extraction using Readability library
  - Extractive summarization with TF-IDF scoring
  - Optional abstractive summarization via OpenRouter API

- **Caching & Performance**:
  - In-memory LRU caching (via cachetools)
  - Redis support for distributed caching (optional)
  - 30-minute default TTL for cached results
  - Concurrent fetching with semaphore control

- **Topic-Level Summarization**:
  - Aggregates content across multiple results
  - Generates both extractive and abstractive summaries
  - OpenRouter integration for advanced NLP summarization

## Tech Stack

- **Web Framework**: FastAPI 0.115.0
- **HTTP Client**: httpx 0.27.2
- **Server**: Uvicorn 0.30.6 (with Gunicorn support)
- **HTML Parsing**: BeautifulSoup4 4.12.3
- **XML Support**: lxml 5.3.0
- **Caching**: cachetools 5.5.0 (optional)
- **Redis Client**: redis 5.1.1 (optional)
- **Browser Automation**: Playwright 1.48.0 (optional)
- **Content Readability**: readability-lxml 0.8.1 (optional)
- **Concurrency**: Python 3.10+ asyncio

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or pip3

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GeekForForge/Scrapping.git
   cd Scrapping
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

For enhanced functionality, install optional packages:

```bash
# Content extraction from complex pages
pip install readability-lxml

# In-memory caching
pip install cachetools

# Distributed caching with Redis
pip install redis

# JavaScript rendering support
pip install playwright
pywright install chromium  # Download browser binary
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# OpenRouter API for abstractive summarization
OPENROUTER_API_KEY=your_api_key_here

# Redis connection string (if using Redis caching)
REDIS_URL=redis://localhost:6379/0
```

### API Key Setup

The OpenRouter API key is currently hardcoded in `app.py` for convenience. For production:
1. Remove the hardcoded key
2. Set the `OPENROUTER_API_KEY` environment variable
3. The application will use the env var automatically

## Usage

### Running the Server

```bash
# Development mode with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production mode with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```
GET /
```

**Response**:
```json
{
  "status": "ok",
  "message": "God Level Scraper v3 (focused) Running 🚀"
}
```

#### 2. Fetch Resources (Main Endpoint)
```
POST /fetch
```

**Request Body**:
```json
{
  "query": "machine learning"
}
```

**Query Parameters**:
- `platforms` (optional): Comma-separated list of platforms to search. Options: `gfg`, `so`, `gh`, `leetcode`, `w3`. Default: `gfg,so,gh,leetcode,w3`
- `limit` (optional): Results per platform. Default: `4`, Range: `1-8`
- `use_js` (optional): Enable JavaScript rendering. Default: `false`
- `summary_sentences` (optional): Number of sentences in summaries. Default: `3`, Range: `1-6`
- `use_cache` (optional): Enable caching. Default: `true`

**Example Request**:
```bash
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"query": "python asyncio"}' \
  -G --data-urlencode 'platforms=gfg,so,gh' \
  --data-urlencode 'limit=5'
```

**Response**:
```json
{
  "query": "machine learning",
  "count": 12,
  "results": {
    "GFG": [
      {
        "title": "Introduction to Machine Learning",
        "url": "https://www.geeksforgeeks.org/introduction-to-machine-learning/",
        "type": "GFG",
        "scraped": {
          "url": "https://www.geeksforgeeks.org/introduction-to-machine-learning/",
          "meta": {
            "title": "Introduction to Machine Learning - GeeksforGeeks",
            "description": "Machine Learning is a subset of AI..."
          },
          "snippet": "Machine learning is a subset of artificial intelligence...",
          "summary": "ML uses algorithms to learn patterns from data.",
          "word_count": 450
        }
      }
    ],
    "StackOverflow": [...],
    "GitHub": [...]
  },
  "topic_summary": "Machine learning involves algorithms that learn from data...",
  "meta": {
    "engines_queried": ["gfg", "so", "gh", "leetcode", "w3"],
    "use_js": false,
    "cache_used": true,
    "summary_type": "abstractive"
  }
}
```

## How It Works

### Architecture Flow

1. **Request Input**: User sends a query with optional filters
2. **Platform Search**: Concurrent searches across selected platforms
3. **URL Collection**: URLs are deduplicated and merged
4. **Cache Check**: Check in-memory and Redis cache
5. **Fetching**: Parallel fetches with concurrency control
6. **Processing**: Extract metadata, text, and generate summaries
7. **Caching**: Store results in cache with TTL
8. **Aggregation**: Group results by platform
9. **Topic Summary**: Generate cross-platform summary
10. **Response**: Return structured results

### Platform-Specific Handlers

**GeeksforGeeks**: Attempts direct URL first, then searches and scrapes results

**StackOverflow**: Searches questions, extracts best matches with scoring

**GitHub**: Searches repositories using GitHub's search interface

**LeetCode**: Uses curated problem mappings with fallback search

**W3Schools**: Searches educational content, builds resource links

### Content Extraction Strategy

1. **Metadata**: Extracts title, description, OpenGraph tags, JSON-LD
2. **Main Text**:
   - Attempts Readability extraction
   - Falls back to `<article>` tag content
   - Final fallback to largest `<div>` heuristic
3. **Summary**:
   - Extractive: TF-IDF scoring of sentences
   - Abstractive: OpenRouter API (if configured)

## Behavior Changes from v2

- ✅ **Platform Restriction**: Only queries coding/resource platforms
- ✅ **Lower Noise**: Removed generic search engines
- ✅ **Optimized Limits**: Smaller per-platform results
- ✅ **Better Caching**: Improved in-memory + Redis support
- ✅ **Optional JS Rendering**: Playwright support for JS-heavy sites
- ✅ **Smarter Concurrency**: Reduced from 10 to 6 concurrent requests

## Important Security Notes ⚠️

1. **API Keys**: The OpenRouter key is hardcoded. For production:
   - Remove the hardcoded value
   - Use environment variables
   - Rotate keys regularly

2. **CORS Settings**: Currently allows all origins (`*`). Restrict in production:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

3. **Rate Limiting**: Consider adding rate limiting middleware for production

4. **User Agents**: Rotating user agents help avoid blocks but respect `robots.txt`

## Troubleshooting

### Connection Errors

**Problem**: `Connection refused` or timeout errors

**Solution**: Check internet connectivity and platform availability

### Empty Results

**Problem**: Getting empty results for valid queries

**Solution**:
1. Check if platforms are accessible
2. Try enabling JavaScript rendering (`use_js=true`)
3. Verify query terms are valid
4. Check rate limiting status of platforms

### Cache Issues

**Problem**: Stale cached data

**Solution**:
- Set `use_cache=false` for fresh results
- Clear Redis: `redis-cli FLUSHDB`
- Check TTL: Default is 30 minutes

### Playwright Installation

**Problem**: Playwright not installing correctly

**Solution**:
```bash
pip install --upgrade playwright
pywright install chromium
```

## Performance Optimization

- **Concurrent Requests**: Limited to 6 simultaneous fetches
- **Timeout**: 12 seconds per URL, 18 seconds total per request
- **Caching**: 30-minute TTL reduces redundant fetches
- **Selective Parsing**: Only extracts necessary data
- **Async/Await**: Full asynchronous processing

## File Structure

```
Scrapping/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── __pycache__/           # Python cache (auto-generated)
```

## Testing

Test the API with sample queries:

```bash
# Python Resources
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"query": "python decorators"}'

# Web Development
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"query": "CSS flexbox"}' \
  -G --data-urlencode 'platforms=w3'

# Coding Problems
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"query": "binary search"}' \
  -G --data-urlencode 'platforms=leetcode'
```

## Browser Testing

Access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Known Limitations

- Platform rate limits may trigger 429 errors
- Some sites require authentication
- JavaScript-heavy sites need Playwright enabled
- OpenRouter rate limits apply to summarization
- Redis requires separate infrastructure setup

## Future Improvements

- [ ] GraphQL API support
- [ ] Webhook notifications
- [ ] Advanced filtering and sorting
- [ ] User authentication
- [ ] Analytics dashboard
- [ ] Batch processing
- [ ] Export to multiple formats (PDF, CSV)

## License

This project is part of the GeekForForge initiative.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ by GeekForForge**
