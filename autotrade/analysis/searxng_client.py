"""
SearXNG Search Client
=====================
Provides web search capability via local SearXNG Docker instance.
No external APIs required - all searches go through your private instance.

Usage:
    from autotrade.analysis.searxng_client import SearXNGClient
    
    client = SearXNGClient()
    results = client.search("NVDA stock news")
    
    # Or async
    results = await client.search_async("Python trading libraries")
"""

import logging
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger("SearXNGClient")


def _config_value(section: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a dict config section or a pydantic model."""
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    content: str
    engine: str
    score: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.title}\n{self.url}\n{self.content[:200]}..."


@dataclass
class SearchResponse:
    """Full search response with metadata."""
    query: str
    results: List[SearchResult]
    suggestions: List[str]
    total_results: int
    search_time: float
    
    def to_text(self, max_results: int = 5) -> str:
        """Convert to readable text for LLM consumption."""
        lines = [f"Search results for: {self.query}", f"Found {self.total_results} results\n"]
        
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   {result.content[:300]}")
            lines.append("")
        
        if self.suggestions:
            lines.append(f"Related searches: {', '.join(self.suggestions[:5])}")
        
        return "\n".join(lines)


class SearXNGClient:
    """
    Client for SearXNG search API with failover support.
    
    Default assumes SearXNG is running on localhost:8080.
    Configure via config/trading_config.yaml or environment.
    """
    
    DEFAULT_HOST = "http://localhost:8080"
    
    def __init__(
        self,
        host: Optional[str] = None,
        timeout: int = 10,
        default_engines: Optional[List[str]] = None,
        safe_search: int = 0,  # 0=off, 1=moderate, 2=strict
        backup_hosts: Optional[List[str]] = None
    ):
        """
        Initialize SearXNG client with failover.
        """
        import os
        from config.config_loader import get_config
        
        try:
            cfg = get_config()
            search_cfg = getattr(cfg, "search", None)
            if search_cfg:
                self.host = host or os.environ.get(
                    "SEARXNG_HOST",
                    _config_value(search_cfg, "primary_host", self.DEFAULT_HOST),
                )
                self.backup_hosts = backup_hosts or _config_value(search_cfg, "backup_hosts", [])
                self.timeout = timeout or _config_value(search_cfg, "timeout", 10)
            else:
                self.host = host or os.environ.get("SEARXNG_HOST", self.DEFAULT_HOST)
                self.backup_hosts = backup_hosts or []
                self.timeout = timeout
        except Exception:
            self.host = host or os.environ.get("SEARXNG_HOST", self.DEFAULT_HOST)
            self.backup_hosts = backup_hosts or []
            self.timeout = timeout

        self.default_engines = default_engines or []
        self.safe_search = safe_search
        
        # Remove trailing slashes
        self.host = self.host.rstrip("/")
        self.backup_hosts = [h.rstrip("/") for h in self.backup_hosts]
        
        self.all_hosts = [self.host] + self.backup_hosts
        self.active_host_index = 0
        
        logger.info(f"SearXNGClient initialized. Primary: {self.host}, Backups: {len(self.backup_hosts)}")
    
    @property
    def current_host(self) -> str:
        return self.all_hosts[self.active_host_index]

    def check_connectivity(self) -> bool:
        """Check if any configured SearXNG host is reachable (P4)."""
        original_index = self.active_host_index
        for i in range(len(self.all_hosts)):
            self.active_host_index = i
            if self.health_check():
                if i != original_index:
                    logger.info(f"[SearXNG] Switched to healthy host: {self.current_host}")
                return True
        self.active_host_index = original_index
        return False

    def _build_params(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "en",
        page: int = 1
    ) -> Dict[str, Any]:
        """Build query parameters for search request."""
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": self.safe_search,
            "pageno": page,
        }
        
        # Add categories if specified
        if categories:
            params["categories"] = ",".join(categories)
        
        # Add engines if specified
        engines_to_use = engines or self.default_engines
        if engines_to_use:
            params["engines"] = ",".join(engines_to_use)
        
        return params
    
    def _parse_response(self, query: str, data: Dict) -> SearchResponse:
        """Parse SearXNG JSON response into SearchResponse."""
        results = []
        engine_counts = {}
        
        for item in data.get("results", []):
            engine = item.get("engine", "unknown")
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                engine=engine,
                score=item.get("score", 0.0)
            ))
        
        logger.info(f"Search engines summary: {engine_counts}")
        
        return SearchResponse(
            query=query,
            results=results,
            suggestions=data.get("suggestions", []),
            total_results=len(results),
            search_time=data.get("search_time", 0.0)
        )
    
    def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "en",
        page: int = 1,
        **kwargs
    ) -> SearchResponse:
        """Perform a synchronous search with failover (P4)."""
        max_results = kwargs.pop("max_results", None)
        limit = None
        if max_results is not None:
            try:
                limit = max(1, int(max_results))
            except Exception:
                limit = None

        params = self._build_params(query, categories, engines, language, page)
        
        # Failover loop (P4)
        for i in range(len(self.all_hosts)):
            # Try current active host, then rotate on failure
            host = self.all_hosts[(self.active_host_index + i) % len(self.all_hosts)]
            
            try:
                response = requests.get(
                    f"{host}/search",
                    params=params,
                    timeout=self.timeout,
                    headers={"Accept": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                # If we succeeded on a backup, update active index
                if host != self.host:
                    self.active_host_index = (self.active_host_index + i) % len(self.all_hosts)
                    logger.info(f"[SearXNG] Switched to active host: {host}")

                parsed = self._parse_response(query, data)
                if limit is not None:
                    parsed.results = parsed.results[:limit]
                    parsed.total_results = len(parsed.results)
                return parsed
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                logger.warning(f"[SearXNG] Host {host} failed: {type(e).__name__}. Trying next...")
                continue
            except Exception as e:
                logger.error(f"[SearXNG] Unexpected error on {host}: {e}")
                continue
                
        logger.error("[SearXNG] All search providers failed. Search returned empty.")
        return SearchResponse(
            query=query,
            results=[],
            suggestions=[],
            total_results=0,
            search_time=0.0
        )
    
    async def search_async(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "en",
        page: int = 1,
        **kwargs
    ) -> SearchResponse:
        """Perform an asynchronous search with failover (P4)."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, falling back to sync search")
            return self.search(query, categories, engines, language, page)
        
        max_results = kwargs.pop("max_results", None)
        limit = None
        if max_results is not None:
            try:
                limit = max(1, int(max_results))
            except Exception:
                limit = None

        params = self._build_params(query, categories, engines, language, page)
        
        # Failover loop (P4)
        for i in range(len(self.all_hosts)):
            host = self.all_hosts[(self.active_host_index + i) % len(self.all_hosts)]
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{host}/search",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"Accept": "application/json"}
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        
                        if host != self.host:
                            self.active_host_index = (self.active_host_index + i) % len(self.all_hosts)
                            logger.info(f"[SearXNG] Switched to active host: {host}")

                        parsed = self._parse_response(query, data)
                        if limit is not None:
                            parsed.results = parsed.results[:limit]
                            parsed.total_results = len(parsed.results)
                        return parsed
                        
            except Exception as e:
                logger.warning(f"[SearXNG] Async host {host} failed: {type(e).__name__}. Trying next...")
                continue
                    
        logger.error("[SearXNG] All async search providers failed.")
        return SearchResponse(
            query=query,
            results=[],
            suggestions=[],
            total_results=0,
            search_time=0.0
        )
    
    def search_news(self, query: str, **kwargs) -> SearchResponse:
        """Convenience method for news searches."""
        return self.search(query, categories=["news"], **kwargs)
    
    def search_images(self, query: str, **kwargs) -> SearchResponse:
        """Convenience method for image searches."""
        return self.search(query, categories=["images"], **kwargs)
    
    def search_finance(self, query: str, **kwargs) -> SearchResponse:
        """Convenience method for finance/stock searches."""
        # Add finance-related terms and use news category
        enhanced_query = f"{query} stock market finance"
        return self.search(enhanced_query, categories=["news", "general"], **kwargs)
    
    def search_code(self, query: str, language: str = None, **kwargs) -> SearchResponse:
        """
        Search for code snippets across GitHub, Stack Overflow, etc.
        
        Args:
            query: Code-related search query
            language: Optional programming language filter (e.g., "python", "javascript")
        """
        # Use IT category and code-focused engines
        code_engines = ["github", "stackoverflow", "gitlab", "bitbucket", "codeberg"]
        
        # Enhance query with language if provided
        if language:
            enhanced_query = f"{query} {language} code example"
        else:
            enhanced_query = f"{query} code example"
        
        return self.search(
            enhanced_query, 
            categories=["it"], 
            engines=code_engines,
            **kwargs
        )
    
    def search_repos(self, query: str, language: str = None, **kwargs) -> SearchResponse:
        """
        Search for GitHub/GitLab repositories.
        
        Args:
            query: Repository search query (e.g., "trading bot", "langgraph agent")
            language: Optional programming language filter
        """
        repo_engines = ["github", "gitlab", "codeberg", "bitbucket"]
        
        if language:
            enhanced_query = f"{query} {language} repository github"
        else:
            enhanced_query = f"{query} repository github"
        
        return self.search(
            enhanced_query,
            categories=["it"],
            engines=repo_engines,
            **kwargs
        )
    
    def search_docs(self, query: str, framework: str = None, **kwargs) -> SearchResponse:
        """
        Search for documentation and tutorials.
        
        Args:
            query: Documentation search query
            framework: Optional framework filter (e.g., "langchain", "fastapi")
        """
        if framework:
            enhanced_query = f"{framework} {query} documentation tutorial"
        else:
            enhanced_query = f"{query} documentation tutorial guide"
        
        return self.search(enhanced_query, categories=["general", "it"], **kwargs)
    
    def search_stackoverflow(self, query: str, tags: List[str] = None, **kwargs) -> SearchResponse:
        """
        Search Stack Overflow specifically.
        
        Args:
            query: Question/problem description
            tags: Optional tags like ["python", "pandas", "async"]
        """
        if tags:
            tag_str = " ".join(f"[{t}]" for t in tags)
            enhanced_query = f"site:stackoverflow.com {tag_str} {query}"
        else:
            enhanced_query = f"site:stackoverflow.com {query}"
        
        return self.search(enhanced_query, categories=["general"], **kwargs)
    
    def search_github(self, query: str, search_type: str = "repositories", **kwargs) -> SearchResponse:
        """
        Search GitHub specifically.
        
        Args:
            query: Search query
            search_type: "repositories", "code", "issues", or "discussions"
        """
        type_hints = {
            "repositories": "repository repo",
            "code": "code snippet",
            "issues": "issue bug fix",
            "discussions": "discussion help"
        }
        hint = type_hints.get(search_type, "")
        enhanced_query = f"site:github.com {query} {hint}"
        
        return self.search(enhanced_query, categories=["general", "it"], **kwargs)
    
    def search_pypi(self, query: str, **kwargs) -> SearchResponse:
        """Search for Python packages on PyPI."""
        enhanced_query = f"site:pypi.org {query} python package"
        return self.search(enhanced_query, categories=["general", "it"], **kwargs)
    
    def search_npm(self, query: str, **kwargs) -> SearchResponse:
        """Search for npm packages."""
        enhanced_query = f"site:npmjs.com {query} npm package"
        return self.search(enhanced_query, categories=["general", "it"], **kwargs)
    
    def health_check(self) -> bool:
        """Check if SearXNG instance is reachable."""
        try:
            response = requests.get(
                f"{self.host}/healthz",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            # Try a simple search as fallback health check
            try:
                response = requests.get(
                    f"{self.host}/search",
                    params={"q": "test", "format": "json"},
                    timeout=5
                )
                return response.status_code == 200
            except Exception:
                return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_client: Optional[SearXNGClient] = None


def get_client() -> SearXNGClient:
    """Get or create the default SearXNG client."""
    global _default_client
    if _default_client is None:
        _default_client = SearXNGClient()
    return _default_client


def web_search(query: str, max_results: int = 5) -> str:
    """
    Quick web search returning text suitable for LLM.
    
    Args:
        query: Search query
        max_results: Max results to include
        
    Returns:
        Formatted text with search results
    """
    client = get_client()
    response = client.search(query)
    return response.to_text(max_results=max_results)


def news_search(query: str, max_results: int = 5) -> str:
    """Quick news search returning text suitable for LLM."""
    client = get_client()
    response = client.search_news(query)
    return response.to_text(max_results=max_results)


def code_search(query: str, language: str = None, max_results: int = 5) -> str:
    """Quick code snippet search returning text suitable for LLM."""
    client = get_client()
    response = client.search_code(query, language=language)
    return response.to_text(max_results=max_results)


def repo_search(query: str, language: str = None, max_results: int = 5) -> str:
    """Quick repository search returning text suitable for LLM."""
    client = get_client()
    response = client.search_repos(query, language=language)
    return response.to_text(max_results=max_results)


def docs_search(query: str, framework: str = None, max_results: int = 5) -> str:
    """Quick documentation search returning text suitable for LLM."""
    client = get_client()
    response = client.search_docs(query, framework=framework)
    return response.to_text(max_results=max_results)


def stackoverflow_search(query: str, tags: List[str] = None, max_results: int = 5) -> str:
    """Quick Stack Overflow search returning text suitable for LLM."""
    client = get_client()
    response = client.search_stackoverflow(query, tags=tags)
    return response.to_text(max_results=max_results)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SearXNG Search Client")
    parser.add_argument("query", nargs="?", default="test query", help="Search query")
    parser.add_argument("--host", default=None, help="SearXNG host URL")
    parser.add_argument("--news", action="store_true", help="Search news only")
    parser.add_argument("--code", action="store_true", help="Search for code")
    parser.add_argument("--repos", action="store_true", help="Search for repositories")
    parser.add_argument("--docs", action="store_true", help="Search documentation")
    parser.add_argument("--stackoverflow", action="store_true", help="Search Stack Overflow")
    parser.add_argument("--lang", default=None, help="Programming language filter")
    parser.add_argument("--check", action="store_true", help="Health check only")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    client = SearXNGClient(host=args.host)
    
    if args.check:
        healthy = client.health_check()
        print(f"SearXNG status: {'[OK] Healthy' if healthy else '[FAIL] Unreachable'}")
        exit(0 if healthy else 1)
    
    if args.code:
        response = client.search_code(args.query, language=args.lang)
    elif args.repos:
        response = client.search_repos(args.query, language=args.lang)
    elif args.docs:
        response = client.search_docs(args.query, framework=args.lang)
    elif args.stackoverflow:
        response = client.search_stackoverflow(args.query, tags=[args.lang] if args.lang else None)
    elif args.news:
        response = client.search_news(args.query)
    else:
        response = client.search(args.query)
    
    print(response.to_text())

