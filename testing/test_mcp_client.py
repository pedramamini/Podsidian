import requests
import json
import sys
from typing import Dict, List, Optional

class PodsidianMCPClient:
    """client for testing podsidian mcp service."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """initialize client with base url.

        args:
            base_url: base url of mcp service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """make http request to mcp service.

        args:
            method: http method (get, post, etc)
            endpoint: api endpoint path
            **kwargs: additional request parameters

        returns:
            response data as dict
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"request failed: {e}")
            return {}

    def get_capabilities(self) -> Dict:
        """get server capabilities and available endpoints.

        returns:
            server capabilities dict
        """
        return self._make_request("GET", "/")

    def semantic_search(self, query: str, limit: int = 10, relevance: int = 25) -> List[Dict]:
        """search transcripts using natural language.

        args:
            query: search query string
            limit: maximum results to return
            relevance: minimum relevance score (0-100)

        returns:
            list of search results
        """
        params = {
            "query": query,
            "limit": limit,
            "relevance": relevance
        }
        return self._make_request("GET", "/api/v1/search/semantic", params=params)

    def keyword_search(self, keyword: str, limit: int = 10) -> List[Dict]:
        """search transcripts for exact keyword matches.

        args:
            keyword: text to search for
            limit: maximum results to return

        returns:
            list of search results
        """
        params = {
            "keyword": keyword,
            "limit": limit
        }
        return self._make_request("GET", "/api/v1/search/keyword", params=params)

    def list_episodes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """list all processed episodes.

        args:
            limit: maximum episodes to return
            offset: number of episodes to skip

        returns:
            list of episode info
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        return self._make_request("GET", "/api/v1/episodes", params=params)

    def get_episode(self, episode_id: int) -> Dict:
        """get specific episode details and transcript.

        args:
            episode_id: id of episode to retrieve

        returns:
            episode details dict
        """
        return self._make_request("GET", f"/api/v1/episodes/{episode_id}")

    def list_subscriptions(self) -> List[Dict]:
        """list all podcast subscriptions.

        returns:
            list of subscription info
        """
        return self._make_request("GET", "/api/v1/subscriptions")

    def mute_subscription(self, title: str) -> Dict:
        """mute a podcast subscription.

        args:
            title: title of podcast to mute

        returns:
            updated subscription info
        """
        return self._make_request("POST", f"/api/v1/subscriptions/{title}/mute")

    def unmute_subscription(self, title: str) -> Dict:
        """unmute a podcast subscription.

        args:
            title: title of podcast to unmute

        returns:
            updated subscription info
        """
        return self._make_request("POST", f"/api/v1/subscriptions/{title}/unmute")


def test_capabilities(client: PodsidianMCPClient):
    """test server capabilities endpoint."""
    print("testing server capabilities...")

    capabilities = client.get_capabilities()
    if capabilities:
        print(f"server name: {capabilities.get('name', 'unknown')}")
        print(f"version: {capabilities.get('version', 'unknown')}")
        print("available capabilities:")
        for cap_name, cap_info in capabilities.get('capabilities', {}).items():
            print(f"  {cap_name}:")
            for endpoint_name, endpoint_info in cap_info.items():
                print(f"    {endpoint_name}: {endpoint_info.get('endpoint', 'unknown')}")
    else:
        print("failed to get capabilities")
    print()


def test_episodes(client: PodsidianMCPClient):
    """test episode listing and retrieval."""
    print("testing episode endpoints...")

    # list episodes
    episodes = client.list_episodes(limit=5)
    if episodes:
        print(f"found {len(episodes)} episodes (showing first 5):")
        for episode in episodes:
            print(f"  #{episode.get('id', 'unknown')}: {episode.get('title', 'unknown')}")
            print(f"    podcast: {episode.get('podcast', 'unknown')}")
            print(f"    has transcript: {episode.get('has_transcript', False)}")

        # get details for first episode
        if episodes:
            first_episode_id = episodes[0].get('id')
            if first_episode_id:
                print(f"\ngetting details for episode #{first_episode_id}...")
                episode_details = client.get_episode(first_episode_id)
                if episode_details:
                    print(f"title: {episode_details.get('title', 'unknown')}")
                    transcript = episode_details.get('transcript', '')
                    if transcript:
                        print(f"transcript length: {len(transcript)} characters")
                        print(f"transcript preview: {transcript[:200]}...")
                    else:
                        print("no transcript available")
    else:
        print("no episodes found")
    print()


def test_subscriptions(client: PodsidianMCPClient):
    """test subscription management."""
    print("testing subscription endpoints...")

    subscriptions = client.list_subscriptions()
    if subscriptions:
        print(f"found {len(subscriptions)} subscriptions:")
        for sub in subscriptions:
            muted_status = "muted" if sub.get('muted', False) else "active"
            print(f"  {sub.get('title', 'unknown')} ({muted_status})")
            print(f"    author: {sub.get('author', 'unknown')}")
    else:
        print("no subscriptions found")
    print()


def test_search(client: PodsidianMCPClient):
    """test search functionality."""
    print("testing search endpoints...")

    # test semantic search
    print("testing semantic search...")
    search_queries = [
        "artificial intelligence",
        "climate change",
        "technology trends",
        "health and wellness"
    ]

    for query in search_queries:
        print(f"searching for: '{query}'")
        results = client.semantic_search(query, limit=3, relevance=30)
        if results:
            print(f"  found {len(results)} results:")
            for result in results:
                print(f"    {result.get('podcast', 'unknown')}: {result.get('episode', 'unknown')}")
                print(f"    relevance: {result.get('similarity', 0)}%")
                snippet = result.get('snippet', '')
                if snippet:
                    print(f"    snippet: {snippet[:100]}...")
                else:
                    print("    no snippet available")
        else:
            print("  no results found")
        print()

    # test keyword search
    print("testing keyword search...")
    keywords = ["python", "machine learning", "podcast", "interview"]

    for keyword in keywords:
        print(f"searching for keyword: '{keyword}'")
        results = client.keyword_search(keyword, limit=2)
        if results:
            print(f"  found {len(results)} results:")
            for result in results:
                print(f"    {result.get('podcast', 'unknown')}: {result.get('episode', 'unknown')}")
        else:
            print("  no results found")
        print()


def main():
    """main test function."""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8080"

    print(f"testing podsidian mcp service at {base_url}")
    print("=" * 50)

    client = PodsidianMCPClient(base_url)

    # test all endpoints
    test_capabilities(client)
    test_episodes(client)
    test_subscriptions(client)
    test_search(client)

    print("testing complete!")


if __name__ == "__main__":
    main()
