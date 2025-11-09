import warnings
from typing import List, Dict, Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from serpapi import GoogleSearch

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="langchain_community.utilities.duckduckgo_search"
)
import os
from dotenv import load_dotenv

load_dotenv()
serpapi = os.getenv("SERPAPI_API_KEY")


def github_domain_search(query: str, max_results: int = 5) -> str:
    """
    Performs a specific, free web search within the github.com domain
    using DuckDuckGo to find repositories, code, or documentation.
    """

    # 1. Modify the query to restrict the search to the GitHub domain
    github_query = f"{query} site:github.com"

    try:
        with DDGS() as ddgs:
            # We use the text method (general search) of DDGS
            results = ddgs.text(
                github_query,
                region='en-us',  # Use 'en-us' or 'es-es' based on preference
                safesearch='moderate',
                max_results=max_results
            )

            if not results:
                return f"No results found on GitHub (site:github.com) for: {query}"

            output_lines = []
            for item in results:
                link = item.get('href', '')
                domain = urlparse(link).netloc

                # Although the search already restricts, we ensure the domain is GitHub
                if 'github.com' not in domain:
                    continue

                title = item.get('title', 'No Title')
                snippet = item.get('body', 'No description')

                output_lines.append(f"Title: {title}")
                output_lines.append(f"Description: {snippet}")
                output_lines.append(f"Link: {link}")
                output_lines.append(f"Source: {domain}")
                output_lines.append("-" * 20)

            if not output_lines:
                return "No relevant results found on GitHub after filtering."

            return "\n".join(output_lines)

    except Exception as e:
        return f"Error while searching GitHub (DDGS): {e}"


# --- DuckDuckGo Tool ---
def ddg_news_search(query: str, max_results: int = 3) -> str:
    """Searches for news using DuckDuckGo and returns results with titles, descriptions, and links."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query=query, region='en-us', safesearch='off', timelimit='m')

            if not results:
                return "No recent news found for this query."

            output_lines = []
            count = 0
            for item in results:
                if count >= max_results:
                    break
                output_lines.append(f"Title: {item.get('title')}")
                output_lines.append(f"Description: {item.get('body')}")
                output_lines.append(f"Link: {item.get('url')}")
                output_lines.append(f"Source: {item.get('source')}")
                output_lines.append("-" * 20)
                count += 1

            return "\n".join(output_lines)
    except Exception as e:
        return f"Error while searching for news: {e}"


ddg_news_tool = Tool(
    name="DDGNewsSearch",
    func=ddg_news_search,
    description="""Searches for the latest news on the web about a specific topic and
    returns a list of the 3 most relevant results, including their titles,
    descriptions, and links. Ideal for queries about recent events or updates."""
)


def extract_page_text(url: str, max_chars: int = 1000) -> str:
    """
    Downloads and extracts text from a webpage, returning a more complete summary.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract paragraphs with content
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if
                       len(p.get_text(strip=True)) > 40]

        # Join several paragraphs for more context
        text = " ".join(paragraphs[:5])  # up to 5 paragraphs
        text = " ".join(text.split())  # clean up spaces

        return text[:max_chars] + "..." if len(text) > max_chars else text

    except Exception as e:
        return f"Could not extract content: {e}"


def ddg_general_search(query: str, max_results: int = 3, exclude_links: list[str] = None) -> str:
    """
    Searches DuckDuckGo and returns results with an extended summary,
    excluding Wikipedia and other specified links.
    """
    wikipedia_domain = "wikipedia.org"
    exclude_links = exclude_links or []
    exclude_domains = {urlparse(link).netloc for link in exclude_links}
    exclude_domains.add(wikipedia_domain)  # exclude all Wikipedia

    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region='es-es',
                safesearch='moderate',
                max_results=max_results
            )

            if not results:
                return "No results found for this query."

            output_lines = []
            for item in results:
                link = item.get('href', '')
                domain = urlparse(link).netloc

                # Filtering: exclude if the domain is in the list
                if domain in exclude_domains:
                    continue

                title = item.get('title', 'No Title')
                snippet = item.get('body', 'No description')

                # Extended summary by visiting the page
                extended_summary = extract_page_text(link)

                output_lines.append(f"Title: {title}")
                output_lines.append(f"DDG Summary: {snippet}")
                output_lines.append(f"Extended Summary: {extended_summary}")
                output_lines.append(f"Link: {link}")
                output_lines.append(f"Source: {domain}")
                output_lines.append("-" * 20)

            if not output_lines:
                return "No relevant results found after filtering."

            return "\n".join(output_lines)

    except Exception as e:
        return f"Error while searching content: {e}"


ddg_general_tool = Tool(
    name="DDGGeneralSearch",
    func=ddg_general_search,
    description="Searches general content on the web..."
)


def create_serpapi_detailed_tool(query: str) -> str:
    wrapper = SerpAPIWrapper(serpapi_api_key=serpapi)
    raw = wrapper.results(query)

    hits = raw.get("results") or raw.get("organic_results") or []
    if not hits:
        return f"No results were found for the search: {query}" # Changed from Catalan to English

    formatted_results = []
    for i, result in enumerate(hits[:25]):
        title = result.get("title", "No title")
        link = result.get("link", "No link")
        snippet = result.get("snippet", "No description")
        formatted_results.append(
            f"{i + 1}. **{title}**\n{snippet}\n🔗 {link}\n"
        )

    return "\n".join(formatted_results)


def stackoverflow_detailed_search(query: str, max_results: int = 5) -> str:
    """
    Searches for questions and answers on Stack Overflow using the Stack Exchange API,
    using the 'q' field for a broader and more effective search.
    """
    url = "https://api.stackexchange.com/2.3/search/advanced"  # 🚨 Use advanced search for 'q'

    # We try to isolate the programming language if present, otherwise, we use a broad tag.
    # For the example 'Information about code in python', the 'python' tag is key.

    # 🚨 KEY CHANGE: Use 'q' for the broad query
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,  # Use 'q' to search title and body, more flexible
        "site": "stackoverflow",
        "pagesize": max_results,
        "tagged": "python"  # 💡 Tip: Add a relevant tag to focus the search
    }

    formatted_results = []

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])

        if not items:
            return f"No relevant questions found on Stack Overflow for: {query}"

        for i, item in enumerate(items):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            answer_count = item.get("answer_count", 0)
            is_answered = item.get("is_answered", False)

            status = "✅ ANSWERED" if is_answered else "❌ UNANSWERED"

            formatted_results.append(
                f"{i + 1}. **{title}**\n Answers: {answer_count} ({status})\n🔗 {link}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error while searching Stack Overflow: {e}"


def youtube_video_search_serpapi(query: str, max_results: int = 5) -> str:
    """
    Searches for relevant videos on YouTube using the Google/SerpAPI API.
    Returns up to 5 results with metadata and link.
    """

    # The query no longer needs 'site:youtube.com' because "tbm": "vid" handles video filtering.
    # However, keeping it might improve relevance.
    youtube_query = f"{query} site:youtube.com"

    # Initialize the list of formatted results before the try/except block
    formatted_results: List[str] = []

    try:
        # 1. Execute the search with explicit control of results (num) and type (tbm: "vid")
        search = GoogleSearch(
            {
                "api_key": serpapi,
                "q": query,  # We use the original query if we trust tbm: "vid"
                "engine": "google",
                "num": max_results,  # <-- Forces the API to return 'max_results' (max 100)
                "tbm": "vid"  # <-- Specifically filters by video
            })

        raw: Dict[str, Any] = search.get_dict()

        # 2. 🚨 CORRECTION: Extract video results
        # SerpAPI places relevant results in 'video_results' when using tbm: "vid"
        hits: List[Dict[str, Any]] = raw.get("video_results") or raw.get("organic_results") or []

        if not hits:
            return f"No YouTube videos were found for the search: {query}" # Changed from Catalan to English

        # 3. Format the output for the LLM (limiting to max_results, although the API already did this)
        # We enumerate over all 'hits', which are already limited to 'max_results' by the API.
        for i, result in enumerate(hits):
            title = result.get("title", result.get("name", "No title"))
            link = result.get("link", result.get("url", "No link"))
            snippet = result.get("snippet", result.get("description", "No description"))

            # We filter links that are not clearly from YouTube, although 'tbm': 'vid' is quite reliable.
            if 'youtube.com' not in link and 'youtu.be' not in link:
                continue

            formatted_results.append(
                f"{i + 1}. **VIDEO: {title}**\n{snippet}\n📺 Link: {link}\n"
            )

        if not formatted_results:
            return f"No valid YouTube results were found for the search: {query}" # Changed from Catalan to English

        return "\n".join(formatted_results)

    except Exception as e:
        # Here you can add a logging.error(e) for debugging
        return f"Error while searching YouTube videos (SerpAPI): {e}"


# --- NEW CORRECTED WIKIPEDIA TOOL ---
def wikipedia_structured_search(query: str, max_results: int = 5) -> str:
    """
    Searches Wikipedia and returns results with titles, summaries, and links.
    """
    api_wrapper = WikipediaAPIWrapper()
    try:
        # CORRECTION: We use the Wikipedia client to perform the search
        page_titles = api_wrapper.wiki_client.search(query, results=max_results)

        output_lines = []
        for title in page_titles:
            try:
                # CORRECTION: We access the summary through the page object
                page_obj = api_wrapper.wiki_client.page(title, auto_suggest=False)
                page_summary = page_obj.summary

                # Build the link to the Wikipedia page
                encoded_title = quote(title.replace(" ", "_"))
                page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

                output_lines.append(f"Title: {title}")
                output_lines.append(f"Summary: {page_summary}")
                output_lines.append(f"Link: {page_url}")
                output_lines.append("-" * 20)
            except Exception as page_e:
                # If a specific page throws an error, we log it and continue
                output_lines.append(f"Error getting data for '{title}': {page_e}")
                output_lines.append("-" * 20)

        if not output_lines:
            return "No results found on Wikipedia for this query."

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error while searching Wikipedia: {e}"


wikipedia_tool = Tool(
    name="WikipediaStructuredSearch",
    func=wikipedia_structured_search,
    description="""Accesses encyclopedic information from Wikipedia. Returns a list
    of structured results with the page title, a summary, and a direct link."""
)


def bing_detailed_search_serpapi(query: str) -> str:
    """
    Searches Bing web results using the SerpAPI key.
    """
    formatted_results = []
    try:
        search = GoogleSearch(
            {
                "api_key": serpapi,
                "q": query,
                "engine": "bing",  # 🚨 KEY CHANGE: Use the 'bing' engine
                "num": 5
            })

        raw: Dict[str, Any] = search.get_dict()
        hits = raw.get("organic_results") or []

        if not hits:
            return f"No results found on Bing (SerpAPI) for: {query}"

        for i, result in enumerate(hits):
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No description")

            formatted_results.append(
                f"{i + 1}. **{title}**\n{snippet}\n🔗 {link}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error executing Bing (SerpAPI): {e}"


def run_tool_tests():
    """Runs a series of searches to test the tools."""
    print("--- Testing DuckDuckGo News Search Tool ---")

    duckduckgo_query = "latest Vatican news in English"
    print(f"DuckDuckGo (News) Query: '{duckduckgo_query}'")

    try:
        duckduckgo_result = ddg_news_tool.run(duckduckgo_query)
        print("DuckDuckGo (News) Result:")
        print(duckduckgo_result)
    except Exception as e:
        print(f"Error executing DuckDuckGo (News): {e}")

    print("\n" + "-" * 50 + "\n")

    print("--- Testing Wikipedia Structured Search Tool ---")

    wikipedia_query = "History of the Vatican"
    print(f"Wikipedia Query: '{wikipedia_query}'")

    try:
        wikipedia_result = wikipedia_tool.run(wikipedia_query)
        print("Wikipedia (Structured) Result:")
        print(wikipedia_result)
    except Exception as e:
        print(f"Error executing Wikipedia: {e}")

    print("\n" + "-" * 50 + "\n")

    print("--- Testing DuckDuckGo General Search Tool ---")

    general_query = "Vatican history and architecture"
    print(f"DuckDuckGo (General) Query: '{general_query}'")

    try:
        general_result = ddg_general_tool.invoke(general_query)
        print("DuckDuckGo (General) Result:")
        print(general_result)
    except Exception as e:
        print(f"Error executing DuckDuckGo (General): {e}")

#
# if __name__ == "__main__":
#     run_tool_tests()
# if __name__ == "__main__":
#     query = "Information about code in python"
#     output = bing_detailed_search_serpapi(query)
#     print(output)