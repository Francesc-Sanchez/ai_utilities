from langchain_core.tools import Tool

from src.tools.registry.external_tools.validators import wrap_with_limiter

from src.tools.registry.external_tools.Arxiv import get_arxiv_entry_id , get_arxiv_chain_results , \
    get_arxiv_wrapper , load_arxiv_docs
from src.tools.registry.external_tools.external_tool_functions import ddg_news_search , \
    wikipedia_structured_search , ddg_general_search , create_serpapi_detailed_tool , youtube_video_search_serpapi , \
    stackoverflow_detailed_search , bing_detailed_search_serpapi , github_domain_search

from src.tools.registry.shared_registry import agent_tools

from src.tools.registry.external_tools.pubmed import search_pubmed

search_tool_functions = {}


# Note: 'agent_tools' and 'search_tool_functions' are passed
# assuming the rest of the functions (ddg_search, etc.) are accessible.

def register_search_tools(agent_tools_catalogue: dict , search_func_catalogue: dict):
    """
    Registers all search tools in the provided agent and function catalogues,
    applying a link limiter.

    FIX: Stores the callable function (.func) in search_func_catalogue
    to resolve the "'Tool' object is not callable" error.
    """
    tools = [ ]

    # 1. DuckDuckGo News Tool
    ddg_news_tool = Tool (
        name = "DDGNewsSearch" ,
        func = wrap_with_limiter ( ddg_news_search ) ,
        description = "Searches the web for the latest news on a specific topic and returns a list of the 3 most relevant results, including their titles, descriptions, and links."
        )
    tools.append ( ddg_news_tool )
    search_func_catalogue[ "search_news" ] = ddg_news_tool.func  # FIX APPLIED

    # 2. DuckDuckGo General Tool
    ddg_general_tool = Tool (
        name = "DDGGeneralSearch" ,
        func = wrap_with_limiter ( ddg_general_search ) ,
        description = "Searches for general content on the web, excluding certain domains like Wikipedia."
        )
    tools.append ( ddg_general_tool )
    search_func_catalogue[ "search_web" ] = ddg_general_tool.func  # FIX APPLIED

    # 3. Wikipedia Structured Tool
    wikipedia_tool = Tool (
        name = "WikipediaStructuredSearch" ,
        func = wrap_with_limiter ( wikipedia_structured_search ) ,
        description = "Accesses encyclopedic information from Wikipedia. Returns a list of structured results with the page title, a summary, and a direct link."
        )
    tools.append ( wikipedia_tool )
    search_func_catalogue[ "search_wikipedia" ] = wikipedia_tool.func  # FIX APPLIED

    # 4. Google Tool (via SerpAPI)
    google_tool = Tool (
        name = "search_google_detailed" ,
        func = wrap_with_limiter ( create_serpapi_detailed_tool ) ,
        description = "Searches for current information on Google and returns the 5 most relevant results with title, description, and link."
        )
    tools.append ( google_tool )
    search_func_catalogue[ "search_google" ] = google_tool.func  # FIX APPLIED

    # 5. YouTube Tool (via SerpAPI)
    youtube_serpapi_tool = Tool (
        name = "YouTubeSerpAPISearch" ,
        func = wrap_with_limiter ( youtube_video_search_serpapi ) ,
        description = "Searches for relevant videos on YouTube using the Google API (via SerpAPI). Returns titles, descriptions, and direct links to videos."
        )
    tools.append ( youtube_serpapi_tool )
    search_func_catalogue[ "search_youtube" ] = youtube_serpapi_tool.func  # FIX APPLIED

    # 7. ArXiv Chain Tool
    arxiv_chain_tool = Tool.from_function (
        name = "ArxivResearchChain" ,
        func = wrap_with_limiter ( get_arxiv_chain_results ) ,
        description = "Generates an extensive response based on ArXiv documents regarding a scientific or technical question."
        )
    tools.append ( arxiv_chain_tool )
    search_func_catalogue[ "arxiv_chain_research" ] = arxiv_chain_tool.func  # FIX APPLIED

    # 8. ArXiv ID Tool
    arxiv_id_tool = Tool.from_function (
        name = "ArxivEntryID" ,
        func = get_arxiv_entry_id ,
        description = "Retrieves the entry ID of an ArXiv article based on a query."
        )
    tools.append ( arxiv_id_tool )
    search_func_catalogue[ "arxiv_id_research" ] = arxiv_id_tool.func  # FIX APPLIED

    # 9. ArXiv Wrapper Tool
    arxiv_wrapper_tool = Tool.from_function (
        name = "ArxivRawQuery" ,
        func = wrap_with_limiter ( get_arxiv_wrapper ) ,
        description = "Executes a direct query to the ArXiv API and returns the raw results."
        )
    tools.append ( arxiv_wrapper_tool )
    search_func_catalogue[ "arxiv_wrapper_research" ] = arxiv_wrapper_tool.func  # FIX APPLIED

    # 10. ArXiv Loader Tool
    arxiv_loader_tool = Tool.from_function (
        name = "ArxivDocumentLoader" ,
        func = wrap_with_limiter ( load_arxiv_docs ) ,
        description = "Loads up to 3 complete ArXiv documents based on a query."
        )
    tools.append ( arxiv_loader_tool )
    search_func_catalogue[ "arxiv_loader_research" ] = arxiv_loader_tool.func  # FIX APPLIED

    # 13. PubMed Search Tool
    pubmed_tool = Tool.from_function (
        name = "PubMedSearchTool" ,
        func = wrap_with_limiter ( search_pubmed ) ,
        description = "Searches for scientific articles on PubMed related to genetics, medicine, biology, or related topics. Returns titles, dates, and direct links to the most relevant articles."
        )
    tools.append ( pubmed_tool )
    search_func_catalogue[ "search_pubmed" ] = pubmed_tool.func  # FIX APPLIED

    # --- 🚨 14. BING Search Tool (NEW) ---
    bing_tool = Tool (
        name = "BingSearchTool" ,
        func = wrap_with_limiter ( bing_detailed_search_serpapi ) ,
        description = "Searches for general information on Bing (using the SerpAPI with the Bing engine) and returns results with title, description, and link. Useful as an alternative to Google."
        )
    tools.append ( bing_tool )
    search_func_catalogue[ "search_bing" ] = bing_tool.func  # FIX APPLIED

    # --- 🚨 15. Stack Overflow Search Tool (NEW) ---
    stackoverflow_tool = Tool.from_function (
        name = "StackOverflowSearchTool" ,
        func = wrap_with_limiter ( stackoverflow_detailed_search ) ,
        description = "Searches for relevant technical questions and answers on Stack Overflow (via Stack Exchange API) related to programming and software development."
        )
    tools.append ( stackoverflow_tool )
    search_func_catalogue[ "search_stackoverflow" ] = stackoverflow_tool.func  # FIX APPLIED

    # --- 🚨 16. GitHub Search Tool (NEW) ---
    github_tool = Tool (
        name = "GithubDomainSearch" ,
        func = wrap_with_limiter ( github_domain_search ) ,
        description = "Searches for code, repositories, and development documentation on GitHub using a free search restricted to the domain (site:github.com)."
        )

    # Add to the tools list
    tools.append ( github_tool )

    # Add to the function/mapping catalogue
    search_func_catalogue[ "search_github" ] = github_tool.func  # FIX APPLIED

    # --- KEY REGISTRATION IN AGENT CATALOGUE ---
    agent_tools_catalogue[ "search_agent" ] = tools
    search_func_catalogue[ "search_agent" ] = tools
    # 🚨 NOTE: assuming agent_tools is a global variable where you store the final catalogue
    agent_tools[ "search_agent" ] = tools

    print ( "✅ Search tools successfully registered under key 'search_agent' (16 tools total)." )
    return tools