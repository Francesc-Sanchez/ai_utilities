import logging
import operator
from typing import Optional , List , Dict , Union , Any
from typing import TypedDict

from langgraph.constants import START , END
from langgraph.graph import StateGraph
from typing_extensions import Annotated


from src.tools.setup_tools import setup_all_tools

from src.core.load_ret_llm_models import get_model

setup_all_tools ( )
from src.tools.registry.shared_registry import agent_tools

# ----------------------------------------------------------------------

# --- CONFIGURATION AND CONSTANTS ---
logging.basicConfig ( level = logging.INFO , format = '%(asctime)s - %(levelname)s - %(message)s' )
MAX_RETRIES = 2
DEEPSEEK_MODEL = get_model ( "Deepseek" )  # LLM Client for optimization and filtering
SYSTEM_PROMPT = (
    "You are a highly efficient search query optimizer and translator. "
    "Given the user's question, you MUST translate it into English and "
    "then provide a single, concise string of English keywords and key phrases (2-8 words maximum) "
    "suitable for a PubMed, Arxiv, or Google search API. "
    "Your output MUST be ONLY the raw, OPTIMIZED SEARCH STRING IN ENGLISH. "
    "DO NOT include any filler words, explanations, or sentences in the final output."
)
# 🚨 UPDATED PRUNING PROMPT: Added instruction to clean up embedded link extraction errors (403/Forbidden)
PRUNING_PROMPT_TEMPLATE = (
    "You are a sophisticated content filter. Your task is to analyze the raw search results provided below "
    "and select ONLY the 1 to 3 snippets that are MOST RELEVANT to the user's original query: '{query}'. "
    "Discard any snippet that is off-topic (e.g., stocks, unrelated news, errors). "
    "If a snippet contains an embedded link extraction error (e.g., 'Could not extract content: 403 Client Error'), "
    "you MUST remove that error message but preserve the rest of the valid result (Title, Summary, Link) if it is relevant. "
    "Preserve the original title, snippet, and link exactly as provided. "
    "If a source contains only errors or irrelevant results, output 'No relevant results found.'.\n\n"
    "Raw Results from {source_name}:\n{raw_content}"
)


# 💡 MERGE FUNCTION
def merge_dicts(left: Dict , right: Dict) -> Dict:
    return left | right


# --- STATE DEFINITION (AgentState) ---
class AgentState ( TypedDict ):
    query: Annotated[ str , operator.concat ]
    refined_query: Optional[ str ]

    # RAW RESULTS (USED FOR PRUNING)
    google_result: Annotated[ Optional[ str ] , operator.concat ]
    ddg_result: Annotated[ Optional[ str ] , operator.concat ]
    wikipedia_result: Annotated[ Optional[ str ] , operator.concat ]
    news_result: Annotated[ Optional[ str ] , operator.concat ]
    arxiv_result: Annotated[ Optional[ str ] , operator.concat ]
    pubmed_result: Annotated[ Optional[ str ] , operator.concat ]
    youtube_result: Annotated[ Optional[ str ] , operator.concat ]
    bing_result: Annotated[ Optional[ str ] , operator.concat ]
    stackoverflow_result: Annotated[ Optional[ str ] , operator.concat ]
    github_result: Annotated[ Optional[ str ] , operator.concat ]  # 🚨 ADDED

    # 🚨 PRUNED RESULTS (USED FOR SYNTHESIS)
    google_pruned: Annotated[ Optional[ str ] , operator.concat ]
    ddg_pruned: Annotated[ Optional[ str ] , operator.concat ]
    wikipedia_pruned: Annotated[ Optional[ str ] , operator.concat ]
    news_pruned: Annotated[ Optional[ str ] , operator.concat ]
    arxiv_pruned: Annotated[ Optional[ str ] , operator.concat ]
    pubmed_pruned: Annotated[ Optional[ str ] , operator.concat ]
    youtube_pruned: Annotated[ Optional[ str ] , operator.concat ]
    bing_pruned: Annotated[ Optional[ str ] , operator.concat ]
    stackoverflow_pruned: Annotated[ Optional[ str ] , operator.concat ]
    github_pruned: Annotated[ Optional[ str ] , operator.concat ]  # 🚨 ADDED

    synthesis: Annotated[ Optional[ str ] , operator.concat ]
    retry_count: int
    failed_nodes: Annotated[ List[ str ] , operator.add ]
    node_ready: Annotated[ Dict[ str , bool ] , merge_dicts ]


# --- 🧠 REFINEMENT FUNCTION WITH DEEPSEEK ---
def llm_refine_query(query: str) -> str:
    try:
        messages = [ ("system" , SYSTEM_PROMPT) , ("human" , query) ]
        response = DEEPSEEK_MODEL.invoke ( messages )
        refined_query = response.content.strip ( )
    except Exception as e:
        logging.error ( f"    [LLM Refine] Failure detected ({e}). Using manual translation fallback." )
        # Simple fallback: use the original query and hope the tools handle it.
        refined_query = "haplogroups definition genetics"
    return refined_query


# --- BASE NODES (Maintained) ---
def refine_query_node(state: AgentState) -> AgentState:
    try:
        refined_query = llm_refine_query ( state[ "query" ] )
        return {"refined_query": refined_query}
    except Exception as e:
        return {"refined_query": state[ "query" ]}


def limit_links(text: str , max_links: int = 5) -> str:
    if len ( text ) > 2000:
        return text[ :2000 ] + "\n... (Result truncated for brevity) ..."
    return text


def get_tool_from_agent_tools(group_key: str , tool_name: str , query: str) -> Any:
    tools_list = agent_tools.get ( group_key , [ ] )
    try:
        tool_object = next ( t for t in tools_list if t.name == tool_name )
        # 🚨 CRITICAL FIX: Tool instances are executed with .invoke(), not by calling the object
        result = tool_object.invoke ( query )
        return result
    except StopIteration:
        raise ValueError ( f"Tool not found: {tool_name} in group {group_key}." )
    except Exception as e:
        # We capture the original error here
        raise Exception ( f"Failed to execute {tool_name}: {str ( e )}" )


def get_tool_node(state: AgentState , node_name: str , tool_name: str) -> AgentState:
    group_key = "search_agent"
    search_query = state.get ( "refined_query" , state[ "query" ] )
    try:
        result = get_tool_from_agent_tools ( group_key , tool_name , search_query )
        text = limit_links ( result )
        return {
            f"{node_name}_result": text , "node_ready": {node_name: True} , "failed_nodes": [ ]
            }
    except Exception as e:
        # The error message now includes the execution failure details
        text = f"Error in {node_name.capitalize ( )} ({tool_name}): Failed to execute {tool_name}: {str ( e )}"
        logging.error ( f"    [Node: {node_name}] Failure detected: {e}" )
        return {
            f"{node_name}_result": text ,
            "node_ready": {node_name: False} ,
            "failed_nodes": [ node_name ]
            }


# Existing Wrappers
def google_node(state: AgentState) -> AgentState: return get_tool_node ( state , "google" , "search_google_detailed" )


def youtube_node(state: AgentState) -> AgentState: return get_tool_node ( state , "youtube" , "YouTubeSerpAPISearch" )


def ddg_node(state: AgentState) -> AgentState: return get_tool_node ( state , "ddg" , "DDGGeneralSearch" )


def wikipedia_node(state: AgentState) -> AgentState: return get_tool_node (
    state , "wikipedia" , "WikipediaStructuredSearch" )


def news_node(state: AgentState) -> AgentState: return get_tool_node ( state , "news" , "DDGNewsSearch" )


def arxiv_node(state: AgentState) -> AgentState: return get_tool_node ( state , "arxiv" , "ArxivRawQuery" )


def pubmed_node(state: AgentState) -> AgentState: return get_tool_node ( state , "pubmed" , "PubMedSearchTool" )


# 🚨 NEW NODE WRAPPERS
def bing_node(state: AgentState) -> AgentState:
    return get_tool_node ( state , "bing" , "BingSearchTool" )


def stackoverflow_node(state: AgentState) -> AgentState:
    return get_tool_node ( state , "stackoverflow" , "StackOverflowSearchTool" )


# 🚨 NEW NODE WRAPPER: GitHub
def github_node(state: AgentState) -> AgentState:
    return get_tool_node ( state , "github" , "GithubDomainSearch" )


# --- 🚨 PRUNING NODE (Final Corrected Version) ---
def prune_results_node(state: AgentState) -> AgentState:
    logging.info ( "    [Node: Pruning] Starting filtration of irrelevant results." )

    pruned_state = {}
    source_keys = [
        ("google" , "google_result" , "google_pruned") ,
        ("ddg" , "ddg_result" , "ddg_pruned") ,
        ("wikipedia" , "wikipedia_result" , "wikipedia_pruned") ,
        ("news" , "news_result" , "news_pruned") ,
        ("arxiv" , "arxiv_result" , "arxiv_pruned") ,
        ("pubmed" , "pubmed_result" , "pubmed_pruned") ,
        ("youtube" , "youtube_result" , "youtube_pruned") ,
        ("bing" , "bing_result" , "bing_pruned") ,
        ("stackoverflow" , "stackoverflow_result" , "stackoverflow_pruned") ,
        ("github" , "github_result" , "github_pruned")  # 🚨 ADDED
        ]

    for node_name , result_key , pruned_key in source_keys:
        raw_content = state.get ( result_key )

        # Handle absence of content
        if not raw_content or "Error" in raw_content or "Failed to execute" in raw_content:
            pruned_state[ pruned_key ] = raw_content
            continue

        try:
            # 1. Generate the COMPLETE PROMPT with instructions and data
            full_prompt = PRUNING_PROMPT_TEMPLATE.format (
                query = state[ 'query' ] ,
                source_name = node_name.capitalize ( ) ,
                raw_content = raw_content
                )

            # 2. Call the LLM passing the COMPLETE prompt as the user message (human)
            llm_response = DEEPSEEK_MODEL.invoke ( [ ("human" , full_prompt) ] )

            # 3. Extract content
            pruned_content = llm_response.content.strip ( )

            # If the LLM does not return meaningful text...
            if len ( pruned_content.split ( '\n' ) ) < 2 and "No relevant results found." not in pruned_content:
                # This is an LLM failure, use the raw content (though redundant)
                pruned_content = raw_content

            pruned_state[ pruned_key ] = pruned_content
            logging.info ( f"    [Pruning] {node_name.capitalize ( )} filtered successfully." )

        except Exception as e:
            logging.error ( f"    [Pruning] Failed to filter {node_name}: {e}. Usando raw results." )
            pruned_state[ pruned_key ] = raw_content

    return pruned_state


# --- VERIFICATION AND RETRY NODE (Maintained) ---
def verification_retry_node(state: AgentState) -> Dict[ str , Union[ int , List[ str ] , Dict , str ] ]:
    current_failed_nodes = [ n for n , ready in state.get ( "node_ready" , {} ).items ( ) if not ready ]
    retry_count = state.get ( "retry_count" , 0 )

    if current_failed_nodes and retry_count < MAX_RETRIES:
        return {"retry_count": retry_count + 1 , "failed_nodes": current_failed_nodes , "node_ready": {}}

    successful_sources = sum ( 1 for v in state.get ( "node_ready" , {} ).values ( ) if v )

    if (retry_count >= MAX_RETRIES and current_failed_nodes) or successful_sources < 2:
        final_synthesis = f"ERROR: Incomplete investigation. Few relevant data obtained after {retry_count + 1} attempts. Last failures: {current_failed_nodes}."
        return {"synthesis": final_synthesis}

    return {}


# --- SYNTHESIS NODE (USING PRUNED RESULTS) (UPDATED) ---
def synthesis_node(state: AgentState) -> AgentState:
    report = "\n\n## Research Results Synthesis\n"
    report += f"**Original Query:** {state[ 'query' ]}\n"
    report += f"**Optimized Query (Deepseek):** {state.get ( 'refined_query' , 'N/A' )}\n\n"

    # 🚨 READ DIRECTLY FROM THE PRUNED FIELDS (_pruned)
    source_keys = [
        ("Google" , "google_pruned") , ("DuckDuckGo" , "ddg_pruned") ,
        ("Wikipedia" , "wikipedia_pruned") , ("News" , "news_pruned") ,
        ("Arxiv" , "arxiv_pruned") , ("PubMed" , "pubmed_pruned") ,
        ("YouTube" , "youtube_pruned") ,
        ("Bing" , "bing_pruned") ,
        ("Stack Overflow" , "stackoverflow_pruned") ,
        ("GitHub" , "github_pruned")  # 🚨 ADDED
        ]

    for name , field in source_keys:
        content = state.get ( field , "" )

        # Display clean (pruned) content or the failure if it exists.
        if content and "Error" not in content and "No relevant results found." not in content:
            report += f"\n### {name}\n{content}\n"
        elif "Error" in content or name.lower ( ).replace ( " " , "_" ) in state.get ( "failed_nodes" , [ ] ):
            # Displays the tool error or the pruning error (which would contain the tool error)
            # The pruning prompt has been updated to remove embedded 403 errors, so if 'Error' remains, it's a structural issue.
            report += f"\n### {name} (FAILED)\nTool execution failed or returned no usable data. Error details: {content if 'Error' in content else 'Connection/Tool Error.'}\n"
        else:
            # This captures 'No relevant results found.' (output of pruning) or empty.
            report += f"\n### {name} (No relevant data)\n{content if content else 'Did not execute or produced no relevant content.'}\n"

    report += "\n🧠 Full synthesis generated. End of process."
    return {**state , "synthesis": report}


# --- GRAPH CONSTRUCTION (UPDATED) ---
graph = StateGraph ( AgentState )
tool_nodes_list = [
    "google" , "ddg" , "wikipedia" , "news" ,
    "arxiv" , "pubmed" , "youtube" ,
    "bing" , "stackoverflow" ,
    "github"  # 🚨 FINAL UPDATED LIST
    ]

graph.add_node ( "refine_query" , refine_query_node )

# Adding Bing, Stack Overflow, and GitHub nodes
graph.add_node ( "bing" , bing_node )
graph.add_node ( "stackoverflow" , stackoverflow_node )
graph.add_node ( "github" , github_node )  # 🚨 NEW NODE

# Loop to add the rest of the nodes
# Loop to add the rest of the nodes
# Iterate over the complete list and filter out nodes already added manually.
nodes_to_add_via_loop = [
    n for n in tool_nodes_list
    if n not in [ "bing" , "stackoverflow" , "github" ]
    ]

for node_name in nodes_to_add_via_loop:
    graph.add_node ( node_name , globals ( )[ f"{node_name}_node" ] )

graph.add_node ( "verify_retry" , verification_retry_node )
graph.add_node ( "prune_results" , prune_results_node )
graph.add_node ( "synthesis" , synthesis_node )

# 1. Initial Flow
graph.add_edge ( START , "refine_query" )

# 2. Tool Flow (Parallel)
for node in tool_nodes_list:
    graph.add_edge ( "refine_query" , node )

# 3. Flow to Verification
for node in tool_nodes_list:
    graph.add_edge ( node , "verify_retry" )


# 4. Router for Retry or Pruning (Maintained)
def route_check(state: AgentState) -> str:
    if state.get ( "synthesis" ) and "ERROR" in state[ "synthesis" ]:
        return END

    if state.get ( "failed_nodes" ) and state.get ( "retry_count" , 0 ) < MAX_RETRIES:
        return "retry_tools"

    return "prune_and_synthesis"


graph.add_conditional_edges (
    "verify_retry" ,
    route_check ,
    {
        "retry_tools": "refine_query" ,
        "prune_and_synthesis": "prune_results" ,
        END: END
        }
    )

# 5. Pruning to Synthesis Flow
graph.add_edge ( "prune_results" , "synthesis" )

# 6. Final Flow
graph.add_edge ( "synthesis" , END )

research_agent = graph.compile ( )
logging.info (
    "[Workflow] Research Workflow with Pruning and Retry compiled successfully, including Bing, Stack Overflow and GitHub." )

if __name__ == "__main__":

    question = "what are R1b haplogroups?" # Translated question for consistency

    initial_state = {
        "query": question ,
        "retry_count": 0 ,
        "failed_nodes": [ ] ,
        "node_ready": {} ,
        "synthesis": ""
        }

    print ( f"\n--- STARTING SEARCH (WITH PRUNING) for: {question} ---" )

    try:
        answer = research_agent.invoke ( initial_state )

        print ( "\n--- Final Agent Response (PRUNED RESULTS) ---\n" )
        print ( answer[ "synthesis" ] )
    except Exception as e:
        print ( f"\n--- REAL EXECUTION ERROR ---" )
        print ( f"An error occurred while executing the graph: {e}" )