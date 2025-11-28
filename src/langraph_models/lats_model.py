
import json
import logging
import math
import re
import sys
from collections import deque
from datetime import datetime
from typing import Optional , Annotated , cast
import importlib
import traceback
import os
import html
import re
try:
    import requests
except Exception:
    requests = None

# Allow skipping runtime side-effects (LLM calls) at import time. When set to '1',
# the module will avoid invoking LLM chains and other heavy operations so tooling
# (like graph rendering) can still run.
SKIP_SIDE_EFFECTS = os.environ.get("LATS_NO_SIDE_EFFECTS", "0") == "1"

from langchain_core.messages import BaseMessage , HumanMessage , AIMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel , Field

# Attempt to import the project's internal research retriever module and validate
# that it exposes the expected API. Provide a safe alias for AgentState when
# the real type isn't present so other code can still cast without crashing.
AgentState = dict  # fallback typing alias if real AgentState isn't present
research_ret_mod = None
research_tool_available = False
try:
    research_ret_mod = importlib.import_module("src.langraph_models.research_ret")
    # If the module exports AgentState, use that for typing clarity
    if hasattr(research_ret_mod, "AgentState"):
        AgentState = getattr(research_ret_mod, "AgentState")
    # Validate presence of the function we need
    if not hasattr(research_ret_mod, "get_tool_node"):
        raise AttributeError("src.langraph_models.research_ret does not expose 'get_tool_node'")
    research_tool_available = True
    logging.info("Using internal research retriever (research_ret).")
except Exception as e:
    research_ret_mod = None
    research_tool_available = False
    logging.warning("Internal research retriever not available: %s. Tooling disabled.", e)
    logging.debug("research_ret import traceback:\n%s", traceback.format_exc())

# typing_extensions.TypedDict was previously used; TreeState is now a Pydantic BaseModel so it's unused.
try:
    from typing_extensions import TypedDict
except Exception:
    TypedDict = None

from src.core.load_ret_llm_models import get_model


# AgentState TypedDict not required here; we pass a minimal dict to research_ret_mod
# (import removed to avoid unused-import warnings)


class Reflection ( BaseModel ):
    reflections: str = Field (
        description = "The critique and reflections on the sufficiency, superfluency,"
                      " and general quality of the response"
        )
    score: Annotated[ int , Field (
        description = "Score from 0-10 on the quality of the candidate response." ,
        ge = 0 ,
        le = 10 ,
        ) ]
    found_solution: bool = Field (
        description = "Whether the response has fully solved the question or task."
        )
    # Optional provenance: structured list of source descriptors (title/url/snippet)
    sources: list[ dict ] = Field (
        default_factory = list ,
        description = "List of provenance objects {title,url,snippet} consulted for this reflection"
        )

    def as_message(self):
        return HumanMessage (
            content = f"Reasoning: {self.reflections}\nScore: {self.score}"
            )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    # Explicit attribute to help static analysis tools
    depth: int = 0

    def __init__(
            self ,
            messages: list[ BaseMessage ] ,
            reflection: Reflection ,
            parent: Optional[ "Node" ] = None ,
            metadata: Optional[ dict ] = None ,
            ):
        self.messages = messages
        self.parent = parent
        self.children = [ ]
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        # free-form metadata to attach provenance, tool call summaries, etc.
        self.metadata = metadata or {}
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved ( )
        self.backpropagate ( reflection.normalized_score )

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max ( self.children , key = lambda child: int ( child.is_solved ) * child.value )

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max ( [ child.height for child in self.children ] )
        return 1

    def upper_confidence_bound(self , exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError ( "Cannot obtain UCT from root node" )
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt ( math.log ( self.parent.visits ) / self.visits )
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self , reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self , include_reflections: bool = True):
        if include_reflections:
            return self.messages + [ self.reflection.as_message ( ) ]
        return self.messages

    def get_trajectory(self , include_reflections: bool = True) -> list[ BaseMessage ]:
        """Get messages representing this search branch."""
        messages = [ ]
        node = self
        while node:
            messages.extend (
                node.get_messages ( include_reflections = include_reflections )[ ::-1 ]
                )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[ ::-1 ]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = [ ]
        nodes = deque ( )
        nodes.append ( self )
        while nodes:
            node = nodes.popleft ( )
            all_nodes.extend ( node.children )
            for n in node.children:
                nodes.append ( n )
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [ self ] + self._get_all_children ( )
        best_node = max (
            all_nodes ,
            # We filter out all non-terminal, non-solution trajectories
            key = lambda node: int ( node.is_terminal and node.is_solved ) * node.value ,
            )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


class TreeState ( BaseModel ):
    # The full tree (optional during intermediate steps)
    root: Optional[ Node ] = None
    # The original input
    input: str

    # Allow arbitrary Python types (like Node) in Pydantic v2 models
    model_config = {"arbitrary_types_allowed": True}


llm = get_model ( "gpt-4o" )

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser ,
    PydanticToolsParser ,
    )
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable , RunnableConfig

# Compute project root (two levels up from this file: src/langraph_models -> src -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ensure logs and outputs are written under the repository root for consistency
DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'output')

# Force system prompts to ask the LLM to reply in English regardless of input language
prompt = ChatPromptTemplate.from_messages (
    [
        (
            "system" ,
            "Reflect and grade the assistant response to the user question below. Please answer in English regardless of the user's language.",
            ) ,
        ("user" , "{input}") ,
        MessagesPlaceholder ( variable_name = "candidate" ) ,
        ]
    )

prompt_template = ChatPromptTemplate.from_messages (
    [
        (
            "system" ,
            "You are an AI assistant. Please answer in English regardless of the user's language.",
            ) ,
        ("user" , "{input}") ,
        MessagesPlaceholder ( variable_name = "messages" , optional = True ) ,
        ]
    )

reflection_llm_chain = (
        prompt
        | llm.bind_tools ( tools = [ Reflection ] , tool_choice = "Reflection" ).with_config (
    run_name = "Reflection"
    )
        | PydanticToolsParser ( tools = [ Reflection ] )
)

# Force DEBUG logging configuration early (force=True overrides prior framework handlers such as uvicorn/streamlit)
# This makes the module emit DEBUG logs reliably during development.
logging.basicConfig (
    level = logging.INFO , format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s' , force = True
    )
root_logger = logging.getLogger ( )
root_logger.setLevel ( logging.INFO )

# Module logger
logger = logging.getLogger ( __name__ )
logger.setLevel ( logging.INFO )

# Ensure a stdout handler exists and is set to INFO
have_stdout_handler = any (
    isinstance ( h , logging.StreamHandler ) and getattr ( h , 'stream' , None ) is sys.stdout for h in logger.handlers
    )
if not have_stdout_handler:
    sh = logging.StreamHandler ( sys.stdout )
    sh.setLevel ( logging.INFO )
    sh.setFormatter ( logging.Formatter ( '%(asctime)s - %(levelname)s - %(name)s - %(message)s' ) )
    logger.addHandler ( sh )

# Add (or ensure) a file handler for persistent logs
try:
    # (os already imported at module level)
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    logfile = os.path.join(DEFAULT_LOG_DIR, 'lats_model.log')
    have_file_handler = False
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == logfile:
            have_file_handler = True
            break
    if not have_file_handler:
        fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(fh)
except Exception:
    logger.exception('Could not create file handler for logs; continuing without file logging')

# Emit an initial info message to verify INFO mode is active
logger.info ( 'lats_model.py logging initialized in INFO mode' )
print ( 'lats_model.py: INFO mode active — printing direct to stdout' )
sys.stdout.flush ( )

# Keep propagation off to avoid duplicate messages in environments that already configure root handlers
logger.propagate = False


@as_runnable
def reflection_chain(inputs) -> Reflection:
    logger.debug (
        "reflection_chain invoked; keys: %s" ,
        list ( inputs.keys ( ) ) if isinstance ( inputs , dict ) else type ( inputs ) )
    try:
        tool_choices = reflection_llm_chain.invoke ( inputs )
        reflection = tool_choices[ 0 ]
        logger.info (
            "reflection_chain produced reflection with score=%s found_solution=%s" ,
            getattr ( reflection , 'score' , None ) , getattr ( reflection , 'found_solution' , None ) )
        if not isinstance ( inputs[ "candidate" ][ -1 ] , AIMessage ):
            reflection.found_solution = False
        return reflection
    except Exception as e:
        logger.exception ( "reflection_chain failed: %s" , e )
        raise


# Use the project's internal research retriever instead of external Tavily packages.
# (Duplicate import removed: detection and import already handled above.)
# No external ToolNode here; we'll call research_ret functions directly and
# convert their dict outputs into AIMessage objects that downstream code can consume.
tool_node = None
tools = [ ]


def call_research_tool(tool_call: dict , query: str) -> dict:
    logger.info (
        "call_research_tool: received tool_call type=%s args_keys=%s query_snippet=%s" , tool_call.get ( 'type' ) ,
        list ( tool_call.get ( 'args' , {} ).keys ( ) ) if isinstance (
            tool_call.get ( 'args' ) , dict ) else None ,
        (query[ :80 ] + '...') if query and len ( query ) > 80 else query )
    if not research_tool_available:
        logger.warning ( "research retriever not available when calling tool: %s" , tool_call.get ( 'type' ) )
        return {"error": "research retriever not available"}

    ttype = (tool_call.get ( "type" ) or "").lower ( )
    # Basic mapping heuristics
    if "google" in ttype:
        node_name , tool_name = "google" , "search_google_detailed"
    elif "ddg" in ttype or "duck" in ttype:
        node_name , tool_name = "ddg" , "DDGGeneralSearch"
    elif "wikipedia" in ttype:
        node_name , tool_name = "wikipedia" , "WikipediaStructuredSearch"
    elif "youtube" in ttype:
        node_name , tool_name = "youtube" , "YouTubeSerpAPISearch"
    elif "arxiv" in ttype or "arxivrawquery" in ttype:
        node_name , tool_name = "arxiv" , "ArxivRawQuery"
    else:
        # default to google search
        node_name , tool_name = "google" , "search_google_detailed"

    # Prepare a minimal AgentState-like dict for the research retriever and cast for typing
    agent_state = cast ( AgentState , {"query": query} )
    try:
        res = research_ret_mod.get_tool_node ( agent_state , node_name , tool_name )
        # Log size/summary of response
        if isinstance ( res , dict ):
            content_lens = {k: (len ( v ) if isinstance ( v , str ) else None) for k , v in res.items ( )}
            logger.info ( "call_research_tool: result keys=%s content_lens=%s" , list ( res.keys ( ) ) , content_lens )
            # build structured provenance entries
            sources: list[ dict ] = [ ]

            def add_source(title: Optional[ str ] , url: Optional[ str ] , snippet: Optional[ str ] = None):
                if not title and not url:
                    return
                entry = {}
                if title:
                    entry[ 'title' ] = title
                if url:
                    entry[ 'url' ] = url
                if snippet:
                    entry[ 'snippet' ] = snippet
                sources.append ( entry )

            # common structures: 'items', 'results', 'entries', 'links', 'rows'
            for k in ("items" , "results" , "entries" , "links" , "rows"):
                if k in res and isinstance ( res[ k ] , (list , tuple) ):
                    for itm in res[ k ]:
                        if isinstance ( itm , dict ):
                            url = itm.get ( 'url' ) or itm.get ( 'link' ) or itm.get ( 'source' )
                            title = itm.get ( 'title' ) or itm.get ( 'name' ) or itm.get ( 'headline' )
                            snippet = itm.get ( 'snippet' ) or itm.get ( 'summary' ) or itm.get ( 'description' )
                            add_source ( title , url , snippet )

            # fallback to direct keys
            title = res.get ( 'title' ) or res.get ( 'name' )
            url = res.get ( 'url' ) or res.get ( 'link' )
            if title or url:
                add_source ( title , url , res.get ( 'snippet' ) or res.get ( 'description' ) )

            # attempt to sniff urls in string-valued fields and include as url-only entries
            # Also extract URLs embedded in HTML/text (e.g. google_result) using regex and include snippets.
            url_pattern = re.compile ( r"https?://[\w\-./?=&%#:]+" , re.IGNORECASE )
            for k , v in res.items ( ):
                if isinstance ( v , str ):
                    # quick check for full-url fields
                    if v.startswith ( 'http://' ) or v.startswith ( 'https://' ):
                        add_source ( None , v , None )
                        continue
                    # search for embedded urls
                    found = url_pattern.findall ( v )
                    if found:
                        # include up to first 5 urls with a short snippet from the text
                        for u in found[ :5 ]:
                            snippet = None
                            try:
                                idx = v.find ( u )
                                start = max ( 0 , idx - 80 )
                                end = min ( len ( v ) , idx + len ( u ) + 80 )
                                snippet = v[ start:end ].replace ( '\n' , ' ' )
                            except Exception:
                                snippet = v[ :200 ]
                            add_source ( None , u , snippet )

            # attach structured provenance
            res_with_sources = dict ( res )
            res_with_sources.setdefault ( '__provenance__' , {} )
            res_with_sources[ '__provenance__' ][ 'sources' ] = sources
            logger.info (
                "call_research_tool: extracted %d structured sources from node=%s tool=%s" , len ( sources ) ,
                node_name , tool_name )
            return res_with_sources
        else:
            logger.info ( "call_research_tool: non-dict result type=%s" , type ( res ) )
            return {"content": str ( res ) , "__provenance__": {"sources": [ ]}}
    except Exception as e:
        logger.exception ( "research retriever call failed: %s" , e )
        return {"error": str ( e ) , "__provenance__": {"sources": [ ]}}


prompt_template = ChatPromptTemplate.from_messages (
    [
        (
            "system" ,
            "You are an AI assistant." ,
            ) ,
        ("user" , "{input}") ,
        MessagesPlaceholder ( variable_name = "messages" , optional = True ) ,
        ]
    )

initial_answer_chain = prompt_template | llm.bind_tools ( tools = tools ).with_config (
    run_name = "GenerateInitialCandidate"
    )

parser = JsonOutputToolsParser ( return_id = True )

# initial top-level LLM invocation — guarded to allow import-time dry-run for tooling
if not SKIP_SIDE_EFFECTS:
    initial_response = initial_answer_chain.invoke (
        {"input": "Write a research report on lithium pollution."}
        )
else:
    initial_response = None


# print ( initial_response )


# Helper: fetch default sources — must be defined before generate_initial_response
def fetch_default_search_sources(query: str , max_nodes: int = 3) -> list[ dict ]:
    """When the parser returns no tool calls, perform a few default searches
    (google, wikipedia, ddg) to gather structured provenance entries.
    Returns deduplicated list of provenance dicts {title?, url?, snippet?}.
    """
    if not research_tool_available:
        logger.debug ( "fetch_default_search_sources: research retriever unavailable" )
        return [ ]
    nodes_and_tools = [
        ("google" , "search_google_detailed") ,
        ("wikipedia" , "WikipediaStructuredSearch") ,
        ("ddg" , "DDGGeneralSearch") ,
        ]
    gathered = [ ]
    try:
        for i , (node , tool) in enumerate ( nodes_and_tools ):
            if i >= max_nodes:
                break
            try:
                fake_call = {"type": node , "args": {}}
                raw = call_research_tool ( fake_call , query )
                if isinstance ( raw , dict ):
                    sources = raw.get ( '__provenance__' , {} ).get ( 'sources' , [ ] ) or [ ]
                else:
                    sources = [ ]
                if sources:
                    gathered.extend ( sources )
                    logger.info (
                        "fetch_default_search_sources: gathered %d sources from %s/%s" , len ( sources ) , node , tool )
            except Exception:
                logger.exception ( "fetch_default_search_sources: failed node=%s tool=%s" , node , tool )
    except Exception:
        logger.exception ( "fetch_default_search_sources: unexpected failure" )
    # dedupe by url/title
    seen = set ( )
    dedup = [ ]
    for s in gathered:
        key = (s.get ( 'url' ) , s.get ( 'title' )) if isinstance ( s , dict ) else (None , str ( s ))
        if key in seen:
            continue
        seen.add ( key )
        dedup.append ( s )
    return dedup


# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Accept either a TreeState (Pydantic) or a dict; normalize to dict for processing.

    This avoids KeyError when the graph runtime passes a Pydantic model instance.
    """
    # Normalize state to a plain dict regardless of input type
    try:
        if hasattr ( state , "model_dump" ):
            state_dict = state.model_dump ( )  # pydantic v2 preferred
        elif isinstance ( state , dict ):
            state_dict = state
        else:
            # Fallback: try to coerce to dict via vars()
            state_dict = dict ( vars ( state ) )
    except Exception:
        logger.exception ( "generate_initial_response: failed to normalize state to dict" )
        raise

    input_snippet = (state_dict.get ( "input" , "" )[ :120 ] + "...") if state_dict.get ( "input" ) else "<no input>"
    logger.info ( "generate_initial_response: start; input_snippet=%s" , input_snippet )

    # Validate incoming normalized state
    if "input" not in state_dict:
        logger.error ( "generate_initial_response: invalid state, missing 'input': %s" , repr ( state ) )
        raise KeyError ( f"generate_initial_response requires 'input' in state; received: {repr ( state )}" )

    # Invoke chain and parse safely
    try:
        res = initial_answer_chain.invoke ( {"input": state_dict[ "input" ]} )
        logger.debug ( "initial_answer_chain returned type=%s" , type ( res ) )
    except Exception as e:
        logger.exception ( "initial_answer_chain failed: %s" , e )
        res = AIMessage ( content = "" )

    try:
        parsed = parser.invoke ( res )
        logger.info ( "parser.invoke produced %d parsed tool calls" , len ( parsed ) if parsed is not None else 0 )
    except Exception:
        logger.warning ( "parser.invoke failed on initial response; assuming no tool calls." )
        parsed = [ ]

    tool_responses = [ ]
    all_sources = [ ]
    if research_tool_available and parsed:
        for r in parsed:
            logger.debug ( "Calling research tool for parsed item id=%s type=%s" , r.get ( 'id' ) , r.get ( 'type' ) )
            try:
                raw_result = call_research_tool ( r , state_dict.get ( 'input' , '' ) )
                if isinstance ( raw_result , dict ):
                    content = None
                    for k , v in raw_result.items ( ):
                        if k.endswith ( '_result' ) and isinstance ( v , str ):
                            content = v
                            break
                    if content is None:
                        # try 'content' or 'message' fields
                        content = raw_result.get ( 'content' ) or raw_result.get ( 'message' ) or str ( raw_result )
                    # pull structured provenance if available
                    prov = raw_result.get ( '__provenance__' , {} ).get ( 'sources' , [ ] ) or [ ]
                else:
                    content = str ( raw_result )
                    prov = [ ]
                tool_msg = AIMessage ( content = content )
                tool_responses.append ( tool_msg )
                if prov:
                    # prov is list of dicts; extend structured list
                    all_sources.extend ( prov )
                logger.info (
                    "Tool call produced content length=%d structured_sources=%d" , len ( content ) if content else 0 ,
                    len ( prov ) )
            except Exception:
                logger.exception ( "call_research_tool failed for initial response; skipping tool response." )
    else:
        logger.debug (
            "No parsed tool calls or research retriever unavailable. parsed_len=%d research_available=%s" ,
            len ( parsed ) if parsed is not None else 0 , research_tool_available )
        # If no parsed tool calls, proactively run a few default searches to gather provenance
        try:
            if research_tool_available and (not parsed or len ( parsed ) == 0):
                logger.info ( "No tool calls parsed; running default searches to gather provenance." )
                fetched = fetch_default_search_sources ( state_dict.get ( 'input' , '' ) )
                if fetched:
                    all_sources.extend ( fetched )
                    # add a short AIMessage summarizing top sources so they appear in output_messages
                    top = fetched[ :3 ]
                    summary_lines = [ ]
                    for it in top:
                        url = it.get ( 'url' ) if isinstance ( it , dict ) else None
                        title = it.get ( 'title' ) if isinstance ( it , dict ) else None
                        if title and url:
                            summary_lines.append ( f"{title} <{url}>" )
                        elif url:
                            summary_lines.append ( url )
                    if summary_lines:
                        tool_msg = AIMessage ( content = "Found sources: " + "; ".join ( summary_lines ) )
                        tool_responses.append ( tool_msg )
                        logger.info (
                            "Inserted AIMessage summarizing %d fetched sources into output_messages" ,
                            len ( summary_lines ) )
        except Exception:
            logger.exception ( "Failed to run default searches for provenance" )

    # Build output messages
    output_messages = [ res ]
    for tr in tool_responses:
        if isinstance ( tr , AIMessage ):
            output_messages.append ( tr )
        elif isinstance ( tr , dict ):
            try:
                output_messages.append ( tr.get ( 'messages' , [ ] )[ 0 ] )
            except Exception:
                logger.warning ( "Unexpected dict tool response format; skipping element." )
        else:
            try:
                output_messages.append ( AIMessage ( content = str ( tr ) ) )
            except Exception:
                logger.warning ( "Unexpected tool response type; skipping element." )

    logger.info ( "generate_initial_response: created %d output_messages" , len ( output_messages ) )

    # Try to get reflection; if it fails, create a default one
    try:
        reflection = reflection_chain.invoke ( {"input": state_dict[ "input" ] , "candidate": output_messages} )
        logger.info (
            "reflection_chain returned score=%s found_solution=%s" , getattr ( reflection , 'score' , None ) ,
            getattr ( reflection , 'found_solution' , None ) )
        # attach provenance sources discovered during initial tools to the reflection if present
        try:
            if all_sources and isinstance ( reflection , Reflection ):
                # deduplicate structured sources by url (prefer) then title
                seen = set ( )
                merged = [ ]
                for s in (reflection.sources or [ ]) + all_sources:
                    key = (s.get ( 'url' ) , s.get ( 'title' )) if isinstance ( s , dict ) else (None , str ( s ))
                    if key in seen:
                        continue
                    seen.add ( key )
                    merged.append ( s )
                reflection.sources = merged
                logger.info (
                    "generate_initial_response: attached %d structured sources to reflection" ,
                    len ( reflection.sources ) )
        except Exception:
            logger.exception ( "Failed attaching sources to reflection" )
        if reflection is None or not isinstance ( reflection , Reflection ):
            raise ValueError ( "Invalid reflection returned" )
    except Exception:
        logger.exception ( "reflection_chain failed; creating default Reflection." )
        reflection = Reflection (
            reflections = "No reflection available." , score = 0 , found_solution = False , sources = [ ] )

    metadata = {"tool_calls": parsed , "sources": all_sources}
    root = Node ( output_messages , reflection = reflection , metadata = metadata )
    logger.info (
        "generate_initial_response: root created with depth=%s children_count=%d" , root.depth , len ( root.children ) )
    # Return a plain dict state (the graph runtime expects mapping-like states)
    out_state = {**state_dict}
    out_state[ "root" ] = root
    # Add a structured provenance summary for easy programmatic consumption
    try:
        # dedupe aggregated structured sources
        seen = set ( )
        deduped = [ ]
        for s in all_sources:
            key = (s.get ( 'url' ) , s.get ( 'title' )) if isinstance ( s , dict ) else (None , str ( s ))
            if key in seen:
                continue
            seen.add ( key )
            deduped.append ( s )
        out_state[ 'provenance' ] = {
            'root_sources': deduped ,
            'reflection_sources': getattr ( reflection , 'sources' , [ ] ) if isinstance (
                reflection , Reflection ) else [ ] ,
            'tool_calls': parsed
            }
        logger.info ( 'generate_initial_response: added provenance summary to out_state' )
    except Exception:
        logger.exception ( 'Failed to attach provenance summary to out_state' )
    return out_state


# This generates N candidate values
# for a single input to sample actions from the environment


@as_runnable
def generate_candidates(messages: ChatPromptValue , config: RunnableConfig):
    n = config[ "configurable" ].get ( "N" , 5 )
    logger.info ( "generate_candidates: generating n=%d candidates" , n )
    bound_kwargs = llm.bind_tools ( tools = tools ).kwargs
    chat_result = llm.generate (
        [ messages.to_messages ( ) ] ,
        n = n ,
        callbacks = config[ "callbacks" ] ,
        run_name = "GenerateCandidates" ,
        **bound_kwargs ,
        )
    gens = [ gen.message for gen in chat_result.generations[ 0 ] ]
    logger.info ( "generate_candidates: returning %d candidates" , len ( gens ) )
    return gens


expansion_chain = prompt_template | generate_candidates  # type: ignore

# Guard expansion chain top-level call as well (it triggers the LLM)
if not SKIP_SIDE_EFFECTS:
    res = expansion_chain.invoke ( {"input": "Write a research report on lithium pollution."} )
else:
    res = None
from collections import defaultdict


def select(root: Node) -> Node:
    logger.debug ( "select: start at root with %d children" , len ( root.children ) if root else 0 )
    if not root.children:
        logger.debug ( "select: root has no children; returning root" )
        return root
    node = root
    while node.children:
        max_child = max ( node.children , key = lambda child: child.upper_confidence_bound ( ) )
        logger.debug (
            "select: descending to child with value=%s visits=%s depth=%s" , max_child.value , max_child.visits ,
            max_child.depth )
        node = max_child
    logger.debug ( "select: reached leaf node at depth=%s" , node.depth )
    return node


def normalize_state_to_dict(state) -> dict:
    """Convert TreeState or other model-like objects to a plain dict for processing."""
    if isinstance ( state , dict ):
        return state
    # pydantic v2
    if hasattr ( state , "model_dump" ):
        try:
            return state.model_dump ( )
        except Exception:
            pass
    # fallback to vars() for non-pydantic objects or models without model_dump
    # fallback to vars
    try:
        return dict ( vars ( state ) )
    except Exception:
        logger.debug ( "normalize_state_to_dict: failed to coerce state to dict; returning empty dict" )
        return {}


def expand(state: TreeState , config: RunnableConfig) -> dict:
    # Normalize incoming state to a dict so we can subscript and pass to tool chains safely
    state_dict = normalize_state_to_dict ( state )
    logger.info (
        "expand: invoked; input_snippet=%s" ,
        (state_dict.get ( 'input' , '' )[ :80 ] + '...') if state_dict.get ( 'input' ) else str ( state )
        )
    # Ensure root exists in normalized state
    if not isinstance ( state_dict , dict ) or "root" not in state_dict:
        logger.error ( "expand: invalid state; missing 'root'" )
        raise KeyError ( f"expand requires 'root' in state; received: {repr ( state )}" )
    if "input" not in state_dict:
        logger.error ( "expand: invalid state; missing 'input'" )
        raise KeyError ( f"expand requires 'input' in state; received: {repr ( state )}" )

    root = state_dict.get ( "root" )
    logger.info (
        "expand: root.height=%s root.children=%d" , getattr ( root , 'height' , None ) ,
        len ( root.children ) if hasattr ( root , 'children' ) else 0 )
    best_candidate: Node = select ( root )
    logger.debug (
        "expand: best_candidate value=%s visits=%s depth=%s" , best_candidate.value , best_candidate.visits ,
        best_candidate.depth )
    messages = best_candidate.get_trajectory ( )
    logger.debug ( "expand: best_candidate trajectory length=%d" , len ( messages ) )

    try:
        new_candidates = expansion_chain.invoke ( {"input": state_dict[ "input" ] , "messages": messages} , config )
        logger.info (
            "expand: expansion_chain returned %d new_candidates" ,
            len ( new_candidates ) if hasattr ( new_candidates , '__len__' ) else 1 )
    except Exception:
        logger.exception ( "expansion_chain.invoke failed; returning state unchanged." )
        return state_dict

    try:
        parsed = parser.batch ( new_candidates )
        logger.debug ( "expand: parser.batch returned parsed len=%d" , len ( parsed ) if parsed is not None else 0 )
    except Exception:
        logger.exception ( "parser.batch failed on expansion; assuming no tool calls for candidates." )
        parsed = [ [ ] for _ in new_candidates ]

    flattened = [
        (i , tool_call)
        for i , tool_calls in enumerate ( parsed )
        for tool_call in (tool_calls or [ ])
        ]
    logger.info ( "expand: flattened tool calls count=%d" , len ( flattened ) )

    tool_responses = [ ]
    collected_sources = defaultdict ( list )
    if research_tool_available and flattened:
        for i , tool_call in flattened:
            logger.debug (
                "expand: calling research tool for candidate_index=%d tool_type=%s" , i , tool_call.get ( 'type' ) )
            try:
                raw_result = call_research_tool ( tool_call , state_dict.get ( 'input' , '' ) )
                if isinstance ( raw_result , dict ):
                    content = None
                    for k , v in raw_result.items ( ):
                        if k.endswith ( '_result' ) and isinstance ( v , str ):
                            content = v
                            break
                    if content is None:
                        content = str ( raw_result )
                    prov = raw_result.get ( '__provenance__' , {} ).get ( 'sources' , [ ] ) or [ ]
                    if prov:
                        collected_sources[ i ].extend ( prov )
                else:
                    content = str ( raw_result )
                    prov = [ ]
                msg = AIMessage ( content = content )
                tool_responses.append ( (i , msg) )
            except Exception:
                logger.exception ( "call_research_tool failed for an expansion tool call; skipping it." )
    logger.info ( "expand: collected %d tool_responses" , len ( tool_responses ) )

    collected_responses = defaultdict ( list )
    for i , resp in tool_responses:
        collected_responses[ i ].append ( resp )

    output_messages = [ ]
    for i , candidate in enumerate ( new_candidates ):
        msgs = [ candidate ]
        msgs.extend ( collected_responses.get ( i , [ ] ) )
        output_messages.append ( msgs )
    logger.info ( "expand: prepared %d output_messages for reflection" , len ( output_messages ) )

    reflections = [ ]
    try:
        reflections = reflection_chain.batch (
            [ {"input": state_dict[ "input" ] , "candidate": msges} for msges in output_messages ] ,
            config ,
            )
        logger.info ( "expand: reflection_chain.batch returned %d reflections" , len ( reflections ) )
    except Exception:
        logger.exception ( "reflection_chain.batch failed; creating default reflections." )
        reflections = [ Reflection ( reflections = "No reflection." , score = 0 , found_solution = False ) for _ in
                        output_messages ]

    child_nodes = [ Node ( cand , parent = best_candidate , reflection = reflection ) for cand , reflection in
                    zip ( output_messages , reflections ) ]
    best_candidate.children.extend ( child_nodes )
    logger.info (
        "expand: added %d child_nodes to best_candidate (now children=%d)" , len ( child_nodes ) ,
        len ( best_candidate.children ) )

    # Attach provenance metadata to newly created child nodes when available
    for idx , node in enumerate ( child_nodes ):
        srcs = collected_sources.get ( idx ) or [ ]
        try:
            node.metadata = node.metadata or {}
            node.metadata.setdefault ( 'sources' , [ ] )
            # merge any structured sources produced during reflection as well
            refl = node.reflection
            if isinstance ( refl , Reflection ) and refl.sources:
                node.metadata[ 'sources' ].extend ( refl.sources )
            node.metadata[ 'sources' ].extend ( srcs )
            # deduplicate by url/title
            seen = set ( )
            dedup = [ ]
            for s in node.metadata[ 'sources' ]:
                key = (s.get ( 'url' ) , s.get ( 'title' )) if isinstance ( s , dict ) else (None , str ( s ))
                if key in seen:
                    continue
                seen.add ( key )
                dedup.append ( s )
            node.metadata[ 'sources' ] = dedup
            if node.metadata[ 'sources' ]:
                logger.info (
                    'expand: child %s attached %d provenance sources' , idx , len ( node.metadata[ 'sources' ] ) )
        except Exception:
            logger.exception ( 'Failed to attach provenance metadata to child node %s' , idx )

    # Return the mutated state as a plain dict for downstream graph operations
    state_dict[ "root" ] = root
    return state_dict


# Decision function used by the StateGraph builder to control looping
def should_loop(state):
    """Determine whether to continue expanding or end the graph run.

    Returns either the name of the next node to run ("expand") or the special END marker.
    Accepts a TreeState instance or a plain dict.
    """
    try:
        state_dict = state.model_dump() if hasattr(state, 'model_dump') else (state if isinstance(state, dict) else dict(vars(state)))
    except Exception:
        state_dict = state if isinstance(state, dict) else {}

    root = state_dict.get('root') if isinstance(state_dict, dict) else None
    if root is None:
        return "expand"
    try:
        # prefer root.is_solved if available
        is_solved = getattr(root, 'is_solved', False)
        height = getattr(root, 'height', None)
        children = getattr(root, 'children', None)
    except Exception:
        is_solved = False
        height = None
        children = None

    if not children:
        return "expand"
    if is_solved:
        return END
    if isinstance(height, int) and height > 5:
        return END
    return "expand"

# Optional graphing package import; provide safe fallbacks for environments that lack it.
try:
    from langgraph.graph import END, StateGraph, START
except Exception:
    # Provide minimal placeholders so tooling and static analysis won't break import-time.
    END = 'END'
    START = 'START'
    class StateGraph:
        def __init__(self, *args, **kwargs):
            raise ImportError("langgraph.graph is required to build/compile the runtime graph")

# Build the state graph and compile it
builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")

builder.add_conditional_edges(
    "start",
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    should_loop,
    ["expand", END],
)

graph = builder.compile()


def run_query(question: str):
    """Run the compiled state graph for a single question and save outputs.

    Returns a dict with paths to saved files.
    """
    steps = []
    last_expand_state = None
    # Build an initial TreeState
    try:
        ts = TreeState(input=question)
        initial_state = ts.model_dump()
    except Exception:
        initial_state = {"input": question}

    # Stream the graph (StateGraph API) and capture states
    for step in graph.stream(initial_state):
        steps.append(step)
        name = next(iter(step))
        state_val = list(step.values())[0]
        if name == 'expand':
            last_expand_state = state_val

    # fallback: find a state containing 'root'
    if last_expand_state is None:
        for s in reversed(steps):
            st = list(s.values())[0]
            if isinstance(st, dict) and 'root' in st:
                last_expand_state = st
                break

    if last_expand_state is None:
        raise RuntimeError('No expand/root state found in graph stream')

    root_obj = last_expand_state['root']
    if not isinstance(root_obj, Node):
        raise RuntimeError(f"Expected root Node, got {type(root_obj)}")

    solution_node = root_obj.get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)

    # Collect provenance
    try:
        prov = None
        if isinstance(last_expand_state, dict):
            prov = last_expand_state.get('provenance')
        if not prov:
            prov = {}
            prov['root_sources'] = getattr(root_obj, 'metadata', {}).get('sources', []) or []
            try:
                refl = getattr(solution_node, 'reflection', None)
                prov['reflection_sources'] = getattr(refl, 'sources', []) if refl is not None else []
            except Exception:
                prov['reflection_sources'] = []
            prov['tool_calls'] = []
    except Exception:
        prov = {'root_sources': [], 'reflection_sources': [], 'tool_calls': []}

    # Prepare output directory and write files
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    # schema: last message from best trajectory
    if best_trajectory:
        last_msg = best_trajectory[-1]
        content = getattr(last_msg, 'content', None)
        text_to_save = content if content is not None else str(last_msg)
    else:
        text_to_save = ''
    schema_txt = os.path.join(DEFAULT_OUTPUT_DIR, 'schema.txt')
    with open(schema_txt, 'w', encoding='utf-8') as f:
        f.write(text_to_save or '')
    logger.info('Saved final content to %s', schema_txt)

    # Build solution JSON with top-level provenance for backward compatibility
    sol_out = {
        'question': question,
        'final_info': {
            'root_depth': getattr(root_obj, 'depth', None),
            'children': len(getattr(root_obj, 'children', [])),
            'solution_depth': getattr(solution_node, 'depth', None),
            'solution_value': getattr(solution_node, 'value', None),
        },
        'root_sources': prov.get('root_sources', []) if isinstance(prov, dict) else [],
        'reflection_sources': prov.get('reflection_sources', []) if isinstance(prov, dict) else [],
        'tool_calls': prov.get('tool_calls', []) if isinstance(prov, dict) else [],
        'provenance': prov,
        'best_trajectory': [(getattr(m, 'content', None) or str(m)) for m in best_trajectory]
    }
    sol_json = os.path.join(DEFAULT_OUTPUT_DIR, 'solution.json')
    with open(sol_json, 'w', encoding='utf-8') as f:
        json.dump(sol_out, f, ensure_ascii=False, indent=2)
    logger.info('Saved detailed solution to %s', sol_json)

    bt_txt = os.path.join(DEFAULT_OUTPUT_DIR, 'best_trajectory.txt')
    with open(bt_txt, 'w', encoding='utf-8') as f:
        for m in best_trajectory:
            f.write((getattr(m, 'content', None) or str(m)) + "\n\n")
    logger.info('Saved best_trajectory to %s', bt_txt)

    # At the end of run_query, write the report automatically (if run_query completes successfully)
    # We patch the function by adding the call in-place earlier where sol_out is created
    report_path = write_report(sol_out)

    return {'schema': schema_txt, 'solution_path': sol_json, 'best_trajectory_path': bt_txt, 'report_path': report_path}


def _fetch_url_metadata(url: str, timeout: float = 5.0) -> dict:
    """Try to fetch URL and extract title and DOI (best-effort). Returns dict {title, doi, url}.
    Non-fatal — on any error returns an object with only 'url'."""
    meta = {"url": url, "title": None, "doi": None}
    doi_rx = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
    try:
        if requests is None:
            # fallback to urllib
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as r:
                content = r.read(32768).decode(errors='ignore')
        else:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            if r.status_code != 200:
                return meta
            content = r.text[:32768]
        # extract <title>
        m = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if m:
            title = html.unescape(m.group(1).strip())
            meta['title'] = re.sub(r"\s+", " ", title)
        # attempt to find meta name citation_doi
        m2 = re.search(r"<meta[^>]+name=['\"]citation_doi['\"][^>]*content=['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
        if m2:
            meta['doi'] = m2.group(1).strip()
            return meta
        # or meta property='citation_doi'
        m2 = re.search(r"<meta[^>]+property=['\"]citation_doi['\"][^>]*content=['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
        if m2:
            meta['doi'] = m2.group(1).strip()
            return meta
        # try to sniff DOI anywhere in the snippet
        d = doi_rx.search(content)
        if d:
            meta['doi'] = d.group(0)
    except Exception:
        # on any network/parse error return best-effort meta with url only
        return meta
    return meta


def write_report(sol_out: dict, out_dir: Optional[str] = None) -> str:
    """Write a Spanish Markdown report to DEFAULT_OUTPUT_DIR (or out_dir if given).
    Enriches root_sources by attempting to resolve titles/DOIs.
    Returns the path to the written report."""
    try:
        if out_dir is None:
            out_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, 'report.md')

        # Build content
        lines = []
        lines.append('# Informe generado por LATS')
        lines.append('')
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        lines.append(f'*Generado: {now}*')
        lines.append('')

        question = sol_out.get('question')
        if question:
            lines.append('## Pregunta')
            lines.append('')
            lines.append(question)
            lines.append('')

        fi = sol_out.get('final_info', {})
        lines.append('## Información final')
        lines.append('')
        for k in ('root_depth', 'children', 'solution_depth', 'solution_value'):
            lines.append(f'- **{k}**: {fi.get(k)}')
        lines.append('')

        # provenance
        prov = sol_out.get('provenance') or {}
        root_sources = prov.get('root_sources') if isinstance(prov, dict) else sol_out.get('root_sources', [])
        lines.append('## Fuentes principales (root_sources)')
        lines.append('')
        if root_sources:
            # enrich sources with metadata (title, doi) — do this sequentially to avoid parallel complexity
            enriched = []
            for s in root_sources:
                if not isinstance(s, dict):
                    entry = {'url': str(s)}
                else:
                    entry = dict(s)
                url = entry.get('url')
                # only attempt network resolution for HTTP(s) links
                if url and (url.startswith('http://') or url.startswith('https://')):
                    meta = _fetch_url_metadata(url)
                    if meta.get('title') and not entry.get('title'):
                        entry['title'] = meta['title']
                    if meta.get('doi'):
                        entry['doi'] = meta['doi']
                enriched.append(entry)

            # write as markdown numbered list
            for i, e in enumerate(enriched, 1):
                title = e.get('title') or e.get('url')
                url = e.get('url')
                doi = e.get('doi')
                snippet = e.get('snippet')
                if title and url:
                    lines.append(f'{i}. [{title}]({url})')
                elif url:
                    lines.append(f'{i}. {url}')
                if doi:
                    lines.append(f'   - DOI: `{doi}`')
                if snippet:
                    lines.append('')
                    lines.append(f'    > {snippet.strip()}')
                lines.append('')
        else:
            lines.append('_No se encontraron fuentes principales._')
            lines.append('')

        # best trajectory and schema
        bt = sol_out.get('best_trajectory') or []
        lines.append('## Mejor trayectoria (resumen)')
        lines.append('')
        if bt:
            # include first element summary and then code block with full trajectory
            lines.append(bt[0])
            lines.append('')
            lines.append('```text')
            for m in bt:
                lines.append(str(m))
                lines.append('')
            lines.append('```')
        else:
            lines.append('_No hay trayectoria disponible._')
        lines.append('')

        schema_txt_path = os.path.join(out_dir, 'schema.txt')
        if os.path.exists(schema_txt_path):
            lines.append('## Contenido final (schema.txt)')
            lines.append('')
            try:
                with open(schema_txt_path, 'r', encoding='utf-8') as f:
                    schema = f.read()
                lines.append('```text')
                lines.append(schema)
                lines.append('```')
            except Exception:
                lines.append('_No se pudo leer schema.txt_')
            lines.append('')

        lines.append('## Conclusiones')
        lines.append('')
        lines.append('Resumen y conclusiones basadas en las fuentes y la trayectoria.')
        lines.append('')
        lines.append('---')
        lines.append('*Informe generado automáticamente por LATS.*')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return report_path
    except Exception:
        logger.exception('write_report failed')
        raise


def write_report_from_files(out_dir: Optional[str] = None) -> str:
    """Read DEFAULT_OUTPUT_DIR/solution.json (or given out_dir) and produce report.
    This wrapper allows testing report generation without running the model graph.
    Returns path to report.md"""
    try:
        if out_dir is None:
            out_dir = DEFAULT_OUTPUT_DIR
        sol_path = os.path.join(out_dir, 'solution.json')
        if not os.path.exists(sol_path):
            raise FileNotFoundError(f'solution.json not found at {sol_path}')
        with open(sol_path, 'r', encoding='utf-8') as f:
            sol = json.load(f)
        return write_report(sol, out_dir=out_dir)
    except Exception:
        logger.exception('write_report_from_files failed')
        raise


if __name__ == '__main__':
    import argparse
    cli_arg_parser = argparse.ArgumentParser(description='Run LATS model for a question and save outputs')
    cli_arg_parser.add_argument('-q', '--question', type=str, help='Question to run the model on')
    args = cli_arg_parser.parse_args()
    q = args.question or input('Enter the question: ').strip()
    if not q:
        print('No question provided; exiting.')
    else:
        out = run_query(q)
        print('Saved outputs to:', out['solution_path'], out['best_trajectory_path'])
