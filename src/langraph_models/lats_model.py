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
import hashlib
from urllib.parse import urlparse

from src.langraph_models.research_ret import get_links_from_question , research_agent

try:
    import requests
except Exception:
    requests = None

# Allow skipping runtime side-effects (LLM calls) at import time. When set to '1',
# the module will avoid invoking LLM chains and other heavy operations so tooling
# (like graph rendering) can still run.
SKIP_SIDE_EFFECTS = os.environ.get("LATS_NO_SIDE_EFFECTS", "0") == "1"

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.prompt_values import ChatPromptValue
except Exception:
    # Minimal fallbacks so module imports even if langchain_core is not installed
    class BaseMessage:
        def __init__(self, content: str = ''):
            self.content = content
        def __repr__(self):
            return f"BaseMessage({self.content!r})"

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class ChatPromptValue:
        pass

try:
    from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
except Exception:
    class JsonOutputToolsParser:
        def __init__(self, return_id=False):
            pass
        def invoke(self, msg):
            return []
        def batch(self, msgs):
            return [ [] for _ in msgs ]

    class PydanticToolsParser:
        def __init__(self, tools=None):
            pass

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    class ChatPromptTemplate:
        @staticmethod
        def from_messages(msgs):
            # simple placeholder
            return ChatPromptTemplate()

    class MessagesPlaceholder:
        def __init__(self, variable_name=None, optional=False):
            self.variable_name = variable_name

try:
    from langchain_core.runnables import chain as as_runnable, RunnableConfig
except Exception:
    # Minimal runnable decorator that attaches an invoke method
    def as_runnable(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        # attach an invoke alias used in the code
        wrapper.invoke = func
        return wrapper
    RunnableConfig = object


from pydantic import BaseModel , Field

# Attempt to import the project's internal research retriever module and validate
# that it exposes the expected API. Provide a safe alias for AgentState when
# the real type isn't present so other code can still cast without crashing.
AgentState = dict  # fallback typing alias if real AgentState isn't present
research_ret_mod = None
research_tool_available = False
try:
    # Compute probe paths early so they are available in exception handlers
    _probe_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    _probe_src = os.path.join(_probe_root, 'src')
    # Ensure project paths are on sys.path regardless of retriever import outcome
    if _probe_root not in sys.path:
        sys.path.insert(0, _probe_root)
    if _probe_src not in sys.path:
        sys.path.insert(0, _probe_src)
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
        logging.warning("Failed to import src.langraph_models.research_ret: %s", e)
        logging.debug("research_ret import traceback:\n%s", traceback.format_exc())
        # Persist the traceback to a debug file so users can inspect import failures without a terminal
        try:
            out_dir = os.path.join(_probe_root, 'output') if '_probe_root' in globals() else os.path.join(os.getcwd(), 'output')
            os.makedirs(out_dir, exist_ok=True)
            dbg_path = os.path.join(out_dir, 'retriever_debug.txt')
            with open(dbg_path, 'a', encoding='utf-8') as _f:
                _f.write('\n---\n')
                _f.write(f'[{datetime.now().isoformat()}] Failed to import src.langraph_models.research_ret: {e}\n')
                _f.write(traceback.format_exc())
                _f.write('\n')
        except Exception:
            logging.exception('Failed to write retriever import debug file')
        # Fallback to shim so the rest of the system (report generation, CLI) remains functional
        try:
            research_ret_mod = importlib.import_module("src.langraph_models.research_ret_shim")
            # The shim provides a minimal get_tool_node implementation; mark retriever as available
            research_tool_available = True
            logging.info("Falling back to research_ret_shim; limited retriever functionality available.")
            if hasattr(research_ret_mod, "AgentState"):
                AgentState = getattr(research_ret_mod, "AgentState")
            # record that this is shim-backed for diagnostics
            research_ret_mod.__dict__['__is_shim__'] = True
        except Exception:
            research_ret_mod = None
            research_tool_available = False
            logging.warning("No research retriever available and shim import failed; provenance will be limited.")
except Exception:
    research_ret_mod = None
    research_tool_available = False
    logging.warning("No research retriever available; provenance will be limited.")

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





# Compute project root (two levels up from this file: src/langraph_models -> src -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ensure logs and outputs are written under the repository root for consistency
DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')


# Update DEFAULT_OUTPUT_DIR to ensure all outputs are written to 'src/langraph_models/output'
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

# CLI-tunable global defaults (can be overridden via command-line args in __main__)
# - CLI_N_CANDIDATES: number of candidate completions to generate
# - DEFAULT_SEARCH_NODES: how many default search nodes to run when no tool calls parsed
# - METADATA_TIMEOUT: timeout (seconds) for HTTP metadata fetches
# - ENABLE_METADATA_FETCH: whether to attempt fetching page metadata (title/DOI)
# - ENABLE_CJK_FILTER: whether to skip sources dominated by CJK characters
# - CLI_LOG_LEVEL: override module log level

# Try to obtain the real model, but tolerate failures and provide a DummyLLM
class _DummyLLM:
    def __init__(self):
        self.model_id = 'dummy-llm'
    def bind_tools(self, tools=None, tool_choice=None):
        return self
    def with_config(self, **kwargs):
        return self
    def invoke(self, inputs):
        # Accepts either a list or dict of messages; return a simple AIMessage
        try:
            return AIMessage(content='')
        except Exception:
            class _SimpleAI:
                content = ''
            return _SimpleAI()

try:
    if SKIP_SIDE_EFFECTS:
        raise RuntimeError('Side-effects disabled (SKIP_SIDE_EFFECTS)')
    llm = get_model("gpt-4o")
except Exception as e:
    logger = logging.getLogger(__name__)
    try:
        logger.info('LLM not available or side-effects disabled: %s. Using DummyLLM fallback.', str(e))
    except Exception:
        pass
    llm = _DummyLLM()

CLI_N_CANDIDATES: Optional[int] = None
DEFAULT_SEARCH_NODES: int = 4
METADATA_TIMEOUT: int = 6
ENABLE_METADATA_FETCH: bool = True
ENABLE_CJK_FILTER: bool = True
CLI_LOG_LEVEL: Optional[str] = None

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

# No external ToolNode here; we'll call research_ret functions directly and
# convert their dict outputs into AIMessage objects that downstream code can consume.
tool_node = None
tools = []

# Parser and initial answer chain used by generate_initial_response / expand
try:
    parser = JsonOutputToolsParser(return_id=True)
except Exception:
    class _SimpleParser:
        def invoke(self, msg):
            return []
        def batch(self, msgs):
            return [ [] for _ in msgs ]
    parser = _SimpleParser()

try:
    initial_answer_chain = prompt_template | llm.bind_tools ( tools = tools ).with_config (
        run_name = "GenerateInitialCandidate"
    )
except Exception:
    # fallback minimal chain-like placeholder
    class _SimpleChain:
        def invoke(self, inputs):
            return AIMessage(content="")
        # allow the fallback to be called directly like a runnable
        def __call__(self, inputs):
            return self.invoke(inputs)
    initial_answer_chain = _SimpleChain()

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
tools = []


def _is_unwanted_url(u: str) -> bool:
    """Return True for URLs we want to ignore (noisy localized search engines, binary files, etc.)."""
    try:
        p = urlparse(u)
        netloc = (p.netloc or "").lower()
        blacklist = (
            "baidu.com",
            "zhidao.baidu.com",
            "weibo.com",
            "sina.com",
            "toutiao.com",
            "qq.com",
            "185.199.108.153"  # example of GitHub raw IPs or others if needed
        )
        if any(netloc.endswith(b) for b in blacklist):
            return True
        # skip common non-HTML extensions
        if re.search(r"\.(pdf|jpg|jpeg|png|gif|zip|tar|gz|exe|svg|mp4|mp3)(?:$|\?)", p.path, re.IGNORECASE):
            return True
        return False
    except Exception:
        return True


def call_research_tool(tool_call: dict, query: str, agent_state_extra: Optional[dict] = None) -> dict:
    logger.info(
        "call_research_tool: received tool_call type=%s args_keys=%s query_snippet=%s",
        tool_call.get('type'),
        list(tool_call.get('args', {}).keys()) if isinstance(tool_call.get('args'), dict) else None,
        (query[:80] + '...') if query and len(query) > 80 else query,
    )
    if not research_tool_available:
        logger.warning("research retriever not available when calling tool: %s", tool_call.get('type'))
        return {"error": "research retriever not available", "__provenance__": {"sources": []}}

    ttype = (tool_call.get("type") or "").lower()
    if "google" in ttype:
        node_name, tool_name = "google", "search_google_detailed"
    elif "ddg" in ttype or "duck" in ttype:
        node_name, tool_name = "ddg", "DDGGeneralSearch"
    elif "wikipedia" in ttype:
        node_name, tool_name = "wikipedia", "WikipediaStructuredSearch"
    elif "youtube" in ttype:
        node_name, tool_name = "youtube", "YouTubeSerpAPISearch"
    elif "arxiv" in ttype or "arxivrawquery" in ttype:
        node_name, tool_name = "arxiv", "ArxivRawQuery"
    else:
        node_name, tool_name = "google", "search_google_detailed"

    # Build agent_state with the query and any extra context (links, hints)
    agent_state = cast(AgentState, {"query": query})
    if agent_state_extra and isinstance(agent_state_extra, dict):
        # Merge extras but don't overwrite explicit query if present
        for k, v in agent_state_extra.items():
            if k == 'query':
                continue
            agent_state[k] = v
    logger.debug("call_research_tool: invoking retriever.get_tool_node node=%s tool=%s agent_state_keys=%s query_snippet=%s", node_name, tool_name, list(agent_state.keys()), (query or '')[:200])
    # DEBUG: log full agent_state (may include links) at DEBUG level to trace what is sent to retriever
    logger.debug("call_research_tool: agent_state=%s", json.dumps(agent_state, ensure_ascii=False, default=str))
    try:
        res = research_ret_mod.get_tool_node(agent_state, node_name, tool_name)
        if isinstance(res, dict):
            content_lens = {k: (len(v) if isinstance(v, str) else None) for k, v in res.items()}
            logger.info("call_research_tool: result keys=%s content_lens=%s", list(res.keys()), content_lens)
            sources: list[dict] = []
            # structured detail log for provenance decisions (accepted/discarded + reason)
            prov_details: list[dict] = []

            def add_source(title: Optional[str], url: Optional[str], snippet: Optional[str] = None, origin: Optional[str] = None):
                # record a candidate detail regardless of acceptance so we can explain decisions
                detail = {
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'origin': origin or f"{node_name}/{tool_name}",
                    'accepted': False,
                    'reason': None,
                }
                if not title and not url:
                    detail['reason'] = 'no title/url'
                    prov_details.append(detail)
                    return
                if url:
                    # basic parse check: skip URLs that don't include a netloc (e.g., 'https://?')
                    try:
                        p = urlparse(url)
                        if not (p.scheme in ('http', 'https') and p.netloc):
                            detail['reason'] = 'malformed_url'
                            prov_details.append(detail)
                            logger.debug('add_source: skipping malformed URL %r', url)
                            return
                    except Exception:
                        detail['reason'] = 'urlparse_failed'
                        prov_details.append(detail)
                        logger.debug('add_source: urlparse failed for %r; skipping', url)
                        return
                    if _is_unwanted_url(url):
                        detail['reason'] = 'unwanted_domain_or_binary'
                        prov_details.append(detail)
                        return
                # heuristic: skip sources whose title/snippet is dominated by CJK characters
                def _is_cjk(text: Optional[str]) -> bool:
                    if not text:
                        return False
                    # count CJK characters
                    cjk = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\u3040-\u309f\u30a0-\u30ff]', text)
                    threshold = max(3, int(len(text) * 0.3))
                    return len(cjk) > threshold
                if ENABLE_CJK_FILTER and (_is_cjk(title) or _is_cjk(snippet)):
                    detail['reason'] = 'probable_cjk'
                    prov_details.append(detail)
                    logger.debug('add_source: skipping probable non-English/CJK source title=%s url=%s', title, url)
                    return
                # accepted
                entry = {}
                if title:
                    entry['title'] = title
                if url:
                    entry['url'] = url
                if snippet:
                    entry['snippet'] = snippet
                # ensure each returned source carries its origin (node/tool)
                entry['origin'] = detail['origin']
                sources.append(entry)
                detail['accepted'] = True
                detail['reason'] = 'accepted'
                prov_details.append(detail)

            for k in ("items", "results", "entries", "links", "rows"):
                if k in res and isinstance(res[k], (list, tuple)):
                    for itm in res[k]:
                        if isinstance(itm, dict):
                            url = itm.get('url') or itm.get('link') or itm.get('source')
                            title = itm.get('title') or itm.get('name') or itm.get('headline')
                            snippet = itm.get('snippet') or itm.get('summary') or itm.get('description')
                            add_source(title, url, snippet)

            # fallback to direct keys
            title = res.get('title') or res.get('name')
            url = res.get('url') or res.get('link')
            if title or url:
                add_source(title, url, res.get('snippet') or res.get('description'))

            # attempt to sniff urls in string-valued fields and include as url-only entries
            # Also extract URLs embedded in HTML/text (e.g. google_result) using regex and include snippets.
            url_pattern = re.compile(r"https?://[\w\-./?=&%#:]+", re.IGNORECASE)
            for k, v in res.items():
                if isinstance(v, str):
                    # quick check for full-url fields
                    if v.startswith('http://') or v.startswith('https://'):
                        if not _is_unwanted_url(v):
                            add_source(None, v, None)
                        continue
                    # search for embedded urls
                    found = url_pattern.findall(v)
                    if found:
                        # include up to first 5 urls with a short snippet from the text
                        for u in found[:5]:
                            if _is_unwanted_url(u):
                                continue
                            snippet = None
                            try:
                                idx = v.find(u)
                                start = max(0, idx - 80)
                                end = min(len(v), idx + len(u) + 80)
                                snippet = v[start:end].replace('\n', ' ')
                            except Exception:
                                snippet = v[:200]
                            add_source(None, u, snippet)

            res_with_sources = dict(res)
            # use the double-underscore provenance key consistently across the module
            res_with_sources.setdefault('__provenance__', {})
            # dedupe by url/title
            deduped = []
            seen = set()
            for s in sources:
                key = (s.get('url'), s.get('title'))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(s)

            # Optionally resolve metadata (title/doi) for each discovered URL to improve reports
            if ENABLE_METADATA_FETCH and requests is not None:
                try:
                    for src in deduped:
                        try:
                            u = src.get('url')
                            if u and not src.get('title'):
                                meta = _fetch_url_metadata(u, timeout=METADATA_TIMEOUT, output_dir=DEFAULT_OUTPUT_DIR)
                                if meta.get('title'):
                                    src['title'] = meta.get('title')
                                if meta.get('doi'):
                                    src['doi'] = meta.get('doi')
                                if meta.get('local_path'):
                                    src['local_path'] = meta.get('local_path')
                        except Exception:
                            logger.exception('call_research_tool: metadata fetch failed for %s', src.get('url'))
                except Exception:
                    logger.exception('call_research_tool: bulk metadata resolution failed')

            # attach provenance sources and details
            res_with_sources['__provenance__']['sources'] = deduped
            try:
                res_with_sources['__provenance__']['details'] = prov_details
            except Exception:
                res_with_sources['__provenance__']['details'] = []

            logger.info(
                "call_research_tool: extracted %d structured sources from node=%s tool=%s",
                len(deduped), node_name, tool_name,
            )
            return res_with_sources
        else:
            logger.info("call_research_tool: non-dict result type=%s", type(res))
            return {"content": str(res), "__provenance__": {"sources": []}}
    except Exception as e:
        logger.exception("research retriever call failed: %s", e)
        return {"error": str(e), "__provenance__": {"sources": []}}


def fetch_default_search_sources(query: str , max_nodes: int = 4, agent_state_extra: Optional[dict] = None) -> dict:
    """When the parser returns no tool calls, perform a few default searches
    (google, wikipedia, ddg) to gather structured provenance entries.
    Returns deduplicated list of provenance dicts {title?, url?, snippet?}.
    """
    if not research_tool_available:
        logger.debug ( "fetch_default_search_sources: research retriever unavailable" )
        # Return consistent structure even when retriever isn't present so callers can rely on keys
        return {'sources': [], 'details': []}
    # allow caller to override max_nodes, otherwise use DEFAULT_SEARCH_NODES
    if not max_nodes:
        max_nodes = DEFAULT_SEARCH_NODES

    # Only use the three default nodes requested: google, wikipedia, ddg
    nodes_and_tools = [
        ("google" , "search_google_detailed") ,
        ("wikipedia" , "WikipediaStructuredSearch") ,
        ("ddg" , "DDGGeneralSearch") ,
    ]
    gathered = [ ]
    gathered_details = []
    try:
        for i , (node , tool) in enumerate ( nodes_and_tools ):
            if i >= max_nodes:
                break
            try:
                fake_call = {"type": node , "args": {}}
                # Pass agent_state_extra (links/hints) so the retriever can use them
                raw = call_research_tool ( fake_call , query, agent_state_extra=agent_state_extra )
                if isinstance ( raw , dict ):
                    prov = raw.get('__provenance__', {}) or {}
                    sources = prov.get('sources', []) or []
                    details = prov.get('details', []) or []
                else:
                    sources = [ ]
                    details = []
                if sources:
                    gathered.extend ( sources )
                    logger.info (
                        "fetch_default_search_sources: gathered %d sources from %s/%s" , len ( sources ) , node , tool )
                if details:
                    gathered_details.extend(details)
            except Exception:
                logger.exception ( "fetch_default_search_sources: failed node=%s tool=%s" , node , tool )
    except Exception:
        logger.exception ( "fetch_default_search_sources: unexpected failure" )
    # dedupe by url/title for sources
    seen = set ( )
    dedup = [ ]
    for s in gathered:
        key = (s.get ( 'url' ) , s.get ( 'title' )) if isinstance ( s , dict ) else (None , str ( s ))
        if key in seen:
            continue
        seen.add ( key )
        dedup.append ( s )
    # dedupe details by url/title
    seen_d = set()
    dedup_details = []
    for d in gathered_details:
        if not isinstance(d, dict):
            dedup_details.append(d)
            continue
        keyd = (d.get('url'), d.get('title'))
        if keyd in seen_d:
            continue
        seen_d.add(keyd)
        dedup_details.append(d)
    # return both structured sources and details so callers (generate_initial_response) can include provenance decisions
    return {'sources': dedup, 'details': dedup_details}


def _safe_filename_for_url(url: str) -> str:
    """Return a short filename-safe hash for a URL."""
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()
    return h


def _fetch_url_metadata(url: str, timeout: int = None, output_dir: Optional[str] = None) -> dict:
    """Fetch a URL and extract title and DOI when possible, save HTML and metadata JSON.

    Returns a dict: { 'url', 'title', 'doi', 'status', 'local_path' }
    """
    if not url:
        return {'url': url, 'status': 'no_url'}
    if timeout is None:
        timeout = METADATA_TIMEOUT
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR  # Ensure output_dir defaults to the updated path
    meta = {'url': url, 'title': None, 'doi': None, 'status': None, 'local_path': None}
    if requests is None:
        meta['status'] = 'requests_unavailable'
        return meta
    headers = {
        'User-Agent': 'LATS-metadata-fetcher/1.0 (+https://example.invalid)'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        meta['status'] = f'fetch_failed: {e}'
        return meta
    meta['status'] = f'http_{resp.status_code}'
    # Save HTML snapshot if successful
    try:
        if resp.status_code == 200 and resp.text:
            h = _safe_filename_for_url(url)
            md_dir = os.path.join(output_dir, 'metadata')
            os.makedirs(md_dir, exist_ok=True)
            html_path = os.path.join(md_dir, f"{h}.html")
            try:
                with open(html_path, 'w', encoding='utf-8') as fh:
                    fh.write(resp.text)
                meta['local_path'] = html_path
            except Exception:
                meta['local_path'] = None
            text = resp.text
            # Extract title: og:title, meta name=title, <title>
            title = None
            m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', text, re.I)
            if m:
                title = m.group(1).strip()
            if not title:
                m = re.search(r'<meta[^>]+name=["\']title["\'][^>]+content=["\']([^"\']+)["\']', text, re.I)
                if m:
                    title = m.group(1).strip()
            if not title:
                m = re.search(r'<title[^>]*>([^<]+)</title>', text, re.I)
                if m:
                    title = m.group(1).strip()
            if title:
                meta['title'] = re.sub(r"\s+", ' ', title)
            # Try to find DOI: meta[name=citation_doi], doi.org links, or common DOI pattern
            doi = None
            m = re.search(r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']', text, re.I)
            if m:
                doi = m.group(1).strip()
            if not doi:
                m = re.search(r'https?://doi\.org/(10\.\d{4,9}/[^"\s<>]+)', text, re.I)
                if m:
                    doi = m.group(1).strip()
            if not doi:
                m = re.search(r'(10\.\d{4,9}/[A-Za-z0-9_\-./()+;]+)', text)
                if m:
                    doi = m.group(1).strip().rstrip('.')
            if doi:
                meta['doi'] = doi
    except Exception as e:
        meta['status'] = f'parse_failed: {e}'
    return meta


# Diagnostic route for exploring module behavior and state without a formal request/response
@as_runnable
def debug_inspect_state(state: TreeState) -> dict:
    """Inspect and expose internal state for debugging."""
    logger.info ( "debug_inspect_state: invoked" )
    try:
        if hasattr ( state , "model_dump" ):
            state_dict = state.model_dump ( )  # pydantic v2 preferred
        elif isinstance ( state , dict ):
            state_dict = state
        else:
            state_dict = dict ( vars ( state ) )
    except Exception:
        logger.exception ( "debug_inspect_state: failed to normalize state to dict" )
        raise

    # Basic summary of the state
    lines = [ ]
    try:
        lines.append ( "## Debug inspection of LATS internal state" )
        lines.append ( "" )
        lines.append ( f"- Model version: `{get_model('gpt-4o').model_id}`" )
        lines.append ( f"- Research retriever available: {bool(research_tool_available)}" )
        lines.append ( f"- Input query: {state_dict.get('input')}" )
        lines.append ( f"- Number of tool calls parsed: {len(state_dict.get('tool_calls') or [])}" )
        lines.append ( f"- Number of sources gathered: {len(state_dict.get('sources') or [])}" )
        lines.append ( f"- Number of children in root node: {len(state_dict.get('root', {}).get('children') or [])}" )
        lines.append ( f"- Depth of root node: {state_dict.get('root', {}).get('depth')}" )
        lines.append ( "" )
        lines.append ( "### Raw state content (truncated)" )
        lines.append ( "" )
        # Dump the entire state for inspection
        dump = json.dumps ( state_dict , ensure_ascii = False , indent = 2 )
        for line in dump.split ( '\n' ):
            lines.append ( "> " + line )
    except Exception:
        logger.exception ( "Failed to summarize state content" )
        lines.append ( "> <error generating state summary>" )

    # Join and truncate final output to avoid excessive size
    full_output = "\n".join ( lines )
    if len ( full_output ) > 4000:
        full_output = full_output[ :4000 ] + "\n...<truncated>"

    logger.info ( "debug_inspect_state: output prepared" )
    return {"content": full_output}


# Utility function to check JSON serializability
def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


# Enhanced generate_initial_response with improved JSON serialization, provenance, and error handling
@as_runnable
def generate_initial_response(state: TreeState) -> dict:
    logger.info("generate_initial_response: start; input_snippet=%s", (getattr(state, 'input', '') or '')[:120])
    try:
        # Normalize state to dict - handle various input types
        if hasattr(state, 'model_dump'):
            state_dict = state.model_dump()
        elif isinstance(state, dict):
            state_dict = state
        else:
            # Try to convert to dict or create a default dict
            try:
                if hasattr(state, '__dict__'):
                    state_dict = dict(vars(state))
                else:
                    state_dict = {'input': str(state)}
            except Exception:
                state_dict = {'input': str(state)}
    except Exception:
        logger.exception('generate_initial_response: failed to normalize state')
        state_dict = {}

    query = state_dict.get('query') or state_dict.get('input') or ''
    all_sources, all_details, tool_responses = [], [], []
    # Predeclare evaluated_branches so later references are safe even if branch eval is skipped
    evaluated_branches = []

    # Ingest provided links and ensure they are present in all_sources (origin='user').
    provided = state_dict.get('root_sources') or state_dict.get('links') or []
    provided_urls = []
    for item in provided:
        try:
            if isinstance(item, str):
                url = item
            else:
                url = item.get('url') or item.get('link')
            if not url:
                continue
            provided_urls.append(url)
            # Avoid adding duplicates
            if any(isinstance(s, dict) and s.get('url') == url for s in all_sources):
                continue
            src = {'url': url, 'origin': 'user'}
            # Try to resolve metadata for provided links (title/doi/local_path)
            if ENABLE_METADATA_FETCH and requests is not None and url.startswith(('http://', 'https://')):
                try:
                    meta = _fetch_url_metadata(url, timeout=METADATA_TIMEOUT, output_dir=DEFAULT_OUTPUT_DIR)
                    if meta.get('title'):
                        src['title_resolved'] = meta.get('title')
                    if meta.get('doi'):
                        src['doi'] = meta.get('doi')
                    if meta.get('local_path'):
                        src['local_path'] = meta.get('local_path')
                except Exception:
                    logger.exception('generate_initial_response: metadata fetch failed for provided url=%s', url)
            all_sources.append(src)
        except Exception:
            logger.exception('generate_initial_response: failed ingesting provided link')

    # Agent state extras to pass to research retriever (includes provided links)
    agent_state_extra = {}
    if provided:
        # normalize to list of URLs
        try:
            urls = []
            for it in provided:
                if isinstance(it, str):
                    urls.append(it)
                elif isinstance(it, dict) and it.get('url'):
                    urls.append(it.get('url'))
            if urls:
                agent_state_extra['links'] = urls
        except Exception:
            agent_state_extra['links'] = provided
    logger.info('generate_initial_response: using agent_state_extra=%s', agent_state_extra)

    # Run initial answer chain
    try:
        seed = initial_answer_chain.invoke({'input': query}) if hasattr(initial_answer_chain, 'invoke') else initial_answer_chain({'input': query})
        output_messages = [AIMessage(content=getattr(seed, 'content', str(seed)))]
        logger.info('generate_initial_response: created %d output_messages', len(output_messages))
    except Exception:
        logger.exception('generate_initial_response: initial_answer_chain failed')
        output_messages = [AIMessage(content='')]

    # Parse tool calls
    try:
        parsed = parser.invoke(output_messages[-1]) or []
        parsed = [parsed] if isinstance(parsed, dict) else parsed
        logger.info('generate_initial_response: parser returned %d tool calls', len(parsed))
    except Exception:
        logger.exception('generate_initial_response: parser.invoke failed')
        parsed = []

    # Call research tools (parsed tool calls). We pass agent_state_extra (links) so the retriever can contextualize queries.
    for call in parsed:
        try:
            raw = call_research_tool(call, query, agent_state_extra=agent_state_extra) if research_tool_available else {'content': '', '__provenance__': {'sources': [], 'details': []}}
            content = next((v for k, v in raw.items() if k.endswith('_result') and isinstance(v, str)), raw.get('content', ''))
            prov = raw.get('__provenance__', {}).get('sources', []) or []
            details = raw.get('__provenance__', {}).get('details', []) or []
            # normalize provenance entries to have title/url/snippet keys
            for s in prov:
                if isinstance(s, dict):
                    all_sources.append({'title': s.get('title'), 'url': s.get('url'), 'snippet': s.get('snippet'), 'origin': call.get('type')})
            for d in details:
                all_details.append(d)
            tool_responses.append(getattr(content, 'content', content) if hasattr(content, 'content') else content)
        except Exception:
            logger.exception('generate_initial_response: call_research_tool failed')

    # Fetch default sources to augment provided links unless the user explicitly requested skipping.
    try:
        should_skip_default = bool(globals().get('SKIP_DEFAULT_SEARCH', False))

        if not should_skip_default:
            try:
                fetched = fetch_default_search_sources(query, max_nodes=DEFAULT_SEARCH_NODES, agent_state_extra=agent_state_extra)
                sources_fetched = fetched.get('sources', []) or []
                # Merge fetched sources with provided ones, dedup by url
                seen_urls = set()
                for s in all_sources:
                    if isinstance(s, dict) and s.get('url'):
                        seen_urls.add(s.get('url'))
                for s in sources_fetched:
                    try:
                        if isinstance(s, dict):
                            url = s.get('url')
                            if url and url in seen_urls:
                                continue
                            s.setdefault('origin', None)
                            all_sources.append(s)
                            if url:
                                seen_urls.add(url)
                        else:
                            u = str(s)
                            if u in seen_urls:
                                continue
                            all_sources.append({'url': u, 'origin': None})
                            seen_urls.add(u)
                    except Exception:
                        # Protect merge loop from malformed source entries
                        logger.exception('generate_initial_response: failed merging fetched source: %s', repr(s))
                        continue
                all_details.extend(fetched.get('details', []) or [])
                # Evaluate candidate branches using the LLM and reflection_chain.
                evaluated_branches = []
                try:
                    max_branches = min(len(all_sources), CLI_N_CANDIDATES or 5)
                    # Prepare a compact sources list for prompts (limit to 20 for prompt size)
                    compact_sources = []
                    for ss in all_sources[:20]:
                        if isinstance(ss, dict):
                            compact_sources.append(f"{ss.get('title_resolved') or ss.get('title') or ss.get('url')}: {ss.get('url')}")
                        else:
                            compact_sources.append(str(ss))
                    compact_sources_text = "\n".join([f"- {x}" for x in compact_sources])
                    for idx, src in enumerate(all_sources[:max_branches]):
                        try:
                            src_url = src.get('url') if isinstance(src, dict) else str(src)
                            src_title = (src.get('title_resolved') or src.get('title')) if isinstance(src, dict) else None
                            snippet = src.get('snippet') if isinstance(src, dict) else None
                            prompt_text = (
                                f"{query}\n\n" 
                                f"Available sources (excerpt):\n{compact_sources_text}\n\n"
                                f"Focus on evidence from: {src_url}\n"
                                "Produce a concise (2-4 sentences) evidence-backed paragraph in English that answers the question, and include an inline citation to the URL."
                            )
                            # Generate candidate text
                            try:
                                cand = initial_answer_chain.invoke({'input': prompt_text}) if hasattr(initial_answer_chain, 'invoke') else initial_answer_chain({'input': prompt_text})
                                cand_text = getattr(cand, 'content', str(cand))
                            except Exception:
                                logger.exception('generate_initial_response: candidate generation failed for %s', src_url)
                                cand_text = ''
                            # Score candidate using reflection_chain
                            score = 0.0
                            found_solution = False
                            try:
                                refl_input = {'candidate': [AIMessage(content=cand_text)], 'input': query}
                                refl = reflection_chain.invoke(refl_input) if hasattr(reflection_chain, 'invoke') else reflection_chain(refl_input)
                                # normalized_score property if Reflection instance present
                                score = float(getattr(refl, 'normalized_score', getattr(refl, 'score', 0)) or 0.0)
                                found_solution = bool(getattr(refl, 'found_solution', False))
                            except Exception:
                                logger.exception('generate_initial_response: reflection failed for branch %s', src_url)
                            evaluated_branches.append({'source': src, 'candidate': cand_text, 'score': round(float(score), 3), 'found_solution': found_solution})
                        except Exception:
                            logger.exception('generate_initial_response: failed evaluating branch for source: %s', repr(src))
                    # sort desc
                    evaluated_branches.sort(key=lambda b: b.get('score', 0), reverse=True)
                except Exception:
                    logger.exception('generate_initial_response: overall branch evaluation failed')
                # attach branches to details for reporting
                if evaluated_branches:
                    all_details.append({'evaluated_branches': [{'url': b['source'].get('url') if isinstance(b['source'], dict) else str(b['source']), 'score': b['score']} for b in evaluated_branches]})
            except Exception:
                logger.exception('generate_initial_response: fetch_default_search_sources failed')
    except Exception:
        logger.exception('generate_initial_response: fetch_default_search_sources failed')

    # Build final response with computed final_info (avoid always-returning defaults)
    try:
        initial_response = output_messages[0].content if output_messages else ''

        # Heuristics for final_info fields
        computed_children = len(all_sources) if isinstance(all_sources, (list, tuple)) else 0
        computed_root_depth = int(state_dict.get('root_depth', 1))
        # If parser produced tool calls, assume deeper solution depth proportional to number of parsed calls
        parsed_calls_count = 0
        try:
            parsed_calls_count = len(parsed) if isinstance(parsed, (list, tuple)) else 0
        except Exception:
            parsed_calls_count = 0
        computed_solution_depth = int(state_dict.get('solution_depth', max(1, min(3, 1 + parsed_calls_count))))

        # Heuristic confidence score (0.0 - 1.0): base + signals from tool_responses and number of sources
        conf = 0.4
        if tool_responses:
            conf += 0.2
        try:
            conf += 0.2 * (min(5, computed_children) / 5.0)
        except Exception:
            pass
        # Cap and round
        conf = max(0.0, min(0.99, conf))
        conf = round(conf, 2)

        # If branches were evaluated, include them and pick the best candidate as the synthesized best trajectory
        best_trajectory = state_dict.get('best_trajectory', [initial_response])
        if 'evaluated_branches' in locals() and evaluated_branches:
            final_candidate = evaluated_branches[0]['candidate']
            best_trajectory = [final_candidate] + best_trajectory
        final_response = {
            'question': query,
            'root_sources': all_sources,
            'evaluated_branches': evaluated_branches if 'evaluated_branches' in locals() else [],
            'provided_links': provided_urls,
            'solution_sources': [],
            'details': all_details,
            'tool_responses': tool_responses,
            'initial_response': initial_response,
            'final_info': {
                'root_depth': computed_root_depth,
                'children': computed_children,
                'solution_depth': computed_solution_depth,
                'solution_value': conf
            },
            'steps': state_dict.get('steps', ['start']),
            'best_trajectory': best_trajectory
        }
        # Validate JSON serialization
        if not is_json_serializable(final_response):
            raise ValueError("Final response contains non-serializable fields")
        logger.info('generate_initial_response: completed successfully')
        return final_response
    except Exception:
        logger.exception('generate_initial_response: failed to build final response')
        return {'question': query, 'error': 'Failed to generate response'}


def main(question: Optional[str] = None, links: Optional[list] = None, argv: Optional[list] = None, return_result: bool = False):
    import argparse
    # main now accepts optional: question, links, argv
    # - If called programmatically, pass question and links directly.
    # - If argv is provided, argparse will parse that list instead of sys.argv.
    # We will update module-level tunables via globals() to avoid needing a 'global' statement

    ap = argparse.ArgumentParser ( description = "LATS: Language Model Analysis Tool" )
    ap.add_argument ( "-q" , "--question" , help = "The question or prompt to analyze" )
    ap.add_argument ( "-Q" , "--question-file", help = "Read the question from a file (path)" )
    ap.add_argument ( "-l" , "--link" , action = "append" , help = "Provide a direct link to include in the analysis" )
    ap.add_argument ( "--no-side-effects" , action = "store_true" , help = "Skip any runtime side-effects (LLM calls, etc.)" )
    ap.add_argument ( "--debug" , action = "store_true" , help = "Enable debug-level logging" )
    # Tunable parameters to improve search/metadata behavior
    ap.add_argument("--max-search-nodes", type=int, default=DEFAULT_SEARCH_NODES, help="Max number of default search nodes to run when parser finds no tool calls")
    ap.add_argument("--metadata-timeout", type=int, default=METADATA_TIMEOUT, help="Timeout (seconds) for fetching URL metadata")
    ap.add_argument("--no-metadata-fetch", action='store_true', help="Disable metadata HTTP fetches for discovered URLs")
    ap.add_argument("--no-cjk-filter", action='store_true', help="Disable filtering out probable CJK-language sources")
    ap.add_argument("--candidates", type=int, default=CLI_N_CANDIDATES or 5, help="Number of LLM candidate completions to generate")
    ap.add_argument("--log-level", choices=['DEBUG','INFO','WARNING','ERROR'], default=CLI_LOG_LEVEL or 'INFO', help="Override log level")
    # Parse either provided argv (for programmatic calls/tests) or default to sys.argv
    args = ap.parse_args(argv if argv is not None else None)

    # If caller provided question/links programmatically, override parsed args
    if question:
        args.question = question
    if links:
        # ensure args.link is a list (argparse uses None if not provided)
        args.link = links

    # Log parsed CLI args for easier debugging when runs appear to stop early
    logger.debug('CLI args: %s', args)

    if args.no_side_effects:
        os.environ['LATS_NO_SIDE_EFFECTS'] = '1'

    # Apply tunable parameters to module-level globals via globals()
    mg = globals()
    mg['DEFAULT_SEARCH_NODES'] = args.max_search_nodes or mg.get('DEFAULT_SEARCH_NODES')
    mg['METADATA_TIMEOUT'] = args.metadata_timeout or mg.get('METADATA_TIMEOUT')
    mg['ENABLE_METADATA_FETCH'] = not args.no_metadata_fetch
    mg['ENABLE_CJK_FILTER'] = not args.no_cjk_filter
    mg['CLI_N_CANDIDATES'] = args.candidates
    mg['CLI_LOG_LEVEL'] = args.log_level

    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    question = args.question
    provided_links = args.link
    question_file = args.question_file
    logger.debug('question arg: %s; provided_links: %s; question_file: %s', question, provided_links, question_file)

    # If no question provided via CLI, attempt environment variable, question file, or piped stdin
    if not question:
        # 1) Environment variable
        q_env = os.environ.get('LATS_QUESTION')
        if q_env:
            question = q_env
            logger.info('Using question from LATS_QUESTION environment variable')

    if not question and question_file:
        try:
            with open(question_file, 'r', encoding='utf-8') as _f:
                q_text = _f.read().strip()
            if q_text:
                question = q_text
                logger.info('Using question read from file: %s', question_file)
        except Exception:
            logger.exception('Failed to read question from file: %s', question_file)

    # If still no question and stdin appears to have piped content, read it
    if not question:
        try:
            if sys.stdin is not None and not sys.stdin.isatty():
                # Read all available stdin (this will block if nothing is piped, but in non-interactive CI it will contain data)
                try:
                    piped = sys.stdin.read()
                except Exception:
                    piped = ''
                if piped and piped.strip():
                    question = piped.strip()
                    logger.info('Using question read from piped stdin')
        except Exception:
            logger.exception('Error while attempting to read piped stdin')

    # If still no question, optionally allow an automated dev run when LATS_AUTO_RUN=1
    if not question:
        if os.environ.get('LATS_AUTO_RUN') == '1':
            # Use provided env question or a harmless default prompt for dev/testing
            question = os.environ.get('LATS_QUESTION') or 'Test: summarize recent research on wireless power transfer.'
            logger.info('LATS_AUTO_RUN enabled; using auto question for dev/testing')
        else:
            # Do not prompt interactively (avoids hanging in IDEs); require explicit input
            logger.info('No question provided; exiting to avoid interactive prompt.')
            print('No question provided. Please provide the question using one of:')
            print(' - CLI: --question/-q "Your question"')
            print(' - File: --question-file /path/to/file.txt')
            print(' - Env: set LATS_QUESTION="Your question"')
            print(' - Pipe: echo "Your question" | python ...')
            sys.exit(2)

    # Initial log message with obfuscated question
    logger.info('Analysis starting for question: %s', question[:min(len(question), 50)] + ('...' if len(question) > 50 else ''))

    # Construct initial state. Use a plain dict so we can include arbitrary keys (links/root_sources)
    initial_state = {
        'input': question,
        'links': provided_links or [],
        'root_sources': provided_links or [],
    }

    # Run the analysis. Support both plain callables and runnable wrappers from langchain that expose .invoke
    try:
        # Prefer .invoke() if this is a runnable wrapper (langchain-style),
        # otherwise fall back to calling the function directly.
        if hasattr(generate_initial_response, 'invoke'):
            logger.debug('Invoking generate_initial_response via .invoke()')
            result = generate_initial_response.invoke(initial_state)
        elif callable(generate_initial_response):
            logger.debug('Invoking generate_initial_response as callable')
            result = generate_initial_response(initial_state)
        else:
            raise RuntimeError('generate_initial_response is not callable and has no .invoke')
    except Exception:
        logger.exception('Unexpected error in analysis pipeline')
        print('An error occurred while processing the request. Please check the logs for details.')
        sys.exit(1)

    # Output result (for now, just pretty-print the JSON)
    try:
        import json
        # Print for CLI callers, suppress printing when return_result is True
        if not return_result:
            print(json.dumps(result, ensure_ascii=False, indent=2))

        # Ensure results are written to the output directory
        if not os.path.exists(DEFAULT_OUTPUT_DIR):
            os.makedirs(DEFAULT_OUTPUT_DIR)

        output_files = {
            'schema': os.path.join(DEFAULT_OUTPUT_DIR, 'schema.txt'),
            'trajectory': os.path.join(DEFAULT_OUTPUT_DIR, 'best_trajectory.txt'),
            'solution': os.path.join(DEFAULT_OUTPUT_DIR, 'solution.json'),
            'report': os.path.join(DEFAULT_OUTPUT_DIR, 'report.md')
        }

        # Write outputs to files (always perform this side-effect so both CLI and programmatic
        # callers have consistent artifacts to inspect)
        output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # schema.txt - final best response text (English)
        schema_path = os.path.join(output_dir, 'schema.txt')

        # Ensure we have a sources list (may have been defined during metadata resolution)
        sources = locals().get('sources', result.get('root_sources', []) or [])

        # Resolve metadata for sources if available and enabled to avoid truncated titles
        try:
            if ENABLE_METADATA_FETCH and requests is not None and sources:
                for src in sources:
                    try:
                        if isinstance(src, dict) and src.get('url') and not src.get('title_resolved'):
                            meta = _fetch_url_metadata(src.get('url'), timeout=METADATA_TIMEOUT, output_dir=DEFAULT_OUTPUT_DIR)
                            if meta.get('title'):
                                src['title_resolved'] = meta.get('title')
                            if meta.get('doi'):
                                src['doi'] = meta.get('doi')
                            if meta.get('local_path'):
                                src['local_path'] = meta.get('local_path')
                    except Exception:
                        logger.exception('Failed to resolve metadata for source %s', src.get('url') if isinstance(src, dict) else str(src))
        except Exception:
            logger.exception('Pre-report metadata resolution failed')

        with open(schema_path, 'w', encoding='utf-8') as f:
            f.write("# Filtered Search Analysis\n\n")
            f.write(f"### Query: {result.get('question', '')}\n\n")
            details = result.get('details', [])
            tool_responses = result.get('tool_responses', [])

            # Group sources by origin
            from collections import defaultdict
            sources_by_origin = defaultdict(list)
            for src in sources:
                origin = src.get('origin', 'Unknown') if isinstance(src, dict) else 'Unknown'
                sources_by_origin[origin].append(src)

            for origin, src_list in sources_by_origin.items():
                f.write(f"### Source origin: {origin}\n")
                for src in src_list:
                    # Prefer resolved metadata title; fall back to provided title; normalize capitalization
                    title = None
                    url = ''
                    snippet = ''
                    doi = None
                    if isinstance(src, dict):
                        title = src.get('title_resolved') or src.get('title') or None
                        url = src.get('url', '')
                        snippet = src.get('snippet', '')
                        doi = src.get('doi')
                    else:
                        title = str(src)
                    if title:
                        title = title.strip()
                        if title and title[0].islower():
                            title = title[0].upper() + title[1:]
                    else:
                        title = 'Untitled'
                    f.write(f"- **{title}**\n")
                    if snippet:
                        f.write(f"  {snippet}\n")
                    if doi:
                        f.write(f"  DOI: {doi}\n")
                    if url:
                        f.write(f"  Link: {url}\n")
                    f.write("\n")

            # Add synthesis
            f.write("## Final synthesis\n\n")
            initial_response = result.get('initial_response', '')
            if tool_responses:
                synthesis = " ".join(tool_responses)
                f.write(f"{synthesis}\n")
            elif initial_response:
                f.write(f"{initial_response}\n")
            else:
                f.write("Could not generate a synthesis.\n")
        logger.info('run_query: wrote schema.txt')

        # best_trajectory.txt - plain textual trajectory
        trajectory_path = os.path.join(output_dir, 'best_trajectory.txt')
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {result.get('question', '')}\n\n")
            best_trajectory = result.get('best_trajectory', [])
            for i, resp in enumerate(best_trajectory):
                f.write(f"Step {i+1}: {resp}\n\n")
        logger.info('run_query: wrote best_trajectory.txt')

        # solution.json - structured dump
        solution_path = os.path.join(output_dir, 'solution.json')
        with open(solution_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info('run_query: wrote solution.json')

        # report.md - markdown report (English)
        report_path = os.path.join(output_dir, 'report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LATS Analysis Report\n\n")
            f.write(f"**Query:** {result.get('question', '')}\n\n")
            f.write("## Sources Found\n\n")

            # Ensure we reuse the same resolved sources list
            report_sources = sources or result.get('root_sources', [])

            # Filter out dictionary/glossary sources and keep only relevant scientific/academic links
            filtered_sources = []
            for src in report_sources:
                if isinstance(src, dict):
                    url = src.get('url', '')
                    title = src.get('title_resolved') or src.get('title') or ''
                    snippet = src.get('snippet', '')
                else:
                    url = ''
                    title = str(src)
                    snippet = ''

                if any(domain in url for domain in [
                    'dictionary.cambridge.org', 'wordreference.com', 'collinsdictionary.com', 'wiktionary.org', 'thesaurus.com']):
                    continue
                if not title or len(title) < 8:
                    if snippet:
                        sentences = re.split(r'[.!?]', snippet)
                        title = sentences[0].strip() if sentences else snippet[:200].strip()
                        if not title:
                            title = snippet[:200].strip()
                        if title and title[0].islower():
                            title = title[0].upper() + title[1:]
                if not title:
                    title = 'Untitled'
                filtered_sources.append({'title': title, 'url': url, 'snippet': snippet})

            for src in filtered_sources:
                f.write(f"- [{src['title']}]({src['url']})\n")
                if src['snippet']:
                    f.write(f"  - {src['snippet']}\n")
                f.write("\n")

            # Add best_trajectory and final_info
            f.write("## Trajectory and Solution Info\n\n")
            best_trajectory = result.get('best_trajectory', [])
            for i, step in enumerate(best_trajectory):
                f.write(f"Step {i+1}: {step}\n\n")
            final_info = result.get('final_info', {})
            if final_info:
                f.write(f"Final Info: {json.dumps(final_info, ensure_ascii=False, indent=2)}\n\n")
            f.write("## Search provenance and decisions\n\n")
            details = result.get('details', [])
            if details:
                for d in details:
                    if isinstance(d, dict):
                        f.write(f"- Origin: {d.get('origin') or 'unknown'} — URL: {d.get('url') or 'N/A'} — Accepted: {d.get('accepted', False)} — Reason: {d.get('reason')}\n")
                    else:
                        f.write(f"- {d}\n")
            else:
                f.write("No detailed provenance collected for this run.\n")
            f.write("\n## Tool responses and intermediate outputs\n\n")
            initial_response = result.get('initial_response', '')
            if initial_response:
                f.write(f"- Initial model output: {initial_response}\n")
            tool_responses = result.get('tool_responses', [])
            for resp in tool_responses:
                f.write(f"- {resp}\n")
        logger.info('run_query: wrote report.md')

        # Return or exit depending on caller preference
        if return_result:
            return result
        else:
            # Successful exit for CLI
            sys.exit(0)
    except Exception:
        if not return_result:
            print('<error formatting result as JSON>')
        logger.exception('Error writing outputs')

    # Return or exit depending on caller preference
    if return_result:
        return result
    else:
        # Successful exit for CLI
        sys.exit(0)



# if __name__ == "__main__":
#     # Run CLI main which parses sys.argv. To call programmatically, import
#     # main() and invoke with question/links/return_result=True from your script.
#     main()
if __name__ == "__main__":
    question="What are the most frequent haplotypes in the Pyrenees from the Paleolithic to the Iron Age?"
    # Default behavior: run CLI main which parses sys.argv. To run programmatically,
    # import main() and call it with question/links/return_result=True from your script.
    main(    question=question ,     links=get_links_from_question(question)
             ,
    return_result=True)

def invoke_research_agent(question):
    """
    Function to invoke the research agent with a given question.
    Returns a dictionary containing the links found.
    """
    initial_state = {
        "query": question,
        "retry_count": 0,
        "failed_nodes": [],
        "node_ready": {},
        "synthesis": ""
    }

    try:
        answer = research_agent.invoke(initial_state)
        return {
            "status": "success",
            "links": answer.get("links", [])
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
