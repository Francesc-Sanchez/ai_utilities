RADME_LATS

Summary
-------
This document explains how the LATS model implemented in `src/langraph_models/lats_model.py` works, how its components interact, how to run it, and how to diagnose the common "no links returned" problem. The goal is to provide a clear mental model of the pipeline so you can extend, debug, or adapt it.

High-level architecture
-----------------------
The LATS model implements a small search-based reasoning loop around an LLM. The main ideas are:
- Generate a set of candidate continuations (responses) for a user question.
- Use a reflection/critic LLM chain to score and annotate each candidate.
- Expand the best candidate into new candidates (rollout/expansion) and repeat until a stopping condition.
- Optionally call an internal research retriever to fetch structured provenance (URLs, snippets) that can be attached to candidates and reflections.
- Use a simple tree search with UCT-like metrics to balance exploration and exploitation across candidate branches.

Key components and data structures
----------------------------------
- TreeState (Pydantic model)
  - The serialized state passed between graph nodes. Contains at minimum `input` (the user question) and optional `root` (a Node instance).

- Node
  - Represents a node in the search tree. Stores:
    - messages: a list of `BaseMessage` (the conversational messages in this branch)
    - reflection: a `Reflection` object with score and commentary
    - metadata: free-form dict used to store `sources`/provenance and tool call summaries
    - value and visits: numeric statistics used for selection and backpropagation
  - Methods: get_trajectory(), backpropagate(), upper_confidence_bound(), get_best_solution().

- Reflection (Pydantic model)
  - Stores critique text, score (0-10), boolean found_solution, and structured `sources` list. Also can be converted to a HumanMessage for LLM input.

- LLM chains
  - initial_answer_chain / expansion_chain: prompt templates + LLM that generate candidate text. The code wraps the LLM with a Pydantic parser (JsonOutputToolsParser) to detect structured tool calls.
  - reflection_llm_chain: prompt + LLM used to critique and score candidate messages; output is parsed into the `Reflection` Pydantic model.

- Parser
  - JsonOutputToolsParser parses LLM output for tool call structures (id/type/args). When the LLM includes explicit tool calls, the model handles those.

- research retriever (internal)
  - Dynamically imported module `src.langraph_models.research_ret` (detected via importlib at module import time).
  - Expected API: `get_tool_node(agent_state, node_name, tool_name)` which returns dict-like search results.
  - When available, `call_research_tool()` maps parsed tool calls to node/tool names (google, ddg, wikipedia, arxiv, youtube) and invokes `get_tool_node()`.
  - The module extracts structured `sources` (title/url/snippet) from returned result objects and attaches them to responses/reflections.

Data flow (simplified)
----------------------
1. run_query(question) constructs a TreeState and starts the graph stream.
2. generate_initial_response(state):
   - Calls initial_answer_chain to generate an initial AIMessage candidate.
   - Uses `parser` to find tool calls in that candidate. If parsed tool calls exist and the retriever is available, calls `call_research_tool()` to fetch structured sources and converts results to AIMessages inserted into the candidate messages.
   - If the parser returns no tool calls, the code may proactively run default searches (google, wikipedia, ddg) via `fetch_default_search_sources()` to populate provenance.
   - Calls reflection_chain to score and annotate the candidate(s), attaches provenance to the Reflection if present, and returns a TreeState with a root Node.

3. expand(state, config):
   - Selects a leaf node via select(root) using `upper_confidence_bound` on children.
   - Calls expansion_chain to produce new candidate messages for that trajectory.
   - Parses tool calls for each candidate, calls retriever as needed, and constructs output_messages for reflection.
   - Runs reflection_chain.batch() to produce reflections for each candidate, creates child Node objects, attaches provenance and metadata, and backpropagates scores.
   - Updates the state's provenance summary and returns the mutated state.

4. The graph loop (`should_loop`) repeats expand until a stopping condition: solution found, tree height limit, or no children.

Search and selection strategy
-----------------------------
- Each Node records `value` and `visits`.
- `upper_confidence_bound()` implements a UCT-like formula balancing average reward and exploration term.
- When expanding, the algorithm selects the child with the highest UCT and rolls out from it. After getting a reflection score, it backpropagates normalized reward up the tree.
- get_best_solution() returns the terminal node with the highest value among solution trajectories.

Provenance extraction
---------------------
The code attempts to extract structured provenance from retriever responses by inspecting common keys (items, results, entries, links, rows) and also scanning string fields for embedded URLs via regex. Collected provenance is deduplicated by URL/title and attached to:
- Node.metadata['sources']
- Reflection.sources
- top-level `out_state['provenance']` (compact summary for programmatic consumption)

Why "no links" can still happen
--------------------------------
- `research_ret` module not importable or missing `get_tool_node` -> `research_tool_available` set False. The code logs the full traceback in that case.
- retriever returns empty or unexpected structures -> provenance extraction yields no sources.
- Parser fails to parse tool calls from the LLM output; the model then runs default searches, but if retriever responses are empty, no links are produced.

How to run (PowerShell)
------------------------
Activate venv and run the module from the project root (recommended):

```powershell
& 'C:\Users\franc\PycharmProjects\ai_utilities\.venv\Scripts\Activate.ps1'
python -m src.langraph_models.lats_model -q "What is the current status of lithium pollution research?"
```

Or run interactively:

```powershell
python src\langraph_models\lats_model.py
# then type/paste a question at the prompt
```

Files produced
--------------
- `output/schema.txt` — final best response text
- `output/solution.json` — structured dump with question, provenance, best_trajectory, and steps
- `output/best_trajectory.txt` — plain textual trajectory saved for debugging
- `logs/lats_model.log` — detailed logs and (important) tracebacks when `research_ret` import fails

Debugging checklist (detailed)
-----------------------------
1. Confirm you run from repository root. Running with `-m src.langraph_models.lats_model` ensures `src` is a package root.
2. Verify virtualenv dependencies are installed (inside venv):

```powershell
pip install -r requirements.txt
```

3. Test retriever import to get an immediate traceback if something fails:

```powershell
python -c "import importlib, traceback
try:
  m = importlib.import_module('src.langraph_models.research_ret')
  print('OK, loaded:', hasattr(m,'get_tool_node'))
except Exception:
  traceback.print_exc()"
```

4. Check `logs/lats_model.log` for the entry `research_ret import traceback` — the module logs the full import exception and stack trace.
5. If retriever imports ok but produces no sources, call `get_tool_node` manually with a small agent_state (e.g., {'query': 'your question'}) and inspect returned dict keys.

Extensions & next steps
-----------------------
- Add unit tests for `call_research_tool()` to assert provenance extraction on sample retriever outputs.
- Add a small smoke-test script that runs the retriever import and a default search and writes a short report to `logs/`.
- Consider adding a configurable maximum tree height and time limits for production use.

If you want, I will:
- Add the smoke-test script now and wire it into `scripts/smoke_lats.py`.
- Add a `requirements-lite.txt` with the minimal packages needed to run the module (I can produce a best-effort list based on imports).

Recent Updates and Known Issues
--------------------------------
### Fixed Issues
- **State Normalization Error**: Fixed `TypeError: vars() argument must have __dict__ attribute` in `generate_initial_response` by improving state normalization to handle various input types (dicts, objects without `__dict__`, etc.).
- **JSON Serialization**: Added validation for JSON serializability in the final response to prevent formatting errors.
- **Error Handling**: Enhanced error handling in tool calls, metadata fetching, and default source fetching with better logging and fallbacks.

### Known Issues
- **Irrelevant Search Results**: The research retriever may return results unrelated to the query (e.g., searching for haplotypes in Pyrenees returns results about search engines). This suggests a bug in the `research_ret` module where the query is not being passed correctly or the search is being overridden.
- **Tool Call Parsing**: The parser may fail to extract tool calls from LLM output, leading to default searches that may not yield relevant results.
- **Retriever Import Failures**: If `src.langraph_models.research_ret` cannot be imported, the model falls back to limited functionality without provenance.
- **Metadata Fetching**: URL metadata fetching may fail due to network issues, timeouts, or invalid URLs, affecting provenance enrichment.

### Troubleshooting
- Check `logs/lats_model.log` for detailed error messages and query handling.
- Verify that the query is correctly passed to the retriever by inspecting agent_state in `call_research_tool`.
- Test the retriever manually: `python -c "from src.langraph_models import research_ret; print(research_ret.get_tool_node({'query': 'test query'}, 'google', 'search_google_detailed'))"`
- Ensure all dependencies are installed and the virtual environment is activated.

### Future Improvements
- Implement query validation and sanitization to prevent irrelevant searches.
- Add retry mechanisms for failed tool calls and network requests.
- Enhance provenance extraction to better handle different retriever response formats.
- Add configuration options for search parameters (e.g., number of results, search engines to use).
