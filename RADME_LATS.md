RADME_LATS

Overview

RADME_LATS documents the LATS model implementation contained in `src/langraph_models/lats_model.py`.
This document explains the model structure, runtime flow, how to generate a visual graph of the state machine, where outputs are saved, and quick troubleshooting steps.

High-level architecture

- Core idea: LATS runs a tree-search style planning process where candidate responses are generated, then critiqued (reflected) and re-expanded until a satisfactory solution is found.
- Key types:
  - TreeState: Pydantic model carrying the original input and a `root` Node representing the search tree.
  - Node: In-memory object representing a candidate branch. Contains messages, reflection, metadata, children, and scoring state.
  - Reflection: Pydantic model representing the critique/evaluation of a candidate (score 0-10, reflections text, provenance sources).

Main chains and flows

1. generate_initial_response(state: TreeState) -> dict
   - Normalizes the incoming state to a dict.
   - Calls the initial answer LLM chain (guarded so import-time runs can be dry-run) to produce the first candidate(s).
   - Parses any tool calls; if none are parsed and the internal research retriever is available it runs default searches (Google, Wikipedia, DDG) to gather provenance.
   - Produces a `root` Node with the initial candidate(s) and a reflection attached.
   - Returns an updated plain-dict state with `root` and a small `provenance` summary.

2. expand(state: TreeState, config) -> dict
   - Selects a leaf node to expand using an upper-confidence bound (UCT)-style heuristic.
   - Generates N new candidate messages (via `generate_candidates`) and optionally executes parsed research tool calls.
   - Runs a batch reflection chain to evaluate each candidate and attaches children nodes to the selected leaf.
   - Returns the mutated state (with an updated `root`).

3. Reflection chain
   - Uses a ChatPromptTemplate that forces responses to English (module enforces English system prompts).
   - Reflection outputs are parsed into the `Reflection` Pydantic model which includes score and optional provenance sources.

StateGraph and execution

- A lightweight StateGraph builder composes the nodes `start` and `expand` and uses a `should_loop` decision function to determine whether to continue expanding or finish.
- A utility `run_query(question: str)` is available to run the compiled graph for a single question and write outputs.
- The module can be executed as a script (CLI) to run a single question interactively.

Where outputs and logs go

- Logs: `logs/lats_model.log` (repository-root relative)
- Outputs: By default outputs are written to `src/langraph_models/output/` (this is where the project historically saved outputs). The module also defines `DEFAULT_OUTPUT_DIR` = `<repo-root>/output` and ensures the `run_query` helper writes outputs there when possible; the internal script `scripts/generate_graph.py` sets an environment variable to avoid LLM side-effects at import time and writes outputs under `src/langraph_models/output/`.

Files produced by the run

- `schema.txt`: textual final candidate content
- `solution.json`: JSON with question, final_info and best_trajectory
- `best_trajectory.txt`: human-readable best-trajectory messages

How to generate and view the graph (PowerShell)

1) Activate the virtual environment (PowerShell):

```powershell
& 'C:\Users\franc\PycharmProjects\ai_utilities\.venv\Scripts\Activate.ps1'
```

2) Generate the graph PNG (this script imports `lats_model` with side-effects disabled):

```powershell
python .\scripts\generate_graph.py
```

- Result: the helper attempts to compile the state graph and will log where outputs were saved. If the `langgraph` rendering backend is available, it will save a `graph.png` into `output/` or `src/langraph_models/output/` depending on the environment.

3) View the PNG (from PowerShell / Explorer):
- Open the file path reported in the script output or logs (for example: `src/langraph_models/output/`).

View in Jupyter / IPython

If you prefer to view the graph inline inside a Jupyter notebook, you can use:

```python
from IPython.display import Image, display
# path to the generated PNG
display(Image(filename='src/langraph_models/output/graph.png'))
```

Notes and troubleshooting

- Package requirements: this module depends on `langchain`, `langchain-core`, `langgraph`, and other packages (see `requirements.txt` and `pyproject.toml`). If the graph rendering or LLM chains fail to import, install the project's dependencies in your virtualenv.

- If `generate_graph.py` runs but you don't see `graph.png`:
  - Check `src/langraph_models/logs/lats_model.log` for errors.
  - Graph rendering requires `langgraph` and its rendering backends; if missing the module will still compile the graph logic but won't be able to draw PNGs.

- The repository contains a script `scripts/generate_graph.py` which sets `LATS_NO_SIDE_EFFECTS=1` to avoid calling actual LLMs during graph generation. That script is the recommended way to create the diagram without consuming API credits.

- We forced system prompts to instruct the LLM to reply in English; if you still see content in other languages, verify the LLM model and API behavior and retry.

Developer notes (what I changed)

- `src/langraph_models/lats_model.py`:
  - Ensured system prompts ask for English replies.
  - Fixed a previously misplaced output-saving block and indentation errors.
  - Restored/added `should_loop` helper used by the graph builder.
  - Added a `run_query` helper and a simple CLI so the model can be executed as a script and save outputs.
  - Added a safe fallback around `langgraph.graph` imports so import-time tooling doesn't crash in environments missing that package.

Next steps (optional)

- If you want the graph PNG to always be written to the repository root `output/` folder, we can change the module to prefer `DEFAULT_OUTPUT_DIR` unconditionally (currently both `src/langraph_models/output/` and `output/` are used depending on how the module is run). I avoided a risky global change here and instead added `run_query` which writes to `DEFAULT_OUTPUT_DIR`.

If you'd like, I can now:
- Update the module to always write to `<repo-root>/output/` (safe small change).
- Create a minimal example notebook that imports and displays the graph inline.


