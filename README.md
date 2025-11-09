markdown
# 🧠 Multi-Source Research Agent with LangGraph

## Overview

This project implements a robust, multi-source research agent using [LangGraph](https://github.com/langchain-ai/langgraph), a graph-based orchestration framework for language agents. It refines user queries using a large language model (LLM), executes parallel searches across multiple online sources, filters irrelevant results, and synthesizes a final report.

---

## 🔍 Features

- **Query Refinement**: Uses DeepSeek to translate and optimize user queries for search APIs.
- **Multi-Source Search**: Integrates tools for:
  - Google
  - DuckDuckGo
  - Wikipedia
  - News
  - Arxiv
  - PubMed
  - YouTube
  - Bing
  - Stack Overflow
  - GitHub
- **Error Handling & Retry**: Automatically retries failed sources up to `MAX_RETRIES`.
- **LLM-Based Pruning**: Filters noisy or irrelevant results using a custom prompt and DeepSeek.
- **Synthesis**: Generates a structured summary of findings across all sources.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/research-agent
cd research-agent
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
Make sure to include your .env file with the following:

env
OPENAI_API_KEY=your_openai_key
EMAIL=your_email_for_entrez
🚀 Usage
Run the agent with a custom query:

bash
python src/main.py
Or modify the question variable in main.py:

python
question = "what are R1b haplogroups?"
🧩 Architecture
mermaid
graph TD
    START --> refine_query
    refine_query --> google
    refine_query --> ddg
    refine_query --> wikipedia
    refine_query --> news
    refine_query --> arxiv
    refine_query --> pubmed
    refine_query --> youtube
    refine_query --> bing
    refine_query --> stackoverflow
    refine_query --> github
    google --> verify_retry
    ddg --> verify_retry
    wikipedia --> verify_retry
    news --> verify_retry
    arxiv --> verify_retry
    pubmed --> verify_retry
    youtube --> verify_retry
    bing --> verify_retry
    stackoverflow --> verify_retry
    github --> verify_retry
    verify_retry --> prune_results
    prune_results --> synthesis
    synthesis --> END
🛠️ Tool Registry & Catalog
All tools are loaded via setup_all_tools() and stored in agent_tools. The catalog is built dynamically and includes consistency checks.

🔧 Catalog Construction
python
from src.tools.setup_tools import setup_all_tools
setup_all_tools()

from src.tools.registry.shared_registry import agent_tools

tool_catalog = {
    agent_name.replace("_agent", ""): tools
    for agent_name, tools in agent_tools.items()
}
🧮 Summary Report
Total tools registered: sum(len(tools) for tools in agent_tools.values())

Agents registered: len(agent_tools)

Tools per agent: Listed with name and description

📋 Example Output
Codi
🧮 Total registered tools: 17

📋 REGISTERED AGENTS REPORT
========================================
🔢 Total registered agents: 5

🧠 search_agent: 11 tool(s)
   └─ 🛠️ search_google_detailed — Google search with structured output
   └─ 🛠️ DDGGeneralSearch — DuckDuckGo general search
   └─ 🛠️ WikipediaStructuredSearch — Wikipedia API wrapper
   └─ 🛠️ ArxivRawQuery — Arxiv scientific paper search
   └─ 🛠️ PubMedSearchTool — PubMed biomedical search
   └─ 🛠️ YouTubeSerpAPISearch — YouTube video search
   └─ 🛠️ BingSearchTool — Bing search engine wrapper
   └─ 🛠️ StackOverflowSearchTool — Stack Overflow Q&A search
   └─ 🛠️ GithubDomainSearch — GitHub repository search
   └─ 🛠️ DDGNewsSearch — DuckDuckGo news search
   └─ 🛠️ BraveSearchTool — Brave search engine wrapper
🚨 Consistency Check
Verifies that all agents are correctly cataloged:

python
registered = set(agent_tools.keys())
catalogued = set(tool_catalog.keys())
missing = registered - {key + "_agent" for key in catalogued}
If any are missing, they are printed as:

Codi
⚠️ AGENTS WITHOUT CATALOG ENTRY
❌ some_agent
Otherwise:

Codi
✅ All agents are correctly cataloged.
✅ Example Output
markdown
## Research Results Synthesis

**Original Query:** what are R1b haplogroups?
**Optimized Query (Deepseek):** R1b haplogroup genetic ancestry

### Google
R1b is a major Y-DNA haplogroup found in Western Europe...

### PubMed
**Title:** Genetic structure of R1b lineages in Europe
**Date:** 2021
**Link:** https://pubmed.ncbi.nlm.nih.gov/12345678/

...

🧠 Full synthesis generated. End of process.
📄 License
MIT License. See LICENSE file for details.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

🙋‍♂️ Author
Developed by Francesc Sánchez Parés