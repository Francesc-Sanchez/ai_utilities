from src.tools.setup_tools import setup_all_tools
# Centralized call
setup_all_tools ()
# Execute logging functions
import logging

from src.tools.registry.shared_registry import agent_tools


# Build catalog
tool_catalog = {
    agent_name.replace("_agent", ""): tools
    for agent_name, tools in agent_tools.items()
}
# 🧮 Total registered tools
total_tools = sum(len(tools) for tools in agent_tools.values())
print(f"\n🧮 Total registered tools: {total_tools}")

# 🔍 Registered Agents Report
print("\n📋 REGISTERED AGENTS REPORT")
print("=" * 40)
print(f"🔢 Total registered agents: {len(agent_tools)}\n")

for agent_name, tools in agent_tools.items():
    print(f"🧠 {agent_name}: {len(tools)} tool(s)")
    for tool in tools:
        print(f"   └─ 🛠️ {tool.name} — {tool.description}")
    print()

# 📦 Catalog Report
print("\n📦 TOOL CATALOG BY AGENT")
print("=" * 40)
for agent_key, tools in tool_catalog.items():
    print(f"📁 {agent_key}: {len(tools)} tool(s)")
    for tool in tools:
        print(f"   └─ 🔧 {tool.name}")
    print()

# 🚨 Consistency Verification
registered = set(agent_tools.keys())
catalogued = set(tool_catalog.keys())
missing = registered - {key + "_agent" for key in catalogued}

if missing:
    print("\n⚠️ AGENTS WITHOUT CATALOG ENTRY")
    print("=" * 40)
    for m in missing:
        print(f"❌ {m}")
else:
    print("\n✅ All agents are correctly cataloged.")