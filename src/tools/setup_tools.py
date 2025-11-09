from src.tools.registry.shared_registry import agent_tools
from src.tools.registry.external_tools.register_search_tools import register_search_tools , \
    search_tool_functions



# Aquí añades todos los que quieras
def setup_all_tools():

    register_search_tools ( agent_tools_catalogue = agent_tools , search_func_catalogue = search_tool_functions )

    # Validación rápida
    print ( "🔧 Registerd tools:" , list ( agent_tools.keys ( ) ) )
