import html
import logging
import operator
import re
from typing import Optional , List , Dict , Union , Any , Tuple
from typing import TypedDict

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langgraph.constants import START , END
from langgraph.graph import StateGraph
from typing_extensions import Annotated


from src.tools.setup_tools import setup_all_tools

from src.core.load_ret_llm_models import get_model

setup_all_tools ( )
from src.tools.registry.shared_registry import agent_tools

# ----------------------------------------------------------------------

# --- CONFIGURACIÓN Y CONSTANTES ---
logging.basicConfig ( level = logging.INFO , format = '%(asctime)s - %(levelname)s - %(message)s' )
MAX_RETRIES = 2  # Intentos máximos por herramienta
DEEPSEEK_MODEL = get_model ( "Deepseek" )  # Cliente LLM para optimización y filtrado

# NUEVO: umbral mínimo configurable para contenido de herramienta
MIN_CONTENT_LENGTH = 100  # antes 200, bajado para reducir reintentos innecesarios

# PROMPT para optimizar la query a inglés (siempre necesario para la mejor búsqueda)
SYSTEM_PROMPT = (
    "You are a highly efficient search query optimizer and translator. "
    "Given the user's question, you MUST translate it into English and "
    "then provide a single, concise string of English keywords and key phrases (2-8 words maximum) "
    "suitable for a PubMed, Arxiv, or Google search API. "
    "Your output MUST be ONLY the raw, OPTIMIZED SEARCH STRING IN ENGLISH. "
    "DO NOT include any filler words, explanations, or sentences in the final output."
)

# Spanish variant: when the user's query is in Spanish we want the refined query
# to be produced in Spanish (so web searches use the original language). This
# avoids forcing English-language search terms for web search tools.
SYSTEM_PROMPT_ES = (
    "You are a concise search query optimizer for Spanish-language web searches. "
    "Given the user's question in Spanish, produce a single, concise search string in Spanish (2-8 words/phrases) that will maximize relevant Spanish-language web results. "
    "Return ONLY the raw optimized SEARCH STRING IN SPANISH, without any explanation or extra text."
)

# 🚨 PRUNING PROMPT: Clasifica por idioma ORIGINAL de la página (español vs inglés)
PRUNING_PROMPT_TEMPLATE = (
    "You are a bilingual content filter and language-detection assistant. Analyze the raw search results below and select the most relevant snippets for the user's query: '{query}'.\n"
    "CRITICAL: Determine the ORIGINAL LANGUAGE in which each source page or video was produced (NOT the language of any translation or subtitles). Use cues such as the webpage title, the textual snippet, metadata, or explicit language tags when available.\n"
    "Rules (apply strictly):\n"
    "1) Produce exactly two sections in this order:\n"
    "   A) **Resultados en Español:** — Include ONLY items whose ORIGINAL PAGE LANGUAGE is Spanish.\n"
    "   B) **Results in English:** — Include ONLY items whose ORIGINAL PAGE LANGUAGE is English.\n"
    "2) For each item include: Title (as in the raw result), a 1-2 line concise summary (in the same language as the section), and the exact URL.\n"
    "3) If for a given raw snippet you cannot confidently determine the original language, exclude it from both sections and list it under a short appendix titled 'Undetermined language (excluded)'.\n"
    "4) Aim to select up to 3 items per section (max 6 total). Prefer primary sources and pages whose original language clearly matches the section.\n"
    "5) DO NOT translate titles or snippets; preserve them exactly as provided in the raw results.\n\n"
    "If there are no qualifying items for a section, write: 'No hay resultados disponibles en Español.' or 'No hay resultados disponibles en inglés.' respectively (use the Spanish wording for the English-empty case to match the user's language).\n\n"
    "Raw Results from {source_name}:\n{raw_content}"
)

# PROMPT para la síntesis final
SYNTHESIS_PROMPT_TEMPLATE = (
    "You are a world-class research analyst. Your task is to analyze the following filtered search results "
    "and provide a structured final report to the user's original query: '{query}'.\n\n"
    "CRITICAL INSTRUCTION:\n"
    "1. **RESULTADOS FILTRADOS POR FUENTE:** First, present the content of the search results below, grouping them by source (### Fuente: [Name]). Ensure the original titles, links, and the summarized snippets are clearly visible for each result, preserving the bilingual structure provided.\n"
    "2. **SÍNTESIS FINAL:** After the structured results, provide a single, comprehensive, and cohesive answer to the query: '{query}'. This final answer must be a separate section titled '## Síntesis Final'.\n"
    "3. **LENGUAJE:** The entire final response MUST be written in the original language of the user's query (Spanish, Catalan, or other detected language).\n\n"
    "Sources to be structured and synthesized:\n{sources_content}"
)

# --- Heurística ligera de detección de idioma (fallback local) ---
SPANISH_KEYWORDS = {"el" , "la" , "que" , "de" , "y" , "en" , "para" , "con" , "no" , "por" , "como" , "su" , "se" ,
                    "es"}
ENGLISH_KEYWORDS = {"the" , "and" , "of" , "to" , "in" , "for" , "with" , "is" , "that" , "as" , "it" , "on" , "by"}


def detect_language(text: str) -> str:
    """Detect language prioritizing `langdetect` when available.

    Returns:
      - 'es' for Spanish
      - 'en' for English
      - 'undetermined' otherwise
    """
    # Guard clauses
    if not text or not text.strip ( ):
        return "undetermined"

    # Try to use langdetect if installed
    try:
        from langdetect import detect_langs

        try:
            langs = detect_langs ( text )
            if not langs:
                return "undetermined"
            top = langs[ 0 ]
            lang_code = top.lang
            prob = top.prob
            # Require reasonable confidence
            if prob >= 0.70:
                if lang_code.startswith ( "es" ):
                    return "es"
                if lang_code.startswith ( "en" ):
                    return "en"
                return "undetermined"
            # If low confidence, fall through to heuristic
        except Exception:
            # If langdetect fails on this text, fall back to heuristics below
            pass
    except Exception:
        # langdetect not installed — we'll use heuristic fallback
        pass

    # Heuristic fallback (simple keyword counts)
    words = re.findall ( r"\w+" , text.lower ( ) )
    if not words:
        return "undetermined"
    es_count = sum ( 1 for w in words if w in SPANISH_KEYWORDS )
    en_count = sum ( 1 for w in words if w in ENGLISH_KEYWORDS )
    if es_count == en_count:
        return "undetermined"
    return "es" if es_count > en_count else "en"


def _extract_first_url(text: str) -> str:
    m = re.search ( r"https?://[^\s)\]]+" , text )
    return m.group ( 0 ) if m else ""


def extract_urls_from_text(text: str) -> List[ str ]:
    """Extrae URLs del texto en orden de aparición."""
    return re.findall ( r"https?://[^\s)\]]+" , text )


def detect_language_of_url(url: str , timeout: int = 6) -> str:
    """Intenta obtener la página y detectar su idioma.

    Estrategia:
      1. Hacer GET con headers y timeout.
      2. Si hay <html lang='..'> usarlo.
      3. Si no, extraer texto del <body> y usar langdetect (si existe) o heurística.
      4. Devolver 'es','en' o 'undetermined'.
    """
    try:
        import requests
    except Exception:
        return "undetermined"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.org)"}
    try:
        resp = requests.get ( url , headers = headers , timeout = timeout )
        if resp.status_code != 200 or not resp.content:
            return "undetermined"
        content = resp.content
        soup = BeautifulSoup ( content , "html.parser" )

        # 1. <html lang="...">
        html_tag = soup.find ( "html" )
        if html_tag and html_tag.get ( "lang" ):
            lang_attr = html_tag.get ( "lang" ).lower ( )
            if lang_attr.startswith ( "es" ):
                return "es"
            if lang_attr.startswith ( "en" ):
                return "en"

        # 2. <meta http-equiv="content-language"> or meta[name=language]
        meta_lang = soup.find ( "meta" , attrs = {"http-equiv": lambda v: v and v.lower ( ) == "content-language"} )
        if meta_lang and meta_lang.get ( "content" ):
            ml = meta_lang.get ( "content" ).lower ( )
            if ml.startswith ( "es" ):
                return "es"
            if ml.startswith ( "en" ):
                return "en"

        meta_name = soup.find ( "meta" , attrs = {"name": lambda v: v and v.lower ( ) == "language"} )
        if meta_name and meta_name.get ( "content" ):
            ml = meta_name.get ( "content" ).lower ( )
            if "es" in ml:
                return "es"
            if "en" in ml:
                return "en"

        # 3. Extraer texto y usar detect_language()
        texts = " ".join ( [ t.get_text ( separator = " " ) for t in soup.find_all ( [ "p" , "h1" , "h2" , "h3" ] ) ] )
        texts = html.unescape ( texts )[ :2000 ]
        lang = detect_language ( texts )
        return lang

    except Exception:
        return "undetermined"


def fetch_page_info(url: str , timeout: int = 6) -> Tuple[ str , str , str ]:
    """Fetch page and return (lang, title, snippet). Snippet is first paragraph text.
    Returns ('undetermined','','') on failure.
    """
    try:
        import requests
    except Exception:
        return ("undetermined" , "" , "")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.org)"}
    try:
        resp = requests.get ( url , headers = headers , timeout = timeout )
        if resp.status_code != 200 or not resp.content:
            return ("undetermined" , "" , "")
        soup = BeautifulSoup ( resp.content , "html.parser" )
        # title
        title_tag = soup.find ( "title" )
        title = title_tag.get_text ( ).strip ( ) if title_tag else url
        # first paragraph
        p = soup.find ( "p" )
        snippet = p.get_text ( ).strip ( )[ :400 ] if p else ""
        # detect language using html lang or meta or extracted text
        lang = "undetermined"
        html_tag = soup.find ( "html" )
        if html_tag and html_tag.get ( "lang" ):
            la = html_tag.get ( "lang" ).lower ( )
            if la.startswith ( "es" ):
                lang = "es"
            elif la.startswith ( "en" ):
                lang = "en"
        if lang == "undetermined":
            meta_lang = soup.find ( "meta" , attrs = {"http-equiv": lambda v: v and v.lower ( ) == "content-language"} )
            if meta_lang and meta_lang.get ( "content" ):
                ml = meta_lang.get ( "content" ).lower ( )
                if ml.startswith ( "es" ):
                    lang = "es"
                elif ml.startswith ( "en" ):
                    lang = "en"
        if lang == "undetermined":
            lang = detect_language ( (snippet or title)[ :2000 ] )
        return (lang , title , snippet)
    except Exception:
        return ("undetermined" , "" , "")


def fallback_prune(raw_content: str , query: str) -> str:
    """Produce a structured pruning result (Spanish/English sections) using simple heuristics.
    This is used when the LLM is unavailable or returns an unexpected format.
    """
    items = [ it.strip ( ) for it in re.split ( r"\n\s*\n" , raw_content.strip ( ) ) if it.strip ( ) ]
    spanish_items = [ ]
    english_items = [ ]
    undetermined = [ ]

    for it in items:
        lang = detect_language ( it )
        title = it.splitlines ( )[ 0 ][ :200 ]
        summary = (" ".join ( it.splitlines ( )[ 1: ] ) or it)[ :300 ]
        url = _extract_first_url ( it )
        entry = f"- Title: {title}\n  Summary: {summary}\n  URL: {url if url else 'No URL found'}"
        if lang == "es" and len ( spanish_items ) < 3:
            spanish_items.append ( entry )
        elif lang == "en" and len ( english_items ) < 3:
            english_items.append ( entry )
        else:
            undetermined.append ( entry )

    parts = [ ]
    if spanish_items:
        parts.append ( "**Resultados en Español:**\n" + "\n\n".join ( spanish_items ) )
    else:
        parts.append ( "No hay resultados disponibles en Español." )

    # Localize the 'no results in English' message based on the original query
    qlang = detect_language ( query or "" )
    no_en_msg = "No results available in English." if qlang != 'es' else "No hay resultados disponibles en inglés."

    if english_items:
        parts.append ( "**Results in English:**\n" + "\n\n".join ( english_items ) )
    else:
        parts.append ( no_en_msg )

    if undetermined:
        parts.append ( "Undetermined language (excluded):\n" + "\n\n".join ( undetermined[ :10 ] ) )

    return "\n\n".join ( parts )


# 💡 FUNCIÓN DE FUSIÓN
def merge_dicts(left: Dict , right: Dict) -> Dict:
    return left | right


# --- DEFINICIÓN DEL ESTADO (AgentState) ---
class AgentState ( TypedDict , total = False ):
    query: Annotated[ str , operator.concat ]
    refined_query: Optional[ str ]

    # RESULTADOS BRUTOS (SE USAN PARA EL PRUNING)
    google_result: Annotated[ Optional[ str ] , operator.concat ]
    ddg_result: Annotated[ Optional[ str ] , operator.concat ]
    wikipedia_result: Annotated[ Optional[ str ] , operator.concat ]
    news_result: Annotated[ Optional[ str ] , operator.concat ]
    arxiv_result: Annotated[ Optional[ str ] , operator.concat ]
    pubmed_result: Annotated[ Optional[ str ] , operator.concat ]
    youtube_result: Annotated[ Optional[ str ] , operator.concat ]
    bing_result: Annotated[ Optional[ str ] , operator.concat ]
    stackoverflow_result: Annotated[ Optional[ str ] , operator.concat ]
    github_result: Annotated[ Optional[ str ] , operator.concat ]

    # 🚨 RESULTADOS FILTRADOS (SE USAN PARA LA SÍNTESIS)
    google_pruned: Annotated[ Optional[ str ] , operator.concat ]
    ddg_pruned: Annotated[ Optional[ str ] , operator.concat ]
    wikipedia_pruned: Annotated[ Optional[ str ] , operator.concat ]
    news_pruned: Annotated[ Optional[ str ] , operator.concat ]
    arxiv_pruned: Annotated[ Optional[ str ] , operator.concat ]
    pubmed_pruned: Annotated[ Optional[ str ] , operator.concat ]
    youtube_pruned: Annotated[ Optional[ str ] , operator.concat ]
    bing_pruned: Annotated[ Optional[ str ] , operator.concat ]
    stackoverflow_pruned: Annotated[ Optional[ str ] , operator.concat ]
    github_pruned: Annotated[ Optional[ str ] , operator.concat ]

    synthesis: Annotated[ Optional[ str ] , operator.concat ]
    retry_count: int
    failed_nodes: Annotated[ List[ str ] , operator.add ]
    node_ready: Annotated[ Dict[ str , bool ] , merge_dicts ]


# --- 🧠 FUNCIÓN DE REFINAMIENTO CON DEEPSEEK ---
def llm_refine_query(query: str) -> str:
    """Refine the user's query into a short search string.

    If the user's query is Spanish, ask the LLM to return an optimized SPANISH search string.
    Otherwise, use the English-optimization SYSTEM_PROMPT.
    """
    try:
        lang = detect_language ( query )
    except Exception:
        lang = "undetermined"

    prompt = SYSTEM_PROMPT_ES if lang == "es" else SYSTEM_PROMPT

    try:
        messages = [ ("system" , prompt) , ("human" , query) ]
        response = DEEPSEEK_MODEL.invoke ( messages )
        refined_query = response.content.strip ( )
    except Exception as e:
        logging.error ( f"    [LLM Refine] Fallo detectado ({e}). Usando fallback de traducción/manual." )
        # Fallback de seguridad: simple passthrough (trim) so searches still run
        refined_query = query.strip ( )[ :200 ]
    return refined_query


# --- NODOS BÁSICOS (Mantenidos) ---
def refine_query_node(state: AgentState) -> Dict[ str , Any ]:
    try:
        refined_query = llm_refine_query ( state[ "query" ] )
        return {"refined_query": refined_query}
    except Exception as e:
        return {"refined_query": state.get ( "query" , "" )}


def limitar_enlaces(texto: str , max_links: int = 5) -> str:
    # Función para limitar el tamaño del texto para evitar problemas con el LLM de poda
    if len ( texto ) > 2000:
        return texto[ :2000 ] + "\n... (Resultado recortado para brevedad) ..."
    return texto


def get_tool_from_agent_tools(group_key: str , tool_name: str , query: str) -> Any:
    tools_list = agent_tools.get ( group_key , [ ] )
    try:
        # Usamos .invoke() en lugar de __call__ para evitar la advertencia de depreciación
        tool_object = next ( t for t in tools_list if t.name == tool_name )
        return tool_object.invoke ( {"query": query} )
    except StopIteration:
        # No lanzamos una excepción cruda: devolvemos un mensaje de error claro
        logging.error ( f"Herramienta no encontrada: {tool_name} en el grupo {group_key}." )
        return f"Tool execution failed: Tool not found: {tool_name} in group {group_key}."
    except Exception as e:
        # Capturamos errores de ejecución de la herramienta y devolvemos texto informativo
        logging.error ( f"Fallo al ejecutar {tool_name}: {e}" )
        return f"Tool execution failed: {str ( e )}"


def get_tool_node(state: AgentState , node_name: str , tool_name: str) -> Dict[ str , Any ]:
    group_key = "search_agent"
    search_query = state.get ( "refined_query" , state[ "query" ] )
    try:
        # 1. Ejecutar la herramienta
        logging.info ( f"    [Node: {node_name}] Ejecutando tool '{tool_name}' con query: '{search_query[ :50 ]}...'" )

        # Si la consulta original está en español y estamos llamando a Google, intentamos
        # una búsqueda dirigida en español primero para recuperar páginas originalmente en español.
        try:
            is_query_spanish = detect_language ( state.get ( "query" , "" ) ) == "es"
        except Exception:
            is_query_spanish = False

        # Ensure english_text is always defined (used later when composing labeled blocks)
        english_text = ''

        if node_name == "google" and is_query_spanish:
            # Ejecutar búsqueda dirigida en español como primer intento
            spanish_query = state.get ( "query" , "" ) + " lang:es"
            extra_result = get_tool_from_agent_tools ( group_key , tool_name , spanish_query )
            # También generar y ejecutar una búsqueda dirigida a contenido en INGLÉS
            try:
                # Forzamos refinement en inglés usando SYSTEM_PROMPT
                messages_en = [ ("system" , SYSTEM_PROMPT) , ("human" , state.get ( "query" , "" )) ]
                resp_en = DEEPSEEK_MODEL.invoke ( messages_en )
                english_refined = resp_en.content.strip ( )
            except Exception:
                english_refined = search_query

            english_result = get_tool_from_agent_tools ( group_key , tool_name , english_refined )
            # Ejecutar búsqueda general también (fallback)
            result = get_tool_from_agent_tools ( group_key , tool_name , search_query )

            # Normalizar extra_result a texto
            if isinstance ( extra_result , dict ) and 'output' in extra_result:
                extra_text = extra_result[ 'output' ]
            elif isinstance ( extra_result , str ):
                extra_text = extra_result
            else:
                extra_text = ''

            # Normalizar english_result a texto y, si existe, preprenderlo al bloque general
            if isinstance ( english_result , dict ) and 'output' in english_result:
                english_text = english_result[ 'output' ]
            elif isinstance ( english_result , str ):
                english_text = english_result
            else:
                english_text = ''
        else:
            result = get_tool_from_agent_tools ( group_key , tool_name , search_query )
            extra_text = ''

        # 2. Normalizar el resultado a texto
        if isinstance ( result , dict ) and 'output' in result:
            texto = result[ 'output' ]
        elif isinstance ( result , str ):
            texto = result
        else:
            # Caso de resultado inesperado (ej. None, objeto complejo no serializado)
            texto = f"Resultado inesperado de la herramienta {tool_name} (Tipo: {type ( result )}). No se pudo extraer contenido."

        # Si ejecutamos búsquedas dirigidas en español y/o inglés, insertamos bloques
        # etiquetados para que el LLM pueda distinguir claramente las fuentes por idioma.
        # Condición: si hay contenido suficiente en cualquiera de los bloques dirigidos.
        if (extra_text and len ( extra_text ) > MIN_CONTENT_LENGTH) or (
                english_text and len ( english_text ) > MIN_CONTENT_LENGTH):
            parts = [ ]
            if extra_text and len ( extra_text ) > MIN_CONTENT_LENGTH:
                parts.append ( "Raw Results (Spanish-targeted search):\n" + extra_text )
            if english_text and len ( english_text ) > MIN_CONTENT_LENGTH:
                parts.append ( "Raw Results (English-targeted search):\n" + english_text )
            # Always include the general search as context
            parts.append (
                "Raw Results (General search):\n" + (texto if isinstance ( texto , str ) else str ( texto )) )
            texto = "\n\n".join ( parts )

        texto_limitado = limitar_enlaces ( texto )

        # 3. VERIFICACIÓN DE CONTENIDO MÍNIMO Y ERRORES COMUNES (MECANISMO DE RESILIENCIA)
        # Ahora diferenciamos entre error real y contenido corto para no lanzar excepciones innecesarias.

        # Check A: Error explícito retornado por la herramienta (ej. 'ToolX returned error...')
        if any (
                err_str in texto_limitado.lower ( ) for err_str in
                [ "error" , "failed" , "timeout" , "not found" , "dns" , "host desconocido" ] ):
            # Devolvemos un resultado con estado de fallo, pero sin lanzar excepción que interrumpa el flujo
            logging.error (
                f"    [Node: {node_name}] La herramienta devolvió un mensaje de error o fallo de red: {texto_limitado[ :200 ]}" )
            return {
                f"{node_name}_result": texto_limitado ,
                "node_ready": {node_name: False} ,
                "failed_nodes": [ node_name ]
                }

        # Check B: Contenido de herramienta demasiado corto (umbral ahora configurable)
        if len ( texto_limitado ) < MIN_CONTENT_LENGTH:
            logging.warning (
                f"    [Node: {node_name}] El contenido de la herramienta es corto (Longitud: {len ( texto_limitado )}) pero no parece un error crítico. Se marcará como no listo para posible reintento." )
            return {
                f"{node_name}_result": texto_limitado ,
                "node_ready": {node_name: False} ,
                "failed_nodes": [ node_name ]
                }

        logging.info ( f"    [Node: {node_name}] Ejecución exitosa. Contenido de longitud {len ( texto_limitado )}." )

        return {
            f"{node_name}_result": texto_limitado ,
            "node_ready": {node_name: True} ,
            "failed_nodes": [ ]
            }

    except Exception as e:
        # 4. Manejo de Fallos (incluyendo fallos forzados por contenido corto)
        texto = f"Error en {node_name.capitalize ( )} ({tool_name}): {str ( e )}"
        logging.error ( f"    [Node: {node_name}] Fallo detectado: {e}" )
        return {
            f"{node_name}_result": texto ,
            "node_ready": {node_name: False} ,
            # Sobreescribir failed_nodes con el nodo actual para un seguimiento preciso
            "failed_nodes": [ node_name ]
            }


# Wrappers existentes (Mantenidos, llaman a get_tool_node) — retornan dicts parciales
def google_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node (
    state , "google" , "search_google_detailed" )


def youtube_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node (
    state , "youtube" , "YouTubeSerpAPISearch" )


def ddg_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node ( state , "ddg" , "DDGGeneralSearch" )


def wikipedia_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node (
    state , "wikipedia" , "WikipediaStructuredSearch" )


def news_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node ( state , "news" , "DDGNewsSearch" )


def arxiv_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node ( state , "arxiv" , "ArxivRawQuery" )


def pubmed_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node ( state , "pubmed" , "PubMedSearchTool" )


def bing_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node ( state , "bing" , "BingSearchTool" )


def stackoverflow_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node (
    state , "stackoverflow" , "StackOverflowSearchTool" )


def github_node(state: AgentState) -> Dict[ str , Any ]: return get_tool_node (
    state , "github" , "GithubDomainSearch" )


# --- NODO DE FILTRACIÓN (PRUNING) (Mantenido) ---
def prune_results_node(state: AgentState) -> Dict[ str , Any ]:
    logging.info ( "    [Node: Pruning] Iniciando filtración de resultados irrelevantes." )

    pruned_state = {}
    source_keys = [
        ("google" , "google_result" , "google_pruned") , ("ddg" , "ddg_result" , "ddg_pruned") ,
        ("wikipedia" , "wikipedia_result" , "wikipedia_pruned") , ("news" , "news_result" , "news_pruned") ,
        ("arxiv" , "arxiv_result" , "arxiv_pruned") , ("pubmed" , "pubmed_result" , "pubmed_pruned") ,
        ("youtube" , "youtube_result" , "youtube_pruned") , ("bing" , "bing_result" , "bing_pruned") ,
        ("stackoverflow" , "stackoverflow_result" , "stackoverflow_pruned") ,
        ("github" , "github_result" , "github_pruned")
        ]

    for node_name , result_key , pruned_key in source_keys:
        raw_content = state.get ( result_key )

        # Si raw_content es None o parece contener un mensaje de error, lo propagamos sin filtrar.
        if not raw_content or any (
                err in (raw_content or "").lower ( ) for err in [
                    "error" , "no se pudo" , "tool execution failed" , "tool execution" , "failed" , "timeout" ,
                    "host desconocido" , "dns" ] ):
            pruned_state[ pruned_key ] = raw_content
            continue

        try:
            # If we injected labeled spanish-targeted + general raw results earlier in get_tool_node,
            # handle them deterministically here: run fallback_prune on each block and merge.
            if raw_content and "Raw Results (Spanish-targeted search):" in raw_content and "Raw Results (General search):" in raw_content:
                try:
                    # split the labeled blocks
                    parts = raw_content.split ( "Raw Results (Spanish-targeted search):\n" , 1 )[ 1 ]
                    spanish_block , rest = parts.split ( "\n\nRaw Results (General search):\n" , 1 )
                except Exception:
                    # fallback to full LLM pruning if split fails
                    spanish_block = None
                    rest = None

                if spanish_block is not None and rest is not None:
                    # Use fallback_prune on each to extract their sections
                    spanish_pruned = fallback_prune ( spanish_block , state[ 'query' ] )
                    general_pruned = fallback_prune ( rest , state[ 'query' ] )

                    # Extract the Spanish section from spanish_pruned (or fallback to 'No hay...')
                    sp_section = "No hay resultados disponibles en Español."
                    en_section = "No results available in English."

                    if "**Resultados en Español:**" in spanish_pruned:
                        # take the Spanish section (up to Results in English if present)
                        try:
                            start = spanish_pruned.index ( "**Resultados en Español:**" )
                            end = spanish_pruned.find ( "**Results in English:**" , start )
                            sp_section = spanish_pruned[ start:end ].strip ( ) if end != -1 else spanish_pruned[
                                                                                                 start: ].strip ( )
                        except Exception:
                            sp_section = spanish_pruned

                    if "**Results in English:**" in general_pruned:
                        try:
                            start = general_pruned.index ( "**Results in English:**" )
                            end = len ( general_pruned )
                            en_section = general_pruned[ start:end ].strip ( )
                        except Exception:
                            en_section = general_pruned
                    else:
                        # Attempt to salvage English items by scanning URLs in the general block
                        try:
                            urls_general = extract_urls_from_text ( rest )
                            english_entries = [ ]
                            for u in urls_general:
                                if len ( english_entries ) >= 3:
                                    break
                                try:
                                    lang_u , title_u , snippet_u = fetch_page_info ( u )
                                    if lang_u == 'en':
                                        entry = f"- Title: {title_u}\n  Summary: {snippet_u if snippet_u else title_u}\n  URL: {u}"
                                        english_entries.append ( entry )
                                except Exception:
                                    continue
                            if english_entries:
                                en_section = "**Results in English:**\n" + "\n\n".join ( english_entries )
                        except Exception:
                            pass

                    pruned_content = sp_section + "\n\n" + en_section
                else:
                    # Couldn't split reliably: fall back to LLM
                    full_prompt = PRUNING_PROMPT_TEMPLATE.format (
                        query = state[ 'query' ] , source_name = node_name.capitalize ( ) , raw_content = raw_content
                        )
                    llm_response = DEEPSEEK_MODEL.invoke ( [ HumanMessage ( content = full_prompt ) ] )
                    pruned_content = llm_response.content.strip ( )
            else:
                full_prompt = PRUNING_PROMPT_TEMPLATE.format (
                    query = state[ 'query' ] , source_name = node_name.capitalize ( ) , raw_content = raw_content
                    )
                llm_response = DEEPSEEK_MODEL.invoke ( [ HumanMessage ( content = full_prompt ) ] )
                pruned_content = llm_response.content.strip ( )

            # Guardar el contenido podado para esta fuente (muy importante)
            try:
                pruned_state[ pruned_key ] = pruned_content
                logging.info (
                    f"    [Pruning] {node_name.capitalize ( )} pruned and stored (len={len ( pruned_content )})." )
            except Exception as _e:
                logging.error ( f"    [Pruning] Could not store pruned content for {node_name}: {_e}" )
        except Exception:
            logging.error ( f"    [Pruning] Fallo al filtrar {node_name}: {e}. Usando heurística local." )
            pruned_state[ pruned_key ] = fallback_prune ( raw_content , state[ 'query' ] )

    return pruned_state


# --- NODO DE VERIFICACIÓN Y REINTENTO (Ajustado para mejor claridad de estado) ---
def verification_retry_node(state: AgentState) -> Dict[ str , Union[ int , List[ str ] , Dict , str ] ]:
    # Recolectar nodos que FALLARON en el ÚLTIMO ciclo de herramientas
    current_failed_nodes = [ n for n , ready in state.get ( "node_ready" , {} ).items ( ) if not ready ]

    retry_count = state.get ( "retry_count" , 0 )

    if current_failed_nodes and retry_count < MAX_RETRIES:
        logging.info ( f"    [Verification] Detectados fallos: {current_failed_nodes}. Reintento #{retry_count + 1}." )

        # Limpiar 'node_ready' para que solo los nodos fallidos sean reejecutados en el siguiente ciclo.
        # Al regresar al nodo 'refine_query', los nodos que ya estaban en True no se reejecutan
        # automáticamente, pero la nueva lógica en 'get_tool_node' se asegurará de que
        # los fallidos se reintenten correctamente.
        new_node_ready = {k: v for k , v in state.get ( "node_ready" , {} ).items ( ) if v}

        return {"retry_count": retry_count + 1 , "failed_nodes": current_failed_nodes , "node_ready": new_node_ready}

    # Si no hay fallos o se agotaron los reintentos, pasamos a la síntesis.
    return {}


# --- NODO DE SÍNTESIS (FINAL) ---
def synthesis_node(state: AgentState) -> Dict[ str , Any ]:
    # 1. Comprobar si ha habido un error irrecuperable (ej. pocos resultados tras MAX_RETRIES)
    successful_sources = sum ( 1 for v in state.get ( "node_ready" , {} ).values ( ) if v )
    retry_count = state.get ( "retry_count" , 0 )

    if successful_sources < 2 and retry_count >= MAX_RETRIES:
        final_synthesis = (
            f"ERROR: Investigación incompleta. Se obtuvieron pocos datos relevantes ({successful_sources} fuentes exitosas) "
            f"tras {retry_count} intentos. Últimos fallos: {state.get ( 'failed_nodes' , [ ] )}. No es posible generar una síntesis completa."
        )
        return {**state , "synthesis": final_synthesis}

    # 2. Recopilar contenido filtrado
    sources_content_list = [ ]

    source_keys = [
        ("Google" , "google_pruned") , ("DuckDuckGo" , "ddg_pruned") ,
        ("Wikipedia" , "wikipedia_pruned") , ("News" , "news_pruned") ,
        ("Arxiv" , "arxiv_pruned") , ("PubMed" , "pubmed_pruned") ,
        ("YouTube" , "youtube_pruned") , ("Bing" , "bing_pruned") ,
        ("Stack Overflow" , "stackoverflow_pruned") , ("GitHub" , "github_pruned")
        ]

    for nom , camp in source_keys:
        content = state.get ( camp , "" )

        # Usamos los resultados BRUTOS (raw_content) si el filtrado dio un error
        # o si el resultado filtrado es vacío/irrelevante. Pero principalmente nos
        # centramos en los filtrados que no sean errores explícitos.
        if content and "Error" not in content and "Sin resultados relevantes" not in content and content.strip ( ):
            # Añadimos el encabezado aquí para que el LLM sepa qué agrupar
            sources_content_list.append ( f"### Fuente: {nom}\n{content}" )
        elif content and "Error" in content:
            # Incluimos los errores en la fuente para ser transparentes en la salida
            sources_content_list.append ( f"### Fuente: {nom} (FALLIDA)\n{content}" )

    sources_content = "\n\n" + "\n".join ( sources_content_list )

    if not sources_content_list:
        final_synthesis = f"ERROR: No se pudo obtener ninguna fuente relevante para la consulta: {state[ 'query' ]}."
        return {**state , "synthesis": final_synthesis}

    # 3. Generar el prompt final con la instrucción bilingüe y de estructuración
    final_prompt = SYNTHESIS_PROMPT_TEMPLATE.format (
        query = state[ 'query' ] ,
        sources_content = sources_content
        )

    # 4. Invocar al LLM para la síntesis
    try:
        logging.info ( "    [Node: Synthesis] Generando síntesis estructurada en el idioma de la consulta original." )
        llm_response = DEEPSEEK_MODEL.invoke ( [ HumanMessage ( content = final_prompt ) ] )
        final_synthesis = llm_response.content
    except Exception as e:
        logging.error ( f"    [Node: Synthesis] Fallo al generar la síntesis final: {e}" )
        final_synthesis = (
            f"ERROR: Fallo crítico al generar la síntesis final de la investigación. "
            f"Consulta original: {state[ 'query' ]}. Error de LLM: {e}"
        )

    return {**state , "synthesis": final_synthesis}


# --- CONSTRUCCIÓN DEL GRAFO (Sin cambios en la estructura) ---
graph = StateGraph ( AgentState )
tool_nodes_list = [
    "google" , "ddg" , "wikipedia" , "news" ,
    "arxiv" , "pubmed" , "youtube" ,
    "bing" , "stackoverflow" ,
    "github"
    ]

graph.add_node ( "refine_query" , refine_query_node )

for node_name in tool_nodes_list:
    graph.add_node ( node_name , globals ( )[ f"{node_name}_node" ] )

graph.add_node ( "verify_retry" , verification_retry_node )
graph.add_node ( "prune_results" , prune_results_node )
graph.add_node ( "synthesis" , synthesis_node )

# 1. Flujo Inicial
graph.add_edge ( START , "refine_query" )

# 2. Flujo de Herramientas (Paralelo)
for node in tool_nodes_list:
    graph.add_edge ( "refine_query" , node )

# 3. Flujo a Verificación
for node in tool_nodes_list:
    graph.add_edge ( node , "verify_retry" )


# 4. Router para Reintento o Poda
def route_check(state: AgentState) -> str:
    # Si hay nodos fallidos Y no se han agotado los reintentos, reintentar
    if state.get ( "failed_nodes" ) and state.get ( "retry_count" , 0 ) < MAX_RETRIES:
        return "retry_tools"

    # En cualquier otro caso, continuar a la poda y síntesis.
    return "prune_and_synthesis"


graph.add_conditional_edges (
    "verify_retry" ,
    route_check ,
    {
        "retry_tools": "refine_query" ,
        "prune_and_synthesis": "prune_results"
        }
    )

# 5. Flujo de Poda a Síntesis
graph.add_edge ( "prune_results" , "synthesis" )

# 6. Flujo Final
graph.add_edge ( "synthesis" , END )

research_agent = graph.compile ( )
logging.info (
    "[Workflow] Workflow de Investigación con Pruning y Reintento compilado con éxito, incluyendo Bing, Stack Overflow y GitHub." )


# --- Ejecución de Prueba ---
if __name__ == "__main__":

    question = "your question"

    initial_state = {
        "query": question ,
        "retry_count": 0 ,
        "failed_nodes": [ ] ,
        "node_ready": {} ,
        "synthesis": ""
        }

    print ( f"\n--- INICIANDO BÚSQUEDA (CON PRUNING Y RESILIENCIA) para: {question} ---" )

    try:
        # Esto fallará si las herramientas no están configuradas en el entorno,
        # pero demuestra la lógica del grafo.
        answer = research_agent.invoke ( initial_state )

        print ( "\n--- Respuesta Final del Agente (RESULTADOS FILTRADOS Y SÍNTESIS) ---\n" )
        print ( answer[ "synthesis" ] )
    except Exception as e:
        print ( f"\n--- ERROR DE EJECUCIÓN REAL ---" )
        print ( f"Ocurrió un error al ejecutar el grafo: {e}" )
