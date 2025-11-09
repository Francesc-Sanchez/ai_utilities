import os
from typing import List
from Bio import Entrez
from dotenv import load_dotenv

# Load environment variables
load_dotenv ( )
# Set your email for Entrez access (required by NCBI)
Entrez.email = os.getenv ( "EMAIL" )


def fallback_pubmed_search(query: str) -> str:
    """
    Fallback mechanism: Performs a general web search restricted to the PubMed domain
    using DuckDuckGo (DDG).
    (Requires the ddg_general_search function to be available in the environment).
    """
    # 🚨 NOTE: 'ddg_general_search' is assumed to be available.
    # In a real environment, you would uncomment the import from its location:
    from src.tools.registry.external_tools.external_tool_functions import ddg_general_search

    # Restrict the search to the PubMed domain
    ddg_query = f"{query} site:pubmed.ncbi.nlm.nih.gov"

    try:
        result = ddg_general_search ( ddg_query )
        # Add a header to indicate that this is a fallback result
        return f"--- Fallback Search (DDG/Web) on PubMed ---\n{result}"
    except Exception as e:
        return f"❌ Fallback PubMed search failed: {e}"


def search_pubmed(query: str , max_results: int = 5) -> str:
    """
    Searches for articles in PubMed and handles possible Entrez ('webenv') exceptions.
    If the Entrez search fails, it resorts to a general web search (fallback).
    """
    summaries: List[ str ] = [ ]

    try:
        # 1. Search and retrieve IDs
        handle = Entrez.esearch ( db = "pubmed" , term = query , retmax = max_results )
        record = Entrez.read ( handle )
        handle.close ( )

        ids = record.get ( "IdList" , [ ] )

        if not ids:
            # If there are no results, it's not considered an error, just an information message
            return "No articles were found in PubMed for this query."

        # 2. Retrieve summaries
        id_string = ",".join ( ids )
        handle = Entrez.esummary ( db = "pubmed" , id = id_string )
        records = Entrez.read ( handle )
        handle.close ( )

        for r in records:
            title = r.get ( "Title" , "Untitled" )
            pubdate = r.get ( "PubDate" , "No date" )
            pmid = r.get ( "Id" )
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            summaries.append ( f"**Title:** {title}\n**Date:** {pubdate}\n**Link:** {link}" )

        return "\n\n".join ( summaries )


    except Exception as e:
        # 🚨 FALLBACK ACTIVATED: If Entrez fails (e.g., 'webenv' error),
        # the error is reported, and the fallback mechanism is attempted.
        error_message = f"⚠️ Connection/Entrez Error ({e}). Attempting fallback search..."
        print ( error_message )

        # Call the fallback function
        fallback_result = fallback_pubmed_search ( query )

        # Return the original error message followed by the fallback result
        return f"{error_message}\n{fallback_result}"


def main():
    query = input ( "🔍 Enter your PubMed search query: " )
    print ( "\nSearching PubMed...\n" )
    result = search_pubmed ( query )
    print ( result )


# if __name__ == "__main__":
#     main ( )