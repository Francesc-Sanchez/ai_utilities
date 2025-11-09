from typing import List, Dict, Any

from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


# 🔍 ArXiv Document Loading
def load_arxiv_docs(query: str) -> str:
    """
    Loads up to 3 documents from ArXiv based on the input query and returns their combined content.
    """
    loader = ArxivLoader(query=query, load_max_docs=3)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


# 🔍 Article ID Retrieval
def get_arxiv_entry_id(query: str) -> str:
    """
    Retrieves the Entry ID of the most relevant ArXiv article for the given query.
    """
    retriever = ArxivRetriever()
    docs = retriever.invoke(query)
    # Note: Assuming docs is not empty based on typical retriever behavior
    return docs[0].metadata.get("Entry ID", "No ID found")


# 🔍 Metadata Formatting
def format_docs(docs) -> str:
    """
    Formats and returns the metadata of a list of documents as a readable string.
    """
    return "\n\n".join(str(doc.metadata) for doc in docs)


# 🔍 Structured Wrapper to get Title and Link
def get_arxiv_wrapper_structured(query: str) -> List[Dict[str, Any]]:
    """
    Uses the ArXiv Retriever to get structured metadata (Title and Link)
    for the most relevant documents.
    """
    # Using ArxivRetriever as it is designed to return Document objects with metadata
    retriever = ArxivRetriever(load_max_docs=5)

    # docs is a list of Document objects
    docs = retriever.invoke(query)

    structured_results = []

    for doc in docs:
        metadata = doc.metadata

        # The document link is stored in the metadata under the key 'Entry ID'
        link = metadata.get("Entry ID", "No link found")

        structured_results.append(
            {
                "title": metadata.get("Title", "No title found"),
                "summary": doc.page_content[:600] + "...",  # A short summary snippet
                "link": link
            })

    # Return the structured list, not a simple string.
    return structured_results


# 🔍 Direct Wrapper (Updated for Agent Use)
def get_arxiv_wrapper(query: str) -> str:
    """
    Uses the ArXiv API wrapper to retrieve key metadata (Title and Link) for the given query
    and returns a formatted string suitable for LLM context.
    """
    # Get structured results
    structured_data = get_arxiv_wrapper_structured(query)

    # Format the list of results into a clean string
    output = "ArXiv Results:\n"
    for idx, res in enumerate(structured_data):
        output += f"--- Document {idx + 1} ---\n"
        output += f"Title: {res['title']}\n"
        output += f"Summary (Snippet): {res['summary']}\n"
        output += f"Link: {res['link']}\n\n"

    return output


# 🔍 ArXiv Context Response Chain
def get_arxiv_chain_results(question: str) -> str:
    """
    Generates a detailed answer (~2000 words) to the input question using retrieved ArXiv documents as context.
    Includes links to the original ArXiv sources.
    """
    prompt = ChatPromptTemplate.from_template(
        """Answer the question in more or less 2000 words, based only on the following context. 
        Provide links to the arXiv sources of the documents:
        {context}
        Question: {question}"""
        )

    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    retriever = ArxivRetriever(load_max_docs=3, get_full_documents=True)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: x)
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke(question)


# # 🧪 Function Tests
# if __name__ == "__main__":
#     test_query = "paleogenetics"
#
#     # print("\n🔍 Testing load_arxiv_docs:")
#     # print(load_arxiv_docs(test_query))
#
#     # print("\n🔍 Testing get_arxiv_entry_id:")
#     # print(get_arxiv_entry_id(test_query))
#     # #
#     print("\n🔍 Testing get_arxiv_wrapper:")
#     print(get_arxiv_wrapper(test_query))
#
#     # print("\n🔍 Testing get_arxiv_chain_results:")
#     # print(get_arxiv_chain_results(test_query))