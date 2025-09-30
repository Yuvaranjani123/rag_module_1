import os
from dotenv import load_dotenv
load_dotenv()  # load AZURE_OPENAI_* variables
import chromadb
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
import logging
from logs.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Import our simple prompt configuration
from config.prompt_config import prompt_config

# ---------------------------
# Direct ChromaDB + LLM Query
# ---------------------------
def query_document_internal(collection, embedding_model, query, k=5):
    logger.info(f"Querying ChromaDB for: '{query}' with top {k} results.")
    # Get query embedding
    try:
        query_embedding = embedding_model.embed_query(query)
    except Exception as e:
        logger.error(f"Error getting embedding for query: {e}")
        return {"answer": "Error getting embedding.", "sources": []}
    # Search ChromaDB directly
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return {"answer": "Error querying database.", "sources": []}
    # Build context from results
    if not results['documents'] or not results['documents'][0]:
        logger.info("No relevant documents found for query.")
        return {"answer": "No relevant documents found.", "sources": []}
    context_parts = []
    sources = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"[Source {i+1}] {doc}")
        sources.append({
            "content": doc,
            "page": metadata.get("page_num"),
            "table": metadata.get("table_file"),
            "row_index": metadata.get("row_idx"),
            "type": metadata.get("type"),
            "chunking_method": metadata.get("chunking_method", "unknown"),
            "chunk_idx": metadata.get("chunk_idx"),
        })
    context = "\n\n".join(context_parts)
    # Initialize LLM
    try:
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"Error initializing AzureChatOpenAI: {e}")
        return {"answer": "Error initializing LLM.", "sources": sources}
    # Use our configured prompt template
    try:
        formatted_prompt = prompt_config.format(
            context=context,
            question=query
        )
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        return {"answer": "Error formatting prompt.", "sources": sources}
    # Get LLM response
    try:
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info("LLM response generated successfully.")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        answer = "Error generating answer from LLM."
    return {"answer": answer, "sources": sources}

@api_view(['POST'])
def query_document(request):
    """API endpoint to query the document database."""
    try:
        query_text = request.data.get("query")
        chroma_db_dir = request.data.get("chroma_db_dir")
        k = request.data.get("k", 5)
        logger.info(f"Received query_document API call: query='{query_text}', chroma_db_dir='{chroma_db_dir}', k={k}")
        if not query_text:
            logger.warning("query_document: 'query' is required.")
            return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not chroma_db_dir:
            logger.warning("query_document: 'chroma_db_dir' is required.")
            return Response({"error": "chroma_db_dir is required"}, status=status.HTTP_400_BAD_REQUEST)
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=chroma_db_dir)
        collection = client.get_collection("insurance_chunks")
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_EMBEDDINGS"),
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_TEXT_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        # Perform query
        result = query_document_internal(collection, embeddings, query_text, k)
        logger.info("Query processed and response returned.")
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in query_document API: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ---------------------------
# Main for CLI testing
# ---------------------------
# def main():
#     persist_dir = r"C:\repo\certification\lessons\chunk_manual_rag\data\output\chroma_db\ActivAssure"  
#     client = chromadb.PersistentClient(path=persist_dir)

#     # Use the same collection name as chunker_embedder.py
#     collection = client.get_collection("insurance_chunks")

#     embeddings = AzureOpenAIEmbeddings(
#         deployment=os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_EMBEDDINGS"),
#         openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
#         openai_api_version=os.getenv("AZURE_OPENAI_TEXT_VERSION"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#     )

#     query = input("Enter your query: ")
#     result = query_document_internal(collection, embeddings, query)

#     print("\nAnswer:", result["answer"])
#     print("\nSources:")
#     for s in result["sources"]:
#         page_info = f"Page {s['page']}" if s['page'] else "No page"
#         table_info = f"Table: {s['table']}" if s['table'] else "Text chunk"
#         row_info = f"Row: {s['row_index']}" if s['row_index'] is not None else ""
#         method_info = f"[{s['chunking_method'].upper()}]" if s.get('chunking_method') else ""
        
#         print(f"- {page_info} | {table_info} {row_info} | Type: {s['type']} {method_info}")
#         print(f"  Chunk ID: {s['chunk_idx']}")
#         print(f"  Snippet: {s['content'][:200]}...\n")

# if __name__ == "__main__":
#     main()
