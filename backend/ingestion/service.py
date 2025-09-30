import os
import pandas as pd
import re
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Any
import chromadb
import logging
from logs.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    logger.warning("Warning: scikit-learn not installed. Semantic chunking will not be available.")
    cosine_similarity = None

class ChunkerEmbedder:
    def __init__(self, azure_endpoint: str, azure_api_key: str, azure_api_version: str,
                 embedding_model: str, chroma_persist_dir: str, semantic_threshold: float = 0.75):
        """
        Initialize the ChunkerEmbedder with Azure OpenAI and ChromaDB settings.
        
        Args:
            semantic_threshold: Cosine similarity threshold for semantic chunking (0-1)
        """
        # Initialize Azure OpenAI embeddings via LangChain wrapper
        self.client = AzureOpenAIEmbeddings(
            model=embedding_model,              # Deployment name for embeddings
            deployment=embedding_model,         # Same as model if you named deployment that way
            openai_api_key=azure_api_key,
            openai_api_version=azure_api_version,
            azure_endpoint=azure_endpoint
        )
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)

        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name="insurance_chunks",
                metadata={"description": "Insurance document chunks with embeddings"}
            )
            logger.info("Created new ChromaDB collection 'insurance_chunks'")
        except:
            self.collection = self.chroma_client.get_collection("insurance_chunks")
            logger.info("Using existing ChromaDB collection 'insurance_chunks'")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI (LangChain wrapper)."""
        try:
            logger.debug(f"Getting embedding for text of length {len(text)}")
            return self.client.embed_query(text)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            logger.error(f"Error getting embedding: {e}")
            return None

    def extract_table_references(self, text: str) -> List[str]:
        """Extract table references from text (e.g., [See Vaccination_Cover.csv])."""
        pattern = r'\[See ([^\]]+\.csv)\]'
        logger.debug(f"Extracting table references from text: {text}")
        return re.findall(pattern, text)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # Simple sentence splitting that preserves structure
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                clean_sentences.append(sentence)
        logger.debug(f"Split text into {len(clean_sentences)} sentences")
        return clean_sentences

    def semantic_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Apply semantic chunking to text using embeddings."""
        if cosine_similarity is None:
            print("Warning: scikit-learn not available. Falling back to simple chunking.")
            logger.warning("scikit-learn not available. Falling back to simple chunking.")
            return [text]
            
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for all sentences
        logger.info(f"Semantic chunking: processing {len(sentences)} sentences")
        print(f"   Getting embeddings for {len(sentences)} sentences...")
        embeddings = []
        for sentence in sentences:
            emb = self.get_embedding(sentence)
            if emb:
                embeddings.append(emb)
            else:
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)  # Assuming 1536-dim embeddings
        
        if len(embeddings) <= 1:
            return [text]
        
        # Calculate semantic similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Find semantic boundaries (where similarity drops below threshold)
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_size = len(sentences[0])
        
        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_sentence_size = len(next_sentence)
            
            # Check if we should start a new chunk
            should_split = (
                sim < self.semantic_threshold or  # Semantic boundary
                current_chunk_size + next_sentence_size > max_chunk_size  # Size limit
            )
            
            if should_split and current_chunk_sentences:
                # Finish current chunk
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [next_sentence]
                current_chunk_size = next_sentence_size
            else:
                # Add to current chunk
                current_chunk_sentences.append(next_sentence)
                current_chunk_size += next_sentence_size
        
        # Add final chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
        logger.info(f"Semantic chunking produced {len(chunks)} chunks")
        return chunks if chunks else [text]

    def chunk_text_files(self, output_dir: str) -> List[Dict[str, Any]]:
        """Chunk text files into paragraphs, with heading carryover."""
        text_chunks = []
        carryover_heading = None

        text_files = [f for f in os.listdir(output_dir) if f.endswith("_text.txt")]
        text_files.sort(key=lambda x: int(x.split("_")[1]))  # Sort by page number

        for fname in text_files:
            page_num = int(fname.split("_")[1])
            filepath = os.path.join(output_dir, fname)

            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            paragraphs = re.split(r'\n\s*\n', content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            if len(paragraphs) == 1:  # fallback to line-by-line
                lines = content.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                current_chunk, paragraphs = [], []
                for line in lines:
                    if re.match(r'^\([A-Za-z0-9]+|[IVXivx]+\)', line) and current_chunk:
                        paragraphs.append(' '.join(current_chunk))
                        current_chunk = [line]
                    elif line.startswith('[See ') and line.endswith(']'):
                        if current_chunk:
                            paragraphs.append(' '.join(current_chunk))
                            current_chunk = []
                        paragraphs.append(line)
                    else:
                        current_chunk.append(line)
                if current_chunk:
                    paragraphs.append(' '.join(current_chunk))

            for idx, para in enumerate(paragraphs):
                if not para.strip():
                    continue
                if (re.match(r'^(Section\s+[A-Z][A-Za-z0-9\.]*)', para) and len(para.split()) < 10) or \
                        (re.match(r'^\([A-Za-z0-9]+|[IVXivx]+\)', para) and len(para.split()) < 15):
                    carryover_heading = para
                    continue
                if carryover_heading:
                    para = f"{carryover_heading}\n{para}"
                    carryover_heading = None

                # Apply semantic chunking if paragraph is large
                if len(para) > 800:  # Apply semantic chunking for large paragraphs
                    logger.info(f"   Applying semantic chunking to large paragraph (page {page_num})...")
                    print(f"   Applying semantic chunking to large paragraph (page {page_num})...")
                    semantic_chunks = self.semantic_chunk_text(para)
                    
                    for sem_idx, semantic_chunk in enumerate(semantic_chunks):
                        table_refs = self.extract_table_references(semantic_chunk)
                        text_chunks.append({
                            "text": semantic_chunk.strip(),
                            "metadata": {
                                "type": "text",
                                "page_num": page_num,
                                "source_file": fname,
                                "chunk_idx": f"{idx}_{sem_idx}",
                                "table_references": ",".join(table_refs) if table_refs else "",
                                "chunking_method": "semantic"
                            }
                        })
                else:
                    # Use original chunking for smaller paragraphs
                    table_refs = self.extract_table_references(para)
                    text_chunks.append({
                        "text": para.strip(),
                        "metadata": {
                            "type": "text",
                            "page_num": page_num,
                            "source_file": fname,
                            "chunk_idx": idx,
                            "table_references": ",".join(table_refs) if table_refs else "",
                            "chunking_method": "paragraph"
                        }
                    })

        if carryover_heading:  # orphan heading
            text_chunks.append({
                "text": carryover_heading.strip(),
                "metadata": {
                    "type": "text",
                    "page_num": page_num,
                    "source_file": fname,
                    "chunk_idx": 999,
                    "table_references": "",
                    "chunking_method": "heading"
                }
            })
        logger.info(f"Chunked text files into {len(text_chunks)} chunks")
        return text_chunks

    def chunk_table_files(self, output_dir: str) -> List[Dict[str, Any]]:
        """Chunk table CSV files into rows + header."""
        table_chunks = []
        for fname in os.listdir(output_dir):
            if fname.endswith(".csv") and fname != "table_file_map.csv":
                filepath = os.path.join(output_dir, fname)
                try:
                    df = pd.read_csv(filepath)
                    header_text = f"Table: {fname}\nColumns: {' | '.join(df.columns)}"
                    table_chunks.append({
                        "text": header_text,
                        "metadata": {
                            "type": "table_header",
                            "table_file": fname,
                            "row_idx": -1,
                            "total_rows": len(df)
                        }
                    })
                    for idx, row in df.iterrows():
                        row_text = f"Table: {fname}\n" + " | ".join(
                            [f"{col}: {str(row[col])}" for col in df.columns if pd.notna(row[col])])
                        table_chunks.append({
                            "text": row_text,
                            "metadata": {
                                "type": "table_row",
                                "table_file": fname,
                                "row_idx": idx,
                                "total_rows": len(df),
                                "columns": "|".join(df.columns)
                            }
                        })
                    logger.info(f"Processed table {fname} with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error processing table {fname}: {e}")
                    print(f"Error processing table {fname}: {e}")
                    continue
        logger.info(f"Chunked table files into {len(table_chunks)} chunks")
        return table_chunks

    def embed_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings and store in ChromaDB."""
        documents, embeddings, metadatas, ids = [], [], [], []
        for i, chunk in enumerate(chunks):
            text, metadata = chunk["text"], chunk["metadata"]
            embedding = self.get_embedding(text)
            if embedding is None:
                print(f"Skipping chunk {i} due to embedding error")
                continue
            chunk_type = metadata["type"]
            if chunk_type == "text":
                chunk_id = f"text_{metadata['page_num']}_{metadata['chunk_idx']}"
            elif chunk_type in ["table_row", "table_header"]:
                chunk_id = f"table_{metadata['table_file']}_{metadata['row_idx']}"
            else:
                chunk_id = f"chunk_{i}"
            documents.append(text)
            embeddings.append(embedding)
            metadatas.append(metadata)
            ids.append(chunk_id)
        if documents:
            self.collection.add(documents=documents, embeddings=embeddings,
                                metadatas=metadatas, ids=ids)
            logger.info(f"Stored {len(documents)} chunks in ChromaDB")

    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], output_path: str) -> None:
        """Save chunks to a text file for verification (tables grouped)."""
        with open(output_path, "w", encoding="utf-8") as f:
            i = 1
            idx = 0
            while idx < len(chunks):
                chunk = chunks[idx]
                if chunk["metadata"]["type"] == "table_header":
                    fname = chunk["metadata"]["table_file"]
                    f.write(f"\n=== Table Group: {fname} ===\n")
                    f.write(f"Header: {chunk['text']}\n")
                    idx += 1
                    while idx < len(chunks) and chunks[idx]["metadata"]["type"] == "table_row" and \
                            chunks[idx]["metadata"]["table_file"] == fname:
                        row = chunks[idx]
                        f.write(f"Row {row['metadata']['row_idx']}: {row['text']}\n")
                        idx += 1
                    f.write("\n")
                else:
                    chunking_method = chunk['metadata'].get('chunking_method', 'paragraph')
                    f.write(f"=== Chunk {i} [{chunking_method.upper()}] ===\n")
                    f.write(f"Type: {chunk['metadata']['type']}\n")
                    f.write(f"Chunking Method: {chunking_method}\n")
                    f.write(f"Metadata: {chunk['metadata']}\n")
                    f.write("Text:\n" + chunk["text"] + "\n\n")
                    idx += 1
                    i += 1
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def process_all_data(self, output_dir: str) -> None:
        """Process text + table files, embed, store, and save preview."""
        logger.info("Processing text files...")
        text_chunks = self.chunk_text_files(output_dir)
        logger.info(f"Created {len(text_chunks)} text chunks")
        logger.info("Processing table files...")
        table_chunks = self.chunk_table_files(output_dir)
        logger.info(f"Created {len(table_chunks)} table chunks")
        logger.info("Generating embeddings and storing in ChromaDB...")
        all_chunks = text_chunks + table_chunks
        self.embed_and_store_chunks(all_chunks)
        preview_path = os.path.join(output_dir, "all_chunks_preview.txt")
        self.save_chunks_to_file(all_chunks, preview_path)
        logger.info("Processing complete!")
        logger.info("=== SUMMARY ===")
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        logger.info(f"Text chunks: {len(text_chunks)}")
        logger.info(f"Table chunks: {len(table_chunks)}")
        
        # Count semantic vs paragraph chunks
        semantic_count = sum(1 for chunk in text_chunks 
                           if chunk['metadata'].get('chunking_method') == 'semantic')
        paragraph_count = sum(1 for chunk in text_chunks 
                            if chunk['metadata'].get('chunking_method') == 'paragraph')
        
        logger.info(f"Semantic chunks: {semantic_count}")
        logger.info(f"Paragraph chunks: {paragraph_count}")
        logger.info(f"Collection size: {self.collection.count()}")
        
        if semantic_count > 0:
            logger.info("Semantic chunking is ACTIVE and working!")
        else:
            logger.info("No large paragraphs found - semantic chunking not triggered")

    def query_similar(self, query_text: str, n_results: int = 5,
                      filter_type: str = None) -> Dict[str, Any]:
        """Query similar chunks from ChromaDB."""
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return None
        where_filter = {"type": filter_type} if filter_type else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results


# def main():
#     AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
#     AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
#     AZURE_API_VERSION = os.getenv("AZURE_OPENAI_TEXT_VERSION")
#     EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_EMBEDDINGS")

#     OUTPUT_DIR = r"C:\repo\certification\lessons\chunk_manual_rag\data\output\ActivAssure"
#     CHROMA_DB_DIR = r"C:\repo\certification\lessons\chunk_manual_rag\data\output\chroma_db\ActivAssure"

#     chunker = ChunkerEmbedder(
#         azure_endpoint=AZURE_ENDPOINT,
#         azure_api_key=AZURE_API_KEY,
#         azure_api_version=AZURE_API_VERSION,
#         embedding_model=EMBEDDING_MODEL,
#         chroma_persist_dir=CHROMA_DB_DIR,
#         semantic_threshold=0.75  # Adjust this value (0.5-0.9) based on your needs
#     )

#     chunker.process_all_data(OUTPUT_DIR)

#     # print("\n=== EXAMPLE QUERY ===")
#     # results = chunker.query_similar("vaccination coverage for children", n_results=3)
#     # if results:
#     #     for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
#     #         print(f"\nResult {i+1}:")
#     #         print(f"Type: {metadata['type']}")
#     #         print(f"Content: {doc[:200]}...")
#     #         print(f"Metadata: {metadata}")


# if __name__ == "__main__":
#     main()
