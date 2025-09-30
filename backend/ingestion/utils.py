import os
import csv
import pdfplumber
from collections import defaultdict
import logging
from logs.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# try:
#     from sklearn.metrics.pairwise import cosine_similarity
# except ImportError:
#     logger.warning("Warning: scikit-learn not installed. Semantic chunking will not be available.")
#     cosine_similarity = None

# --- TABLE EXTRACTION SCRIPT ---
def extract_and_save_tables(pdf_path, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    import pdfplumber
    tables = []
    table_page_map = []
    table_file_map_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.find_tables(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3})
            for t_idx, tbl in enumerate(page_tables, start=1):
                tables.append(tbl.extract())
                table_page_map.append(page_num)
                # Default filename for each table before merging
                table_file_map_rows.append([page_num, t_idx, f"table_page_{page_num}_table_{t_idx}.csv"])

    # Merge split tables and track page numbers
    merged_tables = []
    merged_table_pages = []
    i = 0
    while i < len(tables):
        current = tables[i]
        pages = [table_page_map[i]]
        while i + 1 < len(tables):
            next_table = tables[i+1]
            next_page = table_page_map[i+1]
            if next_page == pages[-1] + 1:
                headers_match = current and next_table and current[0] == next_table[0]
                def get_last_item_num(table):
                    for row in reversed(table[1:]):
                        try:
                            num = int(str(row[0]).strip())
                            return num
                        except:
                            continue
                    return None
                def get_first_item_num(table):
                    for row in table[1:]:
                        try:
                            num = int(str(row[0]).strip())
                            return num
                        except:
                            continue
                    return None
                last_num = get_last_item_num(current)
                next_first_num = get_first_item_num(next_table)
                sequential = last_num is not None and next_first_num is not None and next_first_num == last_num + 1
                if headers_match or sequential:
                    current += next_table[1:]
                    pages.append(next_page)
                    i += 1
                else:
                    break
            else:
                break
        merged_tables.append(current)
        merged_table_pages.append(pages)
        i += 1

    # Save only merged tables, with page numbers in filename
    page_range_counter = {}
    for i, (table, pages) in enumerate(zip(merged_tables, merged_table_pages)):
        page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else f"{pages[0]}"
        if page_range not in page_range_counter:
            page_range_counter[page_range] = 1
        else:
            page_range_counter[page_range] += 1
        counter = page_range_counter[page_range]
        filename = f"table_merged_pages_{page_range}_{counter}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(table)
        logger.info(f"Saved merged: {filepath}")

    # Write table_file_map.csv for user editing
    map_csv_path = os.path.join(output_dir, "table_file_map.csv")
    with open(map_csv_path, "w", newline='', encoding="utf-8") as mapf:
        writer = csv.writer(mapf)
        writer.writerow(["page_num", "table_idx", "table_filename"])
        writer.writerows(table_file_map_rows)
    logger.info(f"Generated table_file_map.csv for manual editing: {map_csv_path}")



def load_table_file_map(csv_path):
    table_file_map = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                page_num = int(row['page_num'])
                table_idx = int(row['table_idx'])
                table_filename = row['table_filename']
                table_file_map[(page_num, table_idx)] = table_filename
            except Exception:
                continue
    return table_file_map

def extract_text(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    table_map_csv = os.path.join(output_dir, "table_file_map.csv")
    if os.path.exists(table_map_csv):
        table_file_map = load_table_file_map(table_map_csv)
    else:
        table_file_map = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.find_tables(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3})
            table_bbox_map = [tbl.bbox for tbl in page_tables]
            words = page.extract_words()
            non_table_words = []
            for w in words:
                in_table = False
                for bbox in table_bbox_map:
                    x0, y0, x1, y1 = bbox
                    if x0 <= float(w['x0']) <= x1 and y0 <= float(w['top']) <= y1:
                        in_table = True
                        break
                if not in_table:
                    non_table_words.append(w)

            line_map = defaultdict(list)
            for w in non_table_words:
                line_map[round(w['top'])].append((w['x0'], w['text']))
            sorted_lines = sorted(line_map.items())
            lines = []
            prev_top = None
            for top, words_in_line in sorted_lines:
                line_text = ' '.join([t[1] for t in sorted(words_in_line, key=lambda x: x[0])])
                if prev_top is not None and abs(top - prev_top) > 20:
                    lines.append('')
                lines.append(line_text)
                prev_top = top
            non_table_text = '\n'.join(lines).strip()

            table_refs = []
            for t_idx, _ in enumerate(page_tables, start=1):
                ref_filename = table_file_map.get((page_num, t_idx), None)
                if ref_filename:
                    table_refs.append(f"[See {ref_filename}]")
                else:
                    table_refs.append(f"[See table from page {page_num} table {t_idx}]")

            text_filename = f"page_{page_num}_text.txt"
            text_filepath = os.path.join(output_dir, text_filename)
            with open(text_filepath, "w", encoding="utf-8") as tf:
                if non_table_text:
                    tf.write(non_table_text + '\n')
                for ref in table_refs:
                    tf.write(ref + '\n')

# --- CHUNKING AND EMBEDDING CLASS ---
# class ChunkerEmbedder:
#     def __init__(self, azure_endpoint: str, azure_api_key: str, azure_api_version: str,
#                  embedding_model: str, chroma_persist_dir: str, semantic_threshold: float = 0.75):
#         """
#         Initialize the ChunkerEmbedder with Azure OpenAI and ChromaDB settings.
        
#         Args:
#             semantic_threshold: Cosine similarity threshold for semantic chunking (0-1)
#         """
#         # Initialize Azure OpenAI embeddings via LangChain wrapper
#         self.client = AzureOpenAIEmbeddings(
#             model=embedding_model,              # Deployment name for embeddings
#             deployment=embedding_model,         # Same as model if you named deployment that way
#             openai_api_key=azure_api_key,
#             openai_api_version=azure_api_version,
#             azure_endpoint=azure_endpoint
#         )
#         self.embedding_model = embedding_model
#         self.semantic_threshold = semantic_threshold

#         # Initialize ChromaDB
#         self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)

#         # Create or get collection
#         try:
#             self.collection = self.chroma_client.create_collection(
#                 name="insurance_chunks",
#                 metadata={"description": "Insurance document chunks with embeddings"}
#             )
#         except:
#             self.collection = self.chroma_client.get_collection("insurance_chunks")

#     def get_embedding(self, text: str) -> List[float]:
#         """Get embedding for text using Azure OpenAI (LangChain wrapper)."""
#         try:
#             return self.client.embed_query(text)
#         except Exception as e:
#             logger.error(f"Error getting embedding: {e}")
#             return None

#     def extract_table_references(self, text: str) -> List[str]:
#         """Extract table references from text (e.g., [See Vaccination_Cover.csv])."""
#         pattern = r'\[See ([^\]]+\.csv)\]'
#         return re.findall(pattern, text)

#     def split_into_sentences(self, text: str) -> List[str]:
#         """Split text into sentences while preserving structure."""
#         # Simple sentence splitting that preserves structure
#         sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
#         # Clean and filter sentences
#         clean_sentences = []
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if sentence and len(sentence) > 10:
#                 clean_sentences.append(sentence)
                
#         return clean_sentences

#     def semantic_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
#         """Apply semantic chunking to text using embeddings."""
#         if cosine_similarity is None:
#             logger.warning("scikit-learn not available. Falling back to simple chunking.")
#             return [text]
            
#         sentences = self.split_into_sentences(text)
        
#         if len(sentences) <= 1:
#             return [text]
        
#         logger.info(f"Getting embeddings for {len(sentences)} sentences...")
#         embeddings = []
#         for sentence in sentences:
#             emb = self.get_embedding(sentence)
#             if emb:
#                 embeddings.append(emb)
#             else:
#                 logger.warning(f"Could not get embedding for sentence")
#                 return [text]
                
#         if len(embeddings) <= 1:
#             return [text]
        
#         # Calculate semantic similarities between consecutive sentences
#         similarities = []
#         for i in range(len(embeddings) - 1):
#             sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
#             similarities.append(sim)
        
#         # Find semantic boundaries (where similarity drops below threshold)
#         chunks = []
#         current_chunk_sentences = [sentences[0]]
#         current_chunk_size = len(sentences[0])
        
#         for i, sim in enumerate(similarities):
#             next_sentence = sentences[i + 1]
#             next_sentence_size = len(next_sentence)
            
#             # Check if we should start a new chunk
#             if (sim < self.semantic_threshold or 
#                 current_chunk_size + next_sentence_size > max_chunk_size):
                
#                 # Add current chunk
#                 chunks.append(' '.join(current_chunk_sentences))
#                 current_chunk_sentences = [next_sentence]
#                 current_chunk_size = next_sentence_size
#             else:
#                 # Add to current chunk
#                 current_chunk_sentences.append(next_sentence)
#                 current_chunk_size += next_sentence_size
        
#         # Add final chunk
#         if current_chunk_sentences:
#             chunks.append(' '.join(current_chunk_sentences))
        
#         return chunks if chunks else [text]

#     def chunk_text_files(self, output_dir: str) -> List[Dict[str, Any]]:
#         """Chunk text files into paragraphs, with heading carryover."""
#         text_chunks = []
#         carryover_heading = None

#         text_files = [f for f in os.listdir(output_dir) if f.endswith("_text.txt")]
#         text_files.sort(key=lambda x: int(x.split("_")[1]))  # Sort by page number

#         for fname in text_files:
#             page_num = int(fname.split("_")[1])
#             file_path = os.path.join(output_dir, fname)
            
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read().strip()
            
#             if not content:
#                 continue
                
#             # Extract table references and remove them from text for chunking
#             table_refs = self.extract_table_references(content)
#             text_without_refs = re.sub(r'\[See [^\]]+\.csv\]', '', content).strip()
            
#             if not text_without_refs:
#                 continue
            
#             # Split by double newlines to get paragraphs
#             paragraphs = [p.strip() for p in text_without_refs.split('\n\n') if p.strip()]
            
#             for para_idx, paragraph in enumerate(paragraphs):
#                 if not paragraph:
#                     continue
                    
#                 # Check if this looks like a heading (short, all caps, etc.)
#                 is_heading = (len(paragraph) < 100 and 
#                             (paragraph.isupper() or 
#                              paragraph.count(' ') < 5 and not paragraph.endswith('.')))
                
#                 if is_heading:
#                     carryover_heading = paragraph
#                     continue
                
#                 # Apply semantic chunking to this paragraph
#                 semantic_chunks = self.semantic_chunk_text(paragraph)
                
#                 for chunk_idx, chunk in enumerate(semantic_chunks):
#                     if carryover_heading:
#                         full_text = f"{carryover_heading}\n\n{chunk}"
#                         carryover_heading = None  # Use heading only once
#                     else:
#                         full_text = chunk
                    
#                     # Add table references at the end
#                     if table_refs:
#                         full_text += "\n\n" + "\n".join(table_refs)
                    
#                     text_chunks.append({
#                         "content": full_text,
#                         "metadata": {
#                             "type": "text",
#                             "page_num": page_num,
#                             "paragraph_idx": para_idx,
#                             "chunk_idx": len(text_chunks),
#                             "chunking_method": "semantic" if len(semantic_chunks) > 1 else "paragraph",
#                             "table_references": table_refs
#                         }
#                     })

#         return text_chunks

#     def chunk_table_files(self, output_dir: str) -> List[Dict[str, Any]]:
#         """Chunk table CSV files into rows + header."""
#         table_chunks = []
#         for fname in os.listdir(output_dir):
#             if not fname.endswith(".csv") or fname == "table_file_map.csv":
#                 continue
            
#             file_path = os.path.join(output_dir, fname)
#             try:
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     reader = csv.reader(f)
#                     rows = list(reader)
                
#                 if not rows:
#                     continue
                    
#                 header = rows[0]
#                 for row_idx, row in enumerate(rows[1:], start=1):
#                     if not any(cell.strip() for cell in row):
#                         continue
                    
#                     # Create chunk with header + row
#                     chunk_content = f"Table: {fname}\nHeader: {', '.join(header)}\nRow {row_idx}: {', '.join(row)}"
                    
#                     table_chunks.append({
#                         "content": chunk_content,
#                         "metadata": {
#                             "type": "table",
#                             "table_file": fname,
#                             "row_idx": row_idx,
#                             "chunk_idx": len(table_chunks),
#                             "header": header
#                         }
#                     })
#             except Exception as e:
#                 logger.error(f"Error processing table {fname}: {e}")
        
#         return table_chunks

#     def embed_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
#         """Generate embeddings and store in ChromaDB."""
#         documents, embeddings, metadatas, ids = [], [], [], []
        
#         for i, chunk in enumerate(chunks):
#             content = chunk["content"]
#             embedding = self.get_embedding(content)
            
#             if embedding:
#                 documents.append(content)
#                 embeddings.append(embedding)
#                 metadatas.append(chunk["metadata"])
#                 ids.append(f"chunk_{i}")
#             else:
#                 logger.warning(f"Could not get embedding for chunk {i}")
        
#         if documents:
#             self.collection.add(
#                 documents=documents,
#                 embeddings=embeddings,
#                 metadatas=metadatas,
#                 ids=ids
#             )
#             logger.info(f"Added {len(documents)} chunks to ChromaDB")

#     def save_chunks_to_file(self, chunks: List[Dict[str, Any]], output_path: str) -> None:
#         """Save chunks to a text file for verification."""
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(f"=== CHUNK PREVIEW ({len(chunks)} total chunks) ===\n\n")
            
#             # Group by type
#             text_chunks = [c for c in chunks if c['metadata']['type'] == 'text']
#             table_chunks = [c for c in chunks if c['metadata']['type'] == 'table']
            
#             f.write(f"TEXT CHUNKS: {len(text_chunks)}\n")
#             f.write(f"TABLE CHUNKS: {len(table_chunks)}\n\n")
            
#             for i, chunk in enumerate(chunks):
#                 f.write(f"--- CHUNK {i+1} ---\n")
#                 f.write(f"Type: {chunk['metadata']['type']}\n")
#                 f.write(f"Metadata: {chunk['metadata']}\n")
#                 f.write(f"Content: {chunk['content'][:500]}...\n\n")
        
#         logger.info(f"Saved {len(chunks)} chunks to {output_path}")

#     def process_all_data(self, output_dir: str) -> None:
#         logger.info("Processing text files...")
#         text_chunks = self.chunk_text_files(output_dir)
#         logger.info(f"Created {len(text_chunks)} text chunks")
#         logger.info("Processing table files...")
#         table_chunks = self.chunk_table_files(output_dir)
#         logger.info(f"Created {len(table_chunks)} table chunks")
#         logger.info("Generating embeddings and storing in ChromaDB...")
#         all_chunks = text_chunks + table_chunks
#         self.embed_and_store_chunks(all_chunks)
#         preview_path = os.path.join(output_dir, "all_chunks_preview.txt")
#         self.save_chunks_to_file(all_chunks, preview_path)
#         logger.info("Processing complete!")
#         logger.info("=== SUMMARY ===")
#         logger.info(f"Total chunks processed: {len(all_chunks)}")
#         logger.info(f"Text chunks: {len(text_chunks)}")
#         logger.info(f"Table chunks: {len(table_chunks)}")
#         semantic_count = sum(1 for chunk in text_chunks 
#                            if chunk['metadata'].get('chunking_method') == 'semantic')
#         paragraph_count = sum(1 for chunk in text_chunks 
#                             if chunk['metadata'].get('chunking_method') == 'paragraph')
#         logger.info(f"Semantic chunks: {semantic_count}")
#         logger.info(f"Paragraph chunks: {paragraph_count}")
#         logger.info(f"Collection size: {self.collection.count()}")

#     def query_similar(self, query_text: str, n_results: int = 5, filter_type: str = None) -> Dict[str, Any]:
#         """Query similar chunks from ChromaDB."""
#         query_embedding = self.get_embedding(query_text)
#         if query_embedding is None:
#             return {"documents": [], "metadatas": []}
        
#         where_filter = {"type": filter_type} if filter_type else None
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_results,
#             where=where_filter
#         )
#         return results