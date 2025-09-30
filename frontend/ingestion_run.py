import streamlit as st
import os
import sys
import re
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import pdfplumber
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
DJANGO_API = os.getenv("API_BASE")


@st.cache_resource
def get_cached_chunker_embedder(chroma_db_dir: str, output_dir: str):
    """Cached function to call Django API for chunking and embedding."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Call Django API for chunking and embedding
        resp = requests.post(
            f"{DJANGO_API}/api/chunk_and_embed/",
            json={"output_dir": output_dir, "chroma_db_dir": chroma_db_dir}
        )
        
        if resp.status_code == 200:
            result = resp.json()
            return {"success": True, "message": result.get("message"), "collection_size": result.get("collection_size")}
        else:
            error_msg = resp.json().get("error", "Unknown error") if resp.headers.get('content-type') == 'application/json' else resp.text
            return {"success": False, "error": error_msg}
        
    except Exception as e:
        return {"success": False, "error": f"Error during chunking and embedding: {str(e)}"}


class StreamlitRAGPipeline:
    def __init__(self):
        self.pdf_path = None
        self.pdf_name = None
        self.base_output_dir = None
        self.output_dir = None
        self.chroma_db_dir = None
        self.has_tables_flag = False
        self.table_count = 0
        
    def clean_pdf_name(self, pdf_path: str) -> str:
        """Clean PDF filename to create valid folder name."""
        filename = Path(pdf_path).stem
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        cleaned = re.sub(r'[_\s]+', '_', cleaned)
        cleaned = cleaned.strip('_')
        if len(cleaned) > 50:
            cleaned = cleaned[:50].rstrip('_')
        return cleaned if cleaned else "unnamed_pdf"
    
    def setup_directories(self, pdf_path: str, base_output_dir: str):
        """Setup directory structure for the pipeline."""
        self.pdf_path = pdf_path
        self.pdf_name = self.clean_pdf_name(pdf_path)
        self.base_output_dir = base_output_dir
        self.output_dir = os.path.join(base_output_dir, self.pdf_name)
        self.chroma_db_dir = os.path.join(base_output_dir, "chroma_db", self.pdf_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        
    def analyze_pdf_content(self) -> dict:
        """Analyze PDF to detect tables and get basic stats."""
        table_count = 0
        total_pages = 0
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                tables = page.find_tables(
                    table_settings={
                        "vertical_strategy": "lines", 
                        "horizontal_strategy": "lines", 
                        "snap_tolerance": 3
                    }
                )
                table_count += len(tables)
        
        self.has_tables_flag = table_count > 0
        self.table_count = table_count
        
        return {
            "has_tables": self.has_tables_flag,
            "table_count": table_count,
            "total_pages": total_pages,
            "tables_per_page": round(table_count / total_pages, 2) if total_pages > 0 else 0
        }
    
    def check_existing_extractions(self) -> dict:
        """Check what extractions already exist."""
        if not os.path.exists(self.output_dir):
            return {"has_table_map": False, "has_text_files": False, "has_csv_files": False, 
                   "text_file_count": 0, "csv_file_count": 0}
            
        table_map_path = os.path.join(self.output_dir, "table_file_map.csv")
        text_files = [f for f in os.listdir(self.output_dir) if f.endswith("_text.txt")]
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith(".csv") and f != "table_file_map.csv"]
        
        return {
            "has_table_map": os.path.exists(table_map_path),
            "has_text_files": len(text_files) > 0,
            "has_csv_files": len(csv_files) > 0,
            "text_file_count": len(text_files),
            "csv_file_count": len(csv_files)
        }
    
    def check_manual_review_status(self) -> bool:
        """Check if manual review has been completed."""
        marker_file = os.path.join(self.output_dir, ".manual_review_completed")
        return os.path.exists(marker_file)
    
    def mark_review_completed(self):
        """Mark manual review as completed."""
        marker_file = os.path.join(self.output_dir, ".manual_review_completed")
        table_map_path = os.path.join(self.output_dir, "table_file_map.csv")
        
        with open(marker_file, 'w') as f:
            timestamp = os.path.getmtime(table_map_path) if os.path.exists(table_map_path) else time.time()
            f.write(f"Manual review completed at {timestamp}")
    
    def extract_tables(self, force_reextract=False):
        existing = self.check_existing_extractions()
        if existing["has_csv_files"] and not force_reextract:
            return False, f"Found existing table extractions ({existing['csv_file_count']} CSV files)"

        resp = requests.post(
            f"{DJANGO_API}/api/extract_tables/",
            json={"pdf_path": self.pdf_path, "output_dir": self.output_dir}
        )
        if resp.status_code == 200:
            return True, resp.json().get("message")
        return False, resp.json().get("error", "Unknown error")
    
    def extract_text_content(self, force_reextract=False):
        existing = self.check_existing_extractions()
        if existing["has_text_files"] and not force_reextract:
            return False, f"Found existing text extractions ({existing['text_file_count']} text files)"

        resp = requests.post(
            f"{DJANGO_API}/api/extract_text/",
            json={"pdf_path": self.pdf_path, "output_dir": self.output_dir}
        )
        if resp.status_code == 200:
            return True, resp.json().get("message")
        return False, resp.json().get("error", "Unknown error")
    
    def load_table_mapping(self):
        """Load the table file mapping."""
        table_map_path = os.path.join(self.output_dir, "table_file_map.csv")
        if os.path.exists(table_map_path):
            return pd.read_csv(table_map_path)
        return pd.DataFrame()
    
    def save_table_mapping(self, df):
        """Save the updated table file mapping."""
        table_map_path = os.path.join(self.output_dir, "table_file_map.csv")
        df.to_csv(table_map_path, index=False)
    
    def get_extracted_tables(self):
        """Get list of extracted table files."""
        if not os.path.exists(self.output_dir):
            return []
        return [f for f in os.listdir(self.output_dir) if f.endswith(".csv") and f != "table_file_map.csv"]
    
    def chunk_and_embed(self):
        """Run chunking and embedding process via Django API."""
        result = get_cached_chunker_embedder(self.chroma_db_dir, self.output_dir)
        
        if result["success"]:
            # Create a mock chunker object to maintain compatibility
            class MockChunker:
                def __init__(self, collection_size):
                    self.collection_size = collection_size
                    
                @property
                def collection(self):
                    class MockCollection:
                        def __init__(self, size):
                            self._size = size
                        def count(self):
                            return self._size
                    return MockCollection(self.collection_size)
            
            chunker = MockChunker(result.get("collection_size", 0))
            return chunker, result["message"]
        else:
            return None, result["error"]


def main():
    st.set_page_config(
        page_title="Simple RAG Pipeline",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Simple RAG Pipeline")
    st.markdown("**Intelligent PDF Processing with Table Detection and Human-in-the-Loop Review**")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = StreamlitRAGPipeline()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False
    if 'review_complete' not in st.session_state:
        st.session_state.review_complete = False
    if 'chunker_embedder' not in st.session_state:
        st.session_state.chunker_embedder = None
    if 'embedding_complete' not in st.session_state:
        st.session_state.embedding_complete = False
    
    pipeline = st.session_state.pipeline
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload the PDF document you want to process"
        )
        
        # Auto-detect output directory (hidden from user)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from frontend to project root
        base_output_dir = os.path.join(project_root, "media", "output")
        
        # Azure OpenAI Status
        st.subheader("🔑 Azure OpenAI Status")
        from dotenv import load_dotenv
        load_dotenv()
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        
        if endpoint and api_key:
            st.success("✅ Credentials configured")
            st.write(f"**Endpoint:** {endpoint[:30]}...")
        else:
            st.error("❌ Credentials missing")
            st.write("Please set up your .env file")
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_pdf_path = os.path.join(base_output_dir, "temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pipeline.setup_directories(temp_pdf_path, base_output_dir)
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 PDF File", uploaded_file.name)
        with col2:
            st.metric("📁 Output Folder", pipeline.pdf_name)
        with col3:
            st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        st.divider()
        
        # Step 1: PDF Analysis
        st.header("🔍 Step 1: PDF Analysis")
        
        if st.button("Analyze PDF Content", type="primary"):
            with st.spinner("Analyzing PDF content..."):
                analysis = pipeline.analyze_pdf_content()
                st.session_state.analysis_complete = True
                st.session_state.analysis_result = analysis
        
        if st.session_state.analysis_complete:
            analysis = st.session_state.analysis_result
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📖 Total Pages", analysis["total_pages"])
            with col2:
                st.metric("📊 Tables Found", analysis["table_count"])
            with col3:
                st.metric("📈 Tables/Page", analysis["tables_per_page"])
            with col4:
                workflow = "Table Workflow" if analysis["has_tables"] else "Text-Only Workflow"
                st.metric("⚙️ Workflow", workflow)
            
            if analysis["has_tables"]:
                st.success(f"✅ {analysis['table_count']} tables detected - Using table extraction workflow")
            else:
                st.info("ℹ️ No tables detected - Using text-only extraction")
            
            st.divider()
            
            # Step 2: Content Extraction
            st.header("📤 Step 2: Content Extraction")
            
            existing = pipeline.check_existing_extractions()
            
            if existing["has_csv_files"] or existing["has_text_files"]:
                st.warning("⚠️ Existing extractions found!")
                col1, col2 = st.columns(2)
                with col1:
                    if existing["has_csv_files"]:
                        st.write(f"📊 {existing['csv_file_count']} CSV files (tables)")
                with col2:
                    if existing["has_text_files"]:
                        st.write(f"📄 {existing['text_file_count']} text files")
            
            col1, col2 = st.columns(2)
            
            with col1:
                force_table_reextract = st.checkbox("Force re-extract tables", value=False)
                if st.button("Extract Tables", disabled=not analysis["has_tables"]):
                    with st.spinner("Extracting tables..."):
                        success, message = pipeline.extract_tables(force_table_reextract)
                        if success:
                            st.success(message)
                        else:
                            st.info(message)
                        
                        # Check if we need to refresh extraction status
                        st.session_state.extraction_complete = True
            
            with col2:
                force_text_reextract = st.checkbox("Force re-extract text", value=False)
                if st.button("Extract Text"):
                    with st.spinner("Extracting text..."):
                        success, message = pipeline.extract_text_content(force_text_reextract)
                        if success:
                            st.success(message)
                        else:
                            st.info(message)
                        
                        st.session_state.extraction_complete = True
            
            # Step 3: Human Review (for tables)
            if analysis["has_tables"]:
                st.divider()
                st.header("👤 Step 3: Human Review (Tables)")
                
                review_completed = pipeline.check_manual_review_status()
                
                if review_completed:
                    st.success("✅ Manual review already completed")
                    st.session_state.review_complete = True
                else:
                    table_mapping = pipeline.load_table_mapping()
                    extracted_tables = pipeline.get_extracted_tables()
                    
                    if not table_mapping.empty:
                        st.subheader("📊 Table File Mapping")
                        
                        # Option to upload custom table mapping CSV
                        with st.expander("📤 Upload Custom Table Mapping (Optional)"):
                            st.write("Upload a CSV file with your own table mapping configuration:")
                            st.write("**Required columns:** `page_num`, `table_idx`, `table_filename`")
                            
                            uploaded_mapping = st.file_uploader(
                                "Choose table mapping CSV file",
                                type=['csv'],
                                key="table_mapping_upload",
                                help="Upload a CSV with columns: page_num, table_idx, table_filename"
                            )
                            
                            if uploaded_mapping is not None:
                                try:
                                    uploaded_df = pd.read_csv(uploaded_mapping)
                                    
                                    # Validate required columns
                                    required_cols = ['page_num', 'table_idx', 'table_filename']
                                    missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
                                    
                                    if missing_cols:
                                        st.error(f"❌ Missing required columns: {missing_cols}")
                                        st.write("**Current columns:**", list(uploaded_df.columns))
                                    else:
                                        st.success("✅ Valid table mapping CSV uploaded!")
                                        st.write("**Preview of uploaded mapping:**")
                                        st.dataframe(uploaded_df.head(), width='stretch')
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("🔄 Use Uploaded Mapping", key="use_uploaded"):
                                                table_mapping = uploaded_df.copy()
                                                pipeline.save_table_mapping(table_mapping)
                                                st.success("Table mapping updated with uploaded file!")
                                                st.rerun()
                                        
                                        with col2:
                                            if st.button("📋 Merge with Existing", key="merge_uploaded"):
                                                # Merge logic: update existing entries, add new ones
                                                merged_mapping = table_mapping.copy()
                                                for _, row in uploaded_df.iterrows():
                                                    mask = ((merged_mapping['page_num'] == row['page_num']) & 
                                                           (merged_mapping['table_idx'] == row['table_idx']))
                                                    if mask.any():
                                                        merged_mapping.loc[mask, 'table_filename'] = row['table_filename']
                                                    else:
                                                        merged_mapping = pd.concat([merged_mapping, pd.DataFrame([row])], ignore_index=True)
                                                
                                                table_mapping = merged_mapping
                                                pipeline.save_table_mapping(table_mapping)
                                                st.success("Table mapping merged with uploaded file!")
                                                st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"❌ Error reading CSV file: {str(e)}")
                                    st.write("Please ensure the CSV file is properly formatted.")
                        
                        st.write("Review and edit the table filenames if needed:")
                        
                        # Editable table mapping
                        edited_mapping = st.data_editor(
                            table_mapping,
                            # width='stretch',
                            num_rows="fixed",
                            disabled=["page_num", "table_idx"],
                            key="table_mapping_editor"
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("💾 Save Mapping"):
                                pipeline.save_table_mapping(edited_mapping)
                                st.success("Table mapping saved!")
                        
                        with col2:
                            if st.button("✅ Complete Review"):
                                pipeline.save_table_mapping(edited_mapping)
                                pipeline.mark_review_completed()
                                st.success("Review marked as complete!")
                                st.session_state.review_complete = True
                                st.rerun()
                        
                        with col3:
                            if st.button("📄 Switch to Text-Only"):
                                pipeline.has_tables_flag = False
                                st.info("Switched to text-only mode")
                        
                        # Display extracted table files
                        if extracted_tables:
                            st.subheader("📁 Extracted Table Files")
                            for i, table_file in enumerate(extracted_tables):
                                with st.expander(f"📊 {table_file}"):
                                    try:
                                        df = pd.read_csv(os.path.join(pipeline.output_dir, table_file))
                                        st.dataframe(df.head(10), width='stretch')
                                        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                    except Exception as e:
                                        st.error(f"Error reading file: {e}")
                    
                    else:
                        st.warning("⚠️ No table mapping found. Please extract tables first.")
            
            # Step 4: Chunking and Embedding
            if (not analysis["has_tables"]) or st.session_state.review_complete:
                st.divider()
                st.header("🧠 Step 4: Chunking & Embedding")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Output Directory:** `{pipeline.output_dir}`")
                with col2:
                    st.write(f"**ChromaDB Directory:** `{pipeline.chroma_db_dir}`")
                
                # Check extraction status and show warnings
                current_extractions = pipeline.check_existing_extractions()
                
                # Warning system for missing extractions
                warnings_displayed = False
                
                if analysis["has_tables"] and not current_extractions["has_csv_files"]:
                    st.warning("⚠️ **Tables detected but not extracted!** Please extract tables in Step 2 before chunking.")
                    warnings_displayed = True
                
                if not current_extractions["has_text_files"]:
                    st.warning("⚠️ **No text files found!** Please extract text in Step 2 before chunking.")
                    warnings_displayed = True
                
                if analysis["has_tables"] and current_extractions["has_csv_files"] and not current_extractions["has_text_files"]:
                    st.error("🚨 **Critical:** Only tables extracted, no text content! Both text and tables should be included for comprehensive RAG.")
                    warnings_displayed = True
                
                # Show extraction status summary
                if current_extractions["has_text_files"] or current_extractions["has_csv_files"]:
                    st.subheader("📋 Extraction Status Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if current_extractions["has_text_files"]:
                            st.success(f"✅ Text: {current_extractions['text_file_count']} files")
                        else:
                            st.error("❌ Text: Not extracted")
                    
                    with col2:
                        if analysis["has_tables"]:
                            if current_extractions["has_csv_files"]:
                                st.success(f"✅ Tables: {current_extractions['csv_file_count']} files")
                            else:
                                st.error("❌ Tables: Not extracted")
                        else:
                            st.info("➖ Tables: None detected")
                    
                    with col3:
                        if analysis["has_tables"]:
                            review_status = "✅ Complete" if st.session_state.review_complete else "⏳ Pending"
                            st.write(f"**Review:** {review_status}")
                        else:
                            st.info("**Review:** Not needed")
                
                # Show what will be included in chunking
                if current_extractions["has_text_files"] or current_extractions["has_csv_files"]:
                    with st.expander("� Content to be Chunked", expanded=False):
                        st.write("**The following content will be included in chunking and embedding:**")
                        
                        if current_extractions["has_text_files"]:
                            st.write(f"📝 **Text Content:** {current_extractions['text_file_count']} text files")
                            text_files = [f for f in os.listdir(pipeline.output_dir) if f.endswith("_text.txt")]
                            for text_file in text_files[:3]:  # Show first 3 files
                                st.write(f"   • {text_file}")
                            if len(text_files) > 3:
                                st.write(f"   • ... and {len(text_files) - 3} more files")
                        
                        if current_extractions["has_csv_files"]:
                            st.write(f"📊 **Table Content:** {current_extractions['csv_file_count']} table files")
                            csv_files = [f for f in os.listdir(pipeline.output_dir) if f.endswith(".csv") and f != "table_file_map.csv"]
                            for csv_file in csv_files[:3]:  # Show first 3 files
                                st.write(f"   • {csv_file}")
                            if len(csv_files) > 3:
                                st.write(f"   • ... and {len(csv_files) - 3} more files")
                
                # Disable button if critical extractions are missing
                disable_chunking = False
                if not current_extractions["has_text_files"] and not current_extractions["has_csv_files"]:
                    st.error("🚫 **Cannot proceed:** No content extracted. Please complete Step 2 first.")
                    disable_chunking = True
                
                if st.button("�🚀 Start Chunking & Embedding", type="primary", disabled=disable_chunking):
                    with st.spinner("Processing chunks and generating embeddings..."):
                        chunker, message = pipeline.chunk_and_embed()
                        
                        if chunker:
                            st.success(message)
                            # Store chunker in session state
                            st.session_state.chunker_embedder = chunker
                            st.session_state.embedding_complete = True
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Chunks", chunker.collection.count())
                            with col2:
                                # Count text chunks vs table chunks from metadata
                                st.metric("💾 Database", "ChromaDB")
                            with col3:
                                st.metric("✅ Status", "Complete")
                            
                        else:
                            st.error(message)
                
                # Step 5: Query Interface (show if embedding is complete)
                if st.session_state.embedding_complete and st.session_state.chunker_embedder:
                    st.divider()
                    st.header("🔍 Step 5: Query Interface")
                    
                    chunker = st.session_state.chunker_embedder
                    
                    # Display embedding stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Chunks", chunker.collection.count())
                    with col2:
                        st.metric("💾 Database", "ChromaDB")
                    with col3:
                        st.metric("✅ Status", "Ready for Search")
                    
                    query = st.text_input(
                        "Enter your query:",
                        placeholder="e.g., insurance coverage benefits",
                        key="query_input"
                    )
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        n_results = st.slider("Number of results", 1, 10, 3)
                    with col2:
                        filter_type = st.selectbox(
                            "Filter by type",
                            ["All", "text", "table_row", "table_header"]
                        )
                    
                    if st.button("🔎 Search") and query:
                        filter_param = None if filter_type == "All" else filter_type
                        
                        with st.spinner("Searching..."):
                            results = chunker.query_similar(query, n_results, filter_param)
                            
                            if results and results['documents'][0]:
                                st.subheader(f"📋 Search Results for: '{query}'")
                                
                                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                                    with st.expander(f"Result {i+1} ({metadata['type']})"):
                                        st.write("**Content:**")
                                        st.write(doc[:500] + ("..." if len(doc) > 500 else ""))
                                        
                                        st.write("**Metadata:**")
                                        if metadata['type'] == 'text':
                                            st.write(f"- Source: Page {metadata['page_num']}")
                                            st.write(f"- File: {metadata['source_file']}")
                                        elif metadata['type'] in ['table_row', 'table_header']:
                                            st.write(f"- Source: {metadata['table_file']}")
                                            if metadata['type'] == 'table_row':
                                                st.write(f"- Row: {metadata['row_idx']}")
                                        
                                        if metadata.get('table_references'):
                                            st.write(f"- References: {metadata['table_references']}")
                            else:
                                st.warning("No results found for your query.")
    
    else:
        st.info("👆 Please upload a PDF file to get started")
        
        # Display sample workflow
        st.subheader("🔄 Workflow Overview")
        
        workflow_steps = [
            "📤 **Upload PDF** - Select your document",
            "🔍 **Analysis** - Detect tables and content structure", 
            "📊 **Extraction** - Extract tables and text content",
            "👤 **Review** - Human-in-the-loop table verification (if tables found)",
            "🧠 **Processing** - Chunk content and generate embeddings",
            "🔎 **Query** - Search and retrieve relevant information"
        ]
        
        for step in workflow_steps:
            st.markdown(step)


if __name__ == "__main__":
    main()