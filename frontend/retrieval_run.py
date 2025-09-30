import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

# Django API base URL
DJANGO_API = os.getenv("API_BASE")

st.set_page_config(page_title="Insurance RAG - Retrieval", page_icon="🔍")

st.title("🔍 Insurance Document Retrieval")

# Initialize session state
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

# Configuration sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-detect ChromaDB directories (relative path for portability)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up from frontend to project root
    base_output_dir = os.path.join(project_root, "media", "output")
    chroma_base_dir = os.path.join(base_output_dir, "chroma_db")
    print("ChromaDB base directory:", chroma_base_dir)
    # Get available ChromaDB directories
    available_dirs = []
    if os.path.exists(chroma_base_dir):
        for item in os.listdir(chroma_base_dir):
            item_path = os.path.join(chroma_base_dir, item)
            if os.path.isdir(item_path):
                available_dirs.append(item)
    
    if available_dirs:
        selected_doc = st.selectbox(
            "Select Document Collection",
            available_dirs,
            help="Choose which document collection to query"
        )
        chroma_db_dir = os.path.join(chroma_base_dir, selected_doc)
    else:
        st.warning("⚠️ No ChromaDB collections found")
        chroma_db_dir = st.text_input(
            "ChromaDB Directory (Manual)",
            value="",
            help="Manually enter the path to your ChromaDB directory"
        )
    
    k_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    # Django API Status
    st.subheader("🔗 Django API Status")
    try:
        resp = requests.get(f"{DJANGO_API}/retriever/query/", timeout=5)
        if resp.status_code in [200, 405]:  # 405 means endpoint exists but wrong method
            st.success("✅ Django API accessible")
        else:
            st.error(f"❌ Django API error: {resp.status_code}")
    except Exception as e:
        st.error(f"❌ Django API not accessible: {str(e)}")

# Main query interface
query = st.text_input(
    "Ask a question about the insurance document:",
    value=st.session_state.selected_query,
    placeholder="e.g., What vaccinations are covered for children?"
)

if st.button("🔍 Search", type="primary") and query:
    if not chroma_db_dir:
        st.error("Please provide a ChromaDB directory path")
    else:
        with st.spinner("Retrieving answer..."):
            try:
                # Call Django API
                response = requests.post(
                    f"{DJANGO_API}/retriever/query/",
                    json={
                        "query": query,
                        "chroma_db_dir": chroma_db_dir,
                        "k": k_results
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("📌 Answer")
                    st.write(result["answer"])
                    
                    st.subheader("📑 Sources")
                    
                    if result["sources"]:
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i} - {source.get('type', 'Unknown')}"):
                                # Source metadata
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if source.get('page'):
                                        st.write(f"**Page:** {source['page']}")
                                    if source.get('table'):
                                        st.write(f"**Table:** {source['table']}")
                                    if source.get('row_index') is not None:
                                        st.write(f"**Row:** {source['row_index']}")
                                
                                with col2:
                                    st.write(f"**Type:** {source.get('type', 'Unknown')}")
                                    if source.get('chunking_method'):
                                        st.write(f"**Chunking:** {source['chunking_method'].upper()}")
                                    if source.get('chunk_idx') is not None:
                                        st.write(f"**Chunk ID:** {source['chunk_idx']}")
                                
                                # Content
                                st.write("**Content:**")
                                content = source.get('content', '')
                                if len(content) > 500:
                                    st.write(content[:500] + "...")
                                    # Show full content in a text area instead of nested expander
                                    if st.button(f"Show full content {i}", key=f"show_full_{i}"):
                                        st.text_area("Full Content", content, height=200, key=f"full_content_{i}")
                                else:
                                    st.write(content)
                    else:
                        st.info("No sources found")
                        
                else:
                    error_msg = response.json().get("error", "Unknown error") if response.headers.get('content-type') == 'application/json' else response.text
                    st.error(f"API Error ({response.status_code}): {error_msg}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")
                st.info("Make sure the Django server is running on the configured URL")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sample queries section
st.divider()
st.subheader("💡 Sample Queries")

sample_queries = [
    "What vaccinations are covered for children?",
    "What is the claim process for hospitalization?",
    "What are the annual check-up benefits?",
    "What is the family floater coverage?"
]

cols = st.columns(3)
for i, sample in enumerate(sample_queries):
    with cols[i % 3]:
        if st.button(f"📝 {sample}", key=f"sample_{i}"):
            st.session_state.selected_query = sample
            st.rerun()
