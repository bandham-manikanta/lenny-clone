import streamlit as st
import sys
import os
import time
from pathlib import Path

# Add agent directory to path
agent_path = Path(__file__).parent.parent / "agent"
sys.path.insert(0, str(agent_path))

# Import with error handling
try:
    from rag import LennyRAG
except ImportError as e:
    st.error(f"Critical Error: Could not import LennyRAG. Path: {agent_path}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LennyBot | AI PM Coach",
    page_icon="🧢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-family: 'Inter', sans-serif; font-weight: 700; color: #333; }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #FF4B4B;
        padding: 15px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .source-title { font-weight: 600; color: #333; margin-bottom: 5px; }
    .source-snippet { font-style: italic; color: #555; font-size: 0.85rem; }
    .match-score {
        background-color: #e6ffe6;
        color: #006600;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! 👋 I'm **LennyBot**. \n\nI can help you with Product-Market Fit, Retention, Hiring, and Growth. What's on your mind?"}
    ]

@st.cache_resource(show_spinner=False)
def get_rag_engine():
    return LennyRAG()

try:
    rag = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://substackcdn.com/image/fetch/w_96,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F546947f2-7914-405e-9713-534d0f172476_1280x1280.png", width=60)
    st.markdown("### LennyBot Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)
    top_k = st.slider("Reference Depth", 3, 7, 5)
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.markdown("<h1 class='main-header'>🧢 LennyBot</h1>", unsafe_allow_html=True)
st.caption("Powered by **NVIDIA NIM**, **Llama 3.1**, and **Hybrid RAG**")

# 1. DISPLAY HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Check if this message has sources attached
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} Sources Analyzed"):
                for src in msg["sources"]:
                    score = int(src.get('score', 0) * 100)
                    source_type = "🎙️ Podcast" if "youtube" in src.get('type', '') else "📝 Post"
                    snippet = src.get('snippet', 'Content used for context.')
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-title">
                            {source_type} <span class="match-score">{score}% Match</span>
                        </div>
                        <div class="source-snippet">"...{snippet[:200]}..."</div>
                        <div style="margin-top:5px;">
                            <a href="{src.get('url', '#')}" target="_blank">🔗 View Source</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# 2. HANDLE INPUT
if prompt := st.chat_input("Ask Lenny anything..."):
    
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Consulting Lenny's knowledge base..."):
            try:
                # Run RAG
                result = rag.query_with_metadata(
                    question=prompt,
                    top_k=top_k,
                    temperature=temperature
                )
                
                response_text = result['response']
                chunks = result['chunks']
                
                # Process Sources
                enriched_sources = []
                seen_urls = set()
                for chunk in chunks:
                    url = chunk.get('source_url')
                    if url and url not in seen_urls:
                        enriched_sources.append({
                            "url": url,
                            "score": chunk.get('score', 0),
                            "type": chunk.get('source_type', 'unknown'),
                            "snippet": chunk.get('text', '')
                        })
                        seen_urls.add(url)

                # Display Response
                st.markdown(response_text)
                
                # Display Sources (This is the ONLY place they render for the new message)
                if enriched_sources:
                    with st.expander(f"📚 {len(enriched_sources)} Sources Analyzed"):
                        for src in enriched_sources:
                            score = int(src['score'] * 100)
                            source_type = "🎙️ Podcast" if "youtube" in src.get('type', '') else "📝 Post"
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-title">
                                    {source_type} <span class="match-score">{score}% Match</span>
                                </div>
                                <div class="source-snippet">"...{src['snippet'][:200]}..."</div>
                                <div style="margin-top:5px;">
                                    <a href="{src['url']}" target="_blank">🔗 View Source</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Save to History (So they appear in the loop next time)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": enriched_sources
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
