import streamlit as st
import sys
import os

# Add the root directory to sys.path so we can import 'agent'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.rag import LennyRAG

# Page Config
st.set_page_config(
    page_title="Lenny's PM Assistant",
    page_icon="🧢",
    layout="centered"
)

# Custom CSS for better source display
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .source-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 4px;
    }
    .source-text {
        color: #666;
        font-size: 0.9em;
        font-style: italic;
    }
    .match-score {
        color: #4CAF50;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi! I'm Lenny's AI clone. Ask me anything about product management, growth, or career advice!"
        }
    ]

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

if "rag" not in st.session_state:
    with st.spinner("Loading Lenny's Brain..."):
        try:
            st.session_state.rag = LennyRAG()
        except Exception as e:
            st.error(f"Failed to initialize RAG: {e}")

# Sidebar
with st.sidebar:
    st.image("https://substackcdn.com/image/fetch/w_96,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F546947f2-7914-405e-9713-534d0f172476_1280x1280.png", width=80)
    st.title("Lenny's Clone")
    st.markdown("Powered by **RAG + Llama 3.1**")
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, help="Higher = more creative, Lower = more focused")
    top_k = st.slider("Number of Sources", 3, 10, 5, help="How many relevant sources to retrieve")
    
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hi! I'm Lenny's AI clone. Ask me anything about product management, growth, or career advice!"
            }
        ]
        st.session_state.conversation_context = []
        st.rerun()

# Chat Interface
st.header("🧢 Ask Lenny")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show sources with snippets (only for assistant messages)
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources & Snippets", expanded=False):
                for i, source in enumerate(msg["sources"], 1):
                    # Source header
                    source_type = "🎙️ YouTube" if source['type'] == 'youtube' else "💼 LinkedIn"
                    match_pct = int(source['score'] * 100)
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-title">
                            {source_type} · <span class="match-score">{match_pct}% match</span>
                        </div>
                        <div class="source-text">
                            "{source.get('snippet', 'No snippet available')[:200]}..."
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if source.get('url'):
                        st.caption(f"[Open source]({source['url']})")

# Input
prompt = st.chat_input("Ask me anything about product management...")

# Process input
if prompt:
    if not prompt.strip():  # ✅ Add this
        st.warning("Please enter a question")
        st.stop()
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        try:
            import time
            
            # OPTIMIZED: Single query_with_metadata call (does both retrieval and generation)
            start_time = time.time()
            
            # Show timing
            progress_text = st.empty()
            progress_text.markdown("🔍 Searching knowledge base...")
            
            # Single call - more efficient
            metadata = st.session_state.rag.query_with_metadata(
                question=prompt,
                top_k=top_k,
                temperature=temperature
            )
            
            retrieval_time = time.time() - start_time
            
            sources = metadata.get('sources', [])
            chunks = metadata.get('chunks', [])
            response = metadata.get('response', '')
            
            # Clear progress and show response
            progress_text.empty()
            
            # Display response (already generated!)
            st.markdown(response)
            
            # Show timing info (optional - can remove in production)
            st.caption(f"⚡ Retrieved and generated in {retrieval_time:.2f}s")
            
            # Add snippets to sources
            for source in sources:
                matching_chunk = next(
                    (chunk for chunk in chunks if chunk.get('source_url') == source['url']),
                    None
                )
                if matching_chunk:
                    source['snippet'] = matching_chunk['text']
            
            # Save to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
            # Display sources
            if sources:
                with st.expander("📚 View Sources & Snippets", expanded=False):
                    for i, source in enumerate(sources, 1):
                        source_type = "🎙️ YouTube" if source['type'] == 'youtube' else "💼 LinkedIn"
                        match_pct = int(source['score'] * 100)
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-title">
                                {source_type} · <span class="match-score">{match_pct}% match</span>
                            </div>
                            <div class="source-text">
                                "{source.get('snippet', 'No snippet available')[:200]}..."
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if source.get('url'):
                            st.caption(f"[Open source]({source['url']})")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
