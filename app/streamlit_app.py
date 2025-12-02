import streamlit as st
import sys
from pathlib import Path

# ✅ Fix: Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports work correctly
from agent.rag import LennyRAG

st.set_page_config(
    page_title="LennyBot | AI PM Coach",
    page_icon="🧢",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! 👋 I'm **LennyBot**. What's on your mind?"}
    ]

# Load RAG engine with error handling
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    try:
        return LennyRAG()
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        raise

# Initialize RAG
try:
    with st.spinner("Loading LennyBot..."):
        rag = get_rag_engine()
    st.sidebar.success("✅ LennyBot Ready")
except Exception as e:
    st.error("❌ Failed to load LennyBot")
    st.error(f"Error: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)
    top_k = st.slider("Sources", 3, 7, 5)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Main UI
st.markdown("# 🧢 LennyBot")
st.caption("Powered by **NVIDIA NIM** and **Llama 3.1**")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src.get('authority', 'Source')}: {int(src['score']*100)}% match")
                    if src.get('url'):
                        st.caption(f"[View]({src['url']})")

# Chat input
if prompt := st.chat_input("Ask Lenny..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream response
            for chunk in rag.query(
                question=prompt,
                top_k=top_k,
                temperature=temperature,
                stream=True
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Remove cursor
            message_placeholder.markdown(full_response)
            
            # Get sources
            lenny_chunks, guest_chunks = rag._get_dual_stream_context(prompt, top_k)
            all_chunks = lenny_chunks + guest_chunks
            
            sources = []
            seen_urls = set()
            for chunk in sorted(all_chunks, key=lambda x: x['score'], reverse=True):
                url = chunk.get('source_url')
                if url and url not in seen_urls:
                    sources.append({
                        "type": chunk['source'],
                        "url": url,
                        "score": chunk['score'],
                        "authority": "Lenny's Core Belief" if chunk['source'] == 'linkedin' else "Guest Case Study"
                    })
                    seen_urls.add(url)
            
            # Display sources
            if sources:
                with st.expander(f"📚 {len(sources)} Sources"):
                    for src in sources:
                        st.write(f"- {src['authority']}: {int(src['score']*100)}% match")
                        if src.get('url'):
                            st.caption(f"[View]({src['url']})")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
