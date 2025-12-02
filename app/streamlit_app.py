import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.rag import LennyRAG

st.set_page_config(
    page_title="LennyBot | AI PM Coach",
    page_icon="🧢",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! 👋 I'm **LennyBot**. What's on your mind?"}
    ]

@st.cache_resource(show_spinner=False)
def get_rag_engine():
    return LennyRAG()

try:
    rag = get_rag_engine()
except Exception as e:
    st.error(f"Failed to load RAG: {e}")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)
    top_k = st.slider("Sources", 3, 7, 5)
    if st.button("🗑️ Clear"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

st.markdown("# 🧢 LennyBot")
st.caption("Powered by **NVIDIA NIM** and **Llama 3.1**")

# DISPLAY HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src.get('authority', 'Source')}: {int(src['score']*100)}% match")
                    if src.get('url'):
                        st.caption(f"[View]({src['url']})")

# INPUT
if prompt := st.chat_input("Ask Lenny..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # STREAM the response token by token
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
            
            # Get sources (reuse retrieval without regenerating)
            lenny_chunks, guest_chunks = rag._get_dual_stream_context(prompt, top_k)
            all_chunks = lenny_chunks + guest_chunks
            
            # Format sources
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
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
