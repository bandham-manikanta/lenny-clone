import streamlit as st
import sys
from pathlib import Path

agent_path = Path(__file__).parent.parent / "agent"
sys.path.insert(0, str(agent_path))

from rag import LennyRAG

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
    st.sidebar.success("✅ RAG Engine Loaded")
except Exception as e:
    st.error(f"Failed to load RAG: {e}")
    st.stop()

# SIDEBAR
with st.sidebar:
    temperature = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)
    top_k = st.slider("Sources", 3, 7, 5)
    if st.button("🗑️ Clear"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

st.markdown("# 🧢 LennyBot")

# DISPLAY HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT
if prompt := st.chat_input("Ask Lenny..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            st.write("🔍 Starting query...")
            
            # First, try NON-STREAMING (to see if it works at all)
            result = rag.query_with_metadata(
                question=prompt,
                top_k=top_k,
                temperature=temperature,
                stream=False
            )
            
            st.write("✅ Got result")
            
            response_text = result.get('response', 'No response')
            sources = result.get('sources', [])
            
            st.markdown(response_text)
            
            if sources:
                st.write(f"📚 Found {len(sources)} sources")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
