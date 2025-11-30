"""
Streamlit UI for Lenny Rachitsky Clone
"""

import sys
import os
from pathlib import Path

# Add agent directory to path
agent_path = Path(__file__).parent.parent / "agent"
sys.path.insert(0, str(agent_path))

import streamlit as st
from typing import List, Dict

# Import with error handling
try:
    from rag import LennyRAG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure the agent module is in the correct path")
    st.stop()


# Page config
st.set_page_config(
    page_title="Chat with Lenny Rachitsky",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_rag():
    """Initialize RAG pipeline (cached)"""
    with st.spinner("🚀 Initializing Lenny AI..."):
        try:
            return LennyRAG()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.info("Check environment variables and Qdrant connection")
            return None


def display_sources(sources: List[Dict]):
    """Display source citations"""
    if not sources:
        return
    
    with st.expander("📚 Sources Used", expanded=False):
        for i, source in enumerate(sources, 1):
            source_type = "🎙️ Podcast" if source['type'] == 'youtube' else "💼 LinkedIn"
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{source_type}**")
                if source['url']:
                    st.markdown(f"[View source]({source['url']})")
            
            with col2:
                st.metric("Relevance", f"{source['score']:.0%}")
            
            if i < len(sources):
                st.divider()


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🎙️ Chat with Lenny Rachitsky</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask anything about product management, growth, and building products</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        top_k = st.slider(
            "Number of Sources",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of relevant chunks to retrieve"
        )
        
        source_filter = st.selectbox(
            "Source Filter",
            options=["All Sources", "YouTube Only", "LinkedIn Only"],
            index=0
        )
        
        source_filter_value = None
        if source_filter == "YouTube Only":
            source_filter_value = "youtube"
        elif source_filter == "LinkedIn Only":
            source_filter_value = "linkedin"
        
        st.divider()
        
        st.markdown("### 📖 About")
        st.markdown("""
        This AI is trained on:
        - 🎙️ **100 YouTube podcasts**
        - 💼 **100 LinkedIn posts**
        
        Uses RAG for grounded responses.
        """)
        
        st.divider()
        
        st.markdown("### 💡 Example Questions")
        examples = [
            "What is product-market fit?",
            "How do I improve retention?",
            "What makes a great PM?",
            "How should I price my product?",
            "What are growth loops?",
        ]
        
        for q in examples:
            if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
                st.session_state.example_q = q
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize RAG
    rag = initialize_rag()
    
    if rag is None:
        st.error("❌ Failed to initialize. Check configuration.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                display_sources(msg["sources"])
    
    # Handle input
    user_input = None
    
    if "example_q" in st.session_state:
        user_input = st.session_state.example_q
        del st.session_state.example_q
    else:
        user_input = st.chat_input("Ask Lenny anything...")
    
    # Process input
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Streaming response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in rag.query(
                    question=user_input,
                    top_k=top_k,
                    stream=True,
                    temperature=temperature,
                    source_filter=source_filter_value
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Get sources
                result = rag.query_with_metadata(
                    question=user_input,
                    top_k=top_k,
                    temperature=temperature
                )
                
                display_sources(result['sources'])
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": result['sources']
                })
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Please try again")


if __name__ == "__main__":
    main()
