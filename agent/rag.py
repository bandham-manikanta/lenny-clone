"""
agent/rag.py
Sophisticated RAG Pipeline: Dual-Stream Retrieval + Streaming Response
"""
from typing import Iterator, List, Dict, Optional
from agent.chroma_retriever import ChromaRetriever
from agent.persona import LennyPersona
from agent.llm_client import NvidiaLlamaClient

class LennyRAG:
    def __init__(self):
        print("🚀 Initializing Sophisticated Lenny Agent...")
        self.retriever = ChromaRetriever()
        self.persona = LennyPersona()
        self.llm = NvidiaLlamaClient()
    
    def _get_dual_stream_context(self, question: str, top_k: int):
        """Helper to perform the stratified retrieval"""
        # STREAM 1: Retrieve Lenny's Voice (Force LinkedIn)
        lenny_chunks = self.retriever.search_with_filters(
            query=question,
            source_filter="linkedin",
            top_k=3 # Force 3 Lenny posts
        )
        
        # STREAM 2: Retrieve Guest Evidence (Force YouTube)
        guest_chunks = self.retriever.search_with_filters(
            query=question,
            source_filter="youtube",
            top_k=3 # Force 3 Podcast clips
        )
        
        return lenny_chunks, guest_chunks

    def query(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = True,
        temperature: float = 0.7,
        source_filter: Optional[str] = None,
        max_tokens: int = 1000,
    ) -> str | Iterator[str]:
        """
        Main Query Method - Supports Streaming for UI
        """
        print(f"\n🔍 Sophisticated Query: {question}")
        
        # 1. Retrieval
        if source_filter:
            # If user manually filtered in UI, respect it
            lenny_chunks = []
            guest_chunks = self.retriever.search_with_filters(question, source_filter, top_k)
        else:
            # Default: Use Sophisticated Dual-Stream
            lenny_chunks, guest_chunks = self._get_dual_stream_context(question, top_k)
        
        all_chunks = lenny_chunks + guest_chunks
        
        if not all_chunks:
            msg = "I don't have enough data on that yet. Try asking about PMF or Retention!"
            if stream:
                yield msg
                return
            return msg
        
        # 2. Build Stratified Prompt
        system_prompt = self.persona.get_system_prompt()
        user_prompt = self.persona.get_stratified_prompt(
            question=question, 
            lenny_chunks=lenny_chunks, 
            guest_chunks=guest_chunks
        )
        
        # 3. Generate (Streaming)
        return self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
    
    def query_with_metadata(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Dict:
        """
        Used for the final UI display to show sources
        """
        # 1. Retrieve
        lenny_chunks, guest_chunks = self._get_dual_stream_context(question, top_k)
        all_chunks = lenny_chunks + guest_chunks
        
        # 2. Generate
        system_prompt = self.persona.get_system_prompt()
        user_prompt = self.persona.get_stratified_prompt(question, lenny_chunks, guest_chunks)
        
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            stream=False
        )
        
        # 3. Format Sources
        sources = []
        seen_urls = set()
        # Sort by score to show best relevance first
        sorted_chunks = sorted(all_chunks, key=lambda x: x['score'], reverse=True)
        
        for chunk in sorted_chunks:
            if chunk['source_url'] and chunk['source_url'] not in seen_urls:
                sources.append({
                    "type": chunk['source'],
                    "url": chunk['source_url'],
                    "score": chunk['score'],
                    "authority": "Lenny's Core Belief" if chunk['source'] == 'linkedin' else "Guest Case Study"
                })
                seen_urls.add(chunk['source_url'])
        
        return {
            "response": response,
            "sources": sources,
            "chunks": sorted_chunks
        }
