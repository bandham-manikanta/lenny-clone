"""
Main RAG pipeline for Lenny clone
Combines retrieval, persona, and generation
"""
from typing import Iterator, List, Dict, Optional
from agent.chroma_retriever import ChromaRetriever  # ✅ Changed
from .persona import LennyPersona
from .llm_client import NvidiaLlamaClient

class LennyRAG:
    """RAG pipeline for Lenny Rachitsky clone"""
    
    def __init__(self):
        print("🚀 Initializing Lenny RAG pipeline...")
        
        self.retriever = ChromaRetriever()
        self.persona = LennyPersona()
        self.llm = NvidiaLlamaClient()
        
        print("✅ RAG pipeline ready")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = False,
        temperature: float = 0.7,
        source_filter: Optional[str] = None,
        max_tokens: int = 500,
    ) -> str | Iterator[str]:
        """
        Query the RAG system
        """
        print(f"\n🔍 Processing query: {question}")
        
        # Step 1: Retrieve relevant chunks
        if source_filter:
            chunks = self.retriever.search_with_filters(
                query=question,
                source_filter=source_filter,
                top_k=top_k
            )
        else:
            chunks = self.retriever.search(
                query=question,
                top_k=top_k
            )
        
        if not chunks:
            return "I don't have enough relevant information to answer that question accurately."
        
        # Step 2: Build prompt
        system_prompt = self.persona.get_system_prompt()
        user_prompt = self.persona.get_context_prompt(chunks, question)
        
        # Step 3: Generate response
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
        Query with metadata about sources
        """
        # Retrieve
        chunks = self.retriever.search(question, top_k=top_k)
        
        if not chunks:
            return {
                "response": "I don't have enough relevant information.",
                "sources": [],
                "chunks": []
            }
        
        # Generate
        system_prompt = self.persona.get_system_prompt()
        user_prompt = self.persona.get_context_prompt(chunks, question)
        
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            stream=False
        )
        
        # Extract sources
        sources = []
        seen_urls = set()
        for chunk in chunks:
            if chunk['source_url'] and chunk['source_url'] not in seen_urls:
                sources.append({
                    "type": chunk['source'],
                    "url": chunk['source_url'],
                    "score": chunk['score']
                })
                seen_urls.add(chunk['source_url'])
        
        return {
            "response": response,
            "sources": sources,
            "chunks": chunks
        }
