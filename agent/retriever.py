"""
Qdrant retriever for semantic search
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import hashlib
from qdrant_client.models import Filter, FieldCondition, MatchValue
import time

load_dotenv()


class QdrantRetriever:
    def __init__(self):
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'lenny_clone')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        print(f"🔧 Initializing QdrantRetriever...")
        start = time.time()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=10
        )
        print(f"   ✅ Qdrant client ready ({time.time()-start:.2f}s)")
        
        # Initialize embedding model
        start = time.time()
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device='cpu'
        )
        print(f"   ✅ Embedding model loaded ({time.time()-start:.2f}s)")
        
        # Embedding cache
        self._embedding_cache = {}
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict]:
        """Search with detailed timing"""
        
        total_start = time.time()
        
        # Step 1: Embed query
        embed_start = time.time()
        query_vector = self.embed_query(query)
        embed_time = time.time() - embed_start
        print(f"   ⏱️  Embedding: {embed_time:.3f}s")
        
        # Step 2: Search Qdrant
        search_start = time.time()
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            points = results.points
        except AttributeError:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            points = results
        
        search_time = time.time() - search_start
        print(f"   ⏱️  Qdrant search: {search_time:.3f}s")
        
        # Step 3: Format results
        format_start = time.time()
        formatted_results = []
        for result in points:
            formatted_results.append({
                'text': result.payload['text'],
                'source': result.payload['source'],
                'source_url': result.payload.get('url', result.payload.get('source_url', '')),
                'score': result.score,
                'metadata': result.payload.get('metadata', {}),
            })
        format_time = time.time() - format_start
        
        total_time = time.time() - total_start
        print(f"   ⏱️  Format: {format_time:.3f}s")
        print(f"   ✅ Total retrieval: {total_time:.3f}s ({len(formatted_results)} results)")
        
        return formatted_results
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query with caching"""
        import hashlib
        
        # Create cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._embedding_cache:
            print(f"   💾 Using cached embedding")
            return self._embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cache it
        self._embedding_cache[cache_key] = embedding.tolist()
        
        return self._embedding_cache[cache_key]
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Search for relevant chunks using optimized settings
        """
        # Embed query (now cached!)
        query_vector = self.embed_query(query)
        
        # Use query_points with optimizations
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            points = results.points
        except AttributeError:
            # Fallback to old API
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # ✅ Don't return vectors
            )
            points = results
        
        # Format results (same as before)
        formatted_results = []
        for result in points:
            formatted_results.append({
                'text': result.payload['text'],
                'source': result.payload['source'],
                'source_url': result.payload.get('url', result.payload.get('source_url', '')),
                'score': result.score,
                'metadata': result.payload.get('metadata', {}),
            })
        
        return formatted_results

    
    def search_with_filters(
        self,
        query: str,
        source_filter: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search with source filtering (e.g., only YouTube or only LinkedIn)
        """
        query_vector = self.embed_query(query)
        
        search_params = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        
        if source_filter:
            search_params["query_filter"] = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter)
                    )
                ]
            )
        
        try:
            results = self.client.query_points(**search_params)
            points = results.points
        except AttributeError:
            search_params["query_vector"] = search_params.pop("query")
            results = self.client.search(**search_params)
            points = results
        
        formatted_results = []
        for result in points:
            formatted_results.append({
                'text': result.payload['text'],
                'source': result.payload['source'],
                'source_url': result.payload.get('url', result.payload.get('source_url', '')),
                'score': result.score,
                'metadata': result.payload.get('metadata', {}),
            })
        
        return formatted_results


if __name__ == "__main__":
    # Test the retriever
    retriever = QdrantRetriever()
    
    print("\n🧪 Testing search...")
    
    test_query = "What is product-market fit?"
    results = retriever.search(test_query, top_k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Source: {result['source']} (score: {result['score']:.3f})")
        print(f"   Text: {result['text'][:150]}...")
        print(f"   URL: {result['source_url']}\n")
