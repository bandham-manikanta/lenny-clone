"""
ChromaDB retriever for local semantic search (FAST!)
"""

import os
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import hashlib

load_dotenv()


class ChromaRetriever:
    def __init__(self):
        self.collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'lenny_clone')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.persist_directory = "./chroma_db"
        
        print(f"🔧 Initializing ChromaRetriever...")
        
        # Initialize ChromaDB client (persistent)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"   ✅ Loaded existing collection: {self.collection_name}")
        except:
            # Collection doesn't exist yet
            self.collection = None
            print(f"   ⚠️  Collection not found. Run load_chroma.py first.")
        
        # Initialize embedding model
        print(f"📥 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device='cpu'
        )
        print(f"   ✅ Model loaded")
        
        # Embedding cache
        self._embedding_cache = {}
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query with caching"""
        # Create cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._embedding_cache:
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
        Search for relevant chunks
        """
        if not self.collection:
            print("❌ Collection not initialized")
            return []
        
        # Embed query
        query_vector = self.embed_query(query)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
        
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distance (lower is better)
            # Convert to similarity score (higher is better)
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            
            # Apply threshold
            if similarity < score_threshold:
                continue
            
            metadata = results['metadatas'][0][i]
            
            formatted_results.append({
                'text': results['documents'][0][i],
                'source': metadata.get('source', 'unknown'),
                'source_url': metadata.get('source_url', ''),
                'score': similarity,
                'metadata': metadata,
            })
        
        return formatted_results
    
    def search_with_filters(
        self,
        query: str,
        source_filter: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search with source filtering
        """
        if not self.collection:
            return []
        
        query_vector = self.embed_query(query)
        
        # Build filter
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        # results_copy = results.copy()
        # results = []
        # filtered_results = []
        # for rec in results_copy:
        #     # ChromaDB uses distance (lower is better). 
        #     # Threshold depends on metric (L2 vs Cosine). Assuming Cosine distance < 0.4 is good.
        #     doc, score = results_copy['documents'][0], results_copy['distances'][0]
        #     if score < 0.4: 
        #         results.append(rec)
        
        # Format results (same as above)
        formatted_results = []
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
        
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            metadata = results['metadatas'][0][i]
            
            formatted_results.append({
                'text': results['documents'][0][i],
                'source': metadata.get('source', 'unknown'),
                'source_url': metadata.get('source_url', ''),
                'score': similarity,
                'metadata': metadata,
            })
        
        return formatted_results


if __name__ == "__main__":
    # Test the retriever
    retriever = ChromaRetriever()
    
    if retriever.collection:
        print("\n🧪 Testing search...")
        
        test_query = "What is product-market fit?"
        results = retriever.search(test_query, top_k=3)
        
        print(f"\nQuery: {test_query}")
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Source: {result['source']} (score: {result['score']:.3f})")
            print(f"   Text: {result['text'][:150]}...")
            print(f"   URL: {result['source_url']}\n")
