"""
Load processed chunks into ChromaDB (local vector DB)
"""

import os
import json
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class ChromaLoader:
    def __init__(self):
        self.collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'lenny_clone')
        self.persist_directory = "./chroma_db"
        
        print(f"🔧 Initializing ChromaLoader...")
        print(f"   Collection: {self.collection_name}")
        print(f"   Persist directory: {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
    
    def create_collection(self, recreate: bool = False):
        """Create or get collection"""
        
        if recreate:
            print(f"🗑️  Deleting existing collection...")
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"   ✅ Deleted")
            except:
                pass
        
        # Create new collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"✅ Collection ready: {self.collection_name}")
    
    def upload(self):
        """Upload processed chunks to ChromaDB"""
        
        # Load processed chunks
        try:
            with open('data/processed_chunks.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except FileNotFoundError:
            print("❌ processed_chunks.json not found. Run process_data.py first.")
            return
        
        if not chunks:
            print("❌ No chunks to upload")
            return
        
        print(f"📤 Uploading {len(chunks)} chunks...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk['text'])
            embeddings.append(chunk['embedding'])
            metadatas.append({
                'source': chunk['source'],
                'source_url': chunk.get('source_url', ''),
                'chunk_index': chunk.get('metadata', {}).get('chunk_index', 0),
            })
        
        # Upload in batches
        batch_size = 1000
        
        for i in tqdm(range(0, len(ids), batch_size), desc="Uploading"):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        print(f"✅ Upload complete!")
    
    def verify(self):
        """Verify the upload"""
        print("🔍 Verifying...")
        
        count = self.collection.count()
        print(f"   Total documents in collection: {count}")
        
        # Test search
        test_result = self.collection.query(
            query_embeddings=[[0.1] * 384],  # Dummy embedding
            n_results=1
        )
        
        if test_result['ids']:
            print(f"   ✅ Search test successful")
        else:
            print(f"   ⚠️  Search returned no results")
    
    def run_pipeline(self, recreate_collection: bool = True):
        """Run full pipeline"""
        self.create_collection(recreate=recreate_collection)
        self.upload()
        self.verify()


if __name__ == "__main__":
    loader = ChromaLoader()
    loader.run_pipeline(recreate_collection=True)
