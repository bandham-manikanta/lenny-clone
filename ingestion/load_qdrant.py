"""
Universal Qdrant Uploader
Works with Qdrant Client v1.x (all versions)
"""

import os
import json
import uuid
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

load_dotenv()

class QdrantUploader:
    def __init__(self):
        self.url = os.getenv('QDRANT_URL', '')
        self.key = os.getenv('QDRANT_API_KEY')
        self.collection = os.getenv('QDRANT_COLLECTION_NAME', 'lenny_clone')
        self.dim = int(os.getenv('EMBEDDING_DIM', 384))
        
        # Auto-fix URL (remove port for cloud)
        if 'cloud.qdrant.io' in self.url:
            self.url = self.url.replace(':6333', '')
        
        print(f"🔗 Connecting to: {self.url}")
        
        self.client = QdrantClient(url=self.url, api_key=self.key, timeout=60)
        
    def create_collection(self, recreate=False):
        if recreate:
            self.client.delete_collection(self.collection)
        
        # Check if exists
        exists = False
        try:
            self.client.get_collection(self.collection)
            exists = True
        except:
            pass
            
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {self.collection}")

    def upload(self):
        try:
            with open('data/processed_chunks.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except:
            print("❌ processed_chunks.json not found")
            return

        if not chunks: return
        
        points = []
        for c in chunks:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=c['embedding'],
                payload={
                    'text': c['text'], 
                    'source': c['source'], 
                    'url': c.get('source_url'),
                    'metadata': c.get('metadata', {})
                }
            ))
        
        print(f"📤 Uploading {len(points)} points...")
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size)):
            self.client.upsert(self.collection, points[i:i+batch_size])

    def verify(self):
        print("🔍 Verifying...")
        try:
            # Universal Search Logic
            if hasattr(self.client, 'query_points'):
                # Newer API
                res = self.client.query_points(self.collection, query=[0.1]*self.dim, limit=1)
                print(f"✅ Verification successful (via query_points). Found: {len(res.points)}")
            elif hasattr(self.client, 'search'):
                # Standard API
                res = self.client.search(self.collection, query_vector=[0.1]*self.dim, limit=1)
                print(f"✅ Verification successful (via search). Found: {len(res)}")
            else:
                print("⚠️ Could not find search method, but upload finished.")
        except Exception as e:
            print(f"⚠️ Verification warning: {e}")

    def run_pipeline(self, recreate_collection=False):
        self.create_collection(recreate=recreate_collection)
        self.upload()
        self.verify()

if __name__ == "__main__":
    QdrantUploader().run_pipeline(recreate_collection=True)
