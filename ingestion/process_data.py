import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

class DataProcessor:
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def load_data(self) -> List[Dict]:
        documents = []
        
        # 1. Load Cleaned YouTube Data (Lenny vs Guest separated)
        try:
            with open('data/cleaned_transcripts.json', 'r', encoding='utf-8') as f:
                yt_data = json.load(f)
                print(f"📺 Loaded {len(yt_data)} cleaned YouTube segments")
                documents.extend(yt_data) # These already have 'source': 'youtube_lenny' or 'youtube_guest'
        except FileNotFoundError:
            print("⚠️ Cleaned transcripts not found. Run clean_transcripts.py first.")

        # 2. Load LinkedIn Data
        try:
            with open('data/linkedin_posts.json', 'r', encoding='utf-8') as f:
                li_data = json.load(f)
                print(f"💼 Loaded {len(li_data)} LinkedIn posts")
                for post in li_data:
                    documents.append({
                        "id": post.get("post_id"),
                        "text": post.get("text"),
                        "source": "linkedin",
                        "url": post.get("url")
                    })
        except FileNotFoundError:
            print("⚠️ LinkedIn data not found.")
            
        return documents

    def process_all(self):
        print("🚀 Processing Data...")
        documents = self.load_data()
        
        chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            text = doc.get('text', '')
            if not text: continue
            
            text_chunks = self.text_splitter.split_text(text)
            for i, t in enumerate(text_chunks):
                chunks.append({
                    'chunk_id': f"{doc['id']}_{i}",
                    'text': t,
                    'source': doc['source'], # Critical: keeps 'youtube_lenny' vs 'youtube_guest'
                    'source_url': doc['url'],
                    'metadata': {'chunk_index': i}
                })
                
        # Embed
        print("🧮 Generating Embeddings...")
        texts = [c['text'] for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            
        with open('data/processed_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
            
        print(f"✅ Saved {len(chunks)} processed chunks.")

if __name__ == "__main__":
    dp = DataProcessor()
    dp.process_all()
