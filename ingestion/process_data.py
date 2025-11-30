"""
Process, clean, chunk, and embed the extracted data
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

load_dotenv()


class DataProcessor:
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        
        print(f"🔧 Initializing DataProcessor...")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"   Chunk overlap: {self.chunk_overlap}")
        print(f"   Embedding model: {self.embedding_model_name}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize embedding model
        print(f"📥 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print(f"✅ Embedding model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep punctuation
        # text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
        
        return text.strip()
    
    def load_youtube_data(self) -> List[Dict]:
        """Load YouTube transcripts"""
        try:
            with open('data/youtube_transcripts.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter successful extractions
            successful = [d for d in data if d.get('status') == 'success']
            print(f"📺 Loaded {len(successful)} YouTube transcripts")
            return successful
        
        except FileNotFoundError:
            print("⚠️  YouTube transcripts not found. Run extract_youtube.py first.")
            return []
    
    def load_linkedin_data(self) -> List[Dict]:
        """Load LinkedIn posts"""
        try:
            with open('data/linkedin_posts.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"💼 Loaded {len(data)} LinkedIn posts")
            return data
        
        except FileNotFoundError:
            print("⚠️  LinkedIn posts not found. Run extract_linkedin.py first.")
            return []
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents into smaller pieces"""
        print("✂️  Chunking documents...")
        
        chunks = []
        
        for doc in tqdm(documents, desc="Chunking"):
            text = doc.get('text', doc.get('transcript', ''))
            text = self.clean_text(text)
            
            if not text:
                continue
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Create chunk documents
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'chunk_id': f"{doc.get('source', 'unknown')}_{doc.get('id', 'unknown')}_{i}",
                    'text': chunk_text,
                    'source': doc.get('source'),
                    'source_url': doc.get('url'),
                    'metadata': {
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'original_id': doc.get('id'),
                        'timestamp': doc.get('timestamp', ''),
                    }
                }
                chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks"""
        print("🧮 Generating embeddings...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding.tolist()
        
        print(f"✅ Generated {len(all_embeddings)} embeddings")
        return chunks
    
    def process_all(self) -> List[Dict]:
        """Process all data: load, clean, chunk, embed"""
        
        print("\n" + "="*60)
        print("🚀 Starting data processing pipeline")
        print("="*60 + "\n")
        
        # Load data
        youtube_data = self.load_youtube_data()
        linkedin_data = self.load_linkedin_data()
        
        # Prepare documents
        documents = []
        
        # YouTube transcripts
        for video in youtube_data:
            documents.append({
                'id': video['video_id'],
                'source': 'youtube',
                'url': video['url'],
                'transcript': video['transcript'],
                'text': video['transcript'],
            })
        
        # LinkedIn posts
        for post in linkedin_data:
            documents.append({
                'id': post.get('post_id', ''),
                'source': 'linkedin',
                'url': post.get('url', ''),
                'text': post.get('text', ''),
                'timestamp': post.get('timestamp', ''),
            })
        
        print(f"📊 Total documents: {len(documents)}")
        print(f"   YouTube: {len(youtube_data)}")
        print(f"   LinkedIn: {len(linkedin_data)}\n")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        
        with open('data/processed_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, indent=2)
        
        print(f"\n💾 Saved processed chunks to: data/processed_chunks.json")
        print(f"   Total chunks: {len(chunks_with_embeddings)}")
        print(f"   Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
        
        return chunks_with_embeddings


if __name__ == "__main__":
    processor = DataProcessor()
    chunks = processor.process_all()
    
    print("\n✅ Data processing complete!")
