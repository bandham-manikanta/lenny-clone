"""
Main orchestration script - ChromaDB version
"""
import sys
from dotenv import load_dotenv
import json
load_dotenv()

from extract_youtube import extract_youtube_transcripts
from extract_linkedin import extract_linkedin_posts
from ingestion.process_data_old import DataProcessor
from load_chroma import ChromaLoader  # ✅ Changed from QdrantUploader

def run_full_pipeline(skip_extraction=False):
    print("\n🚀 LENNY CLONE - INGESTION (ChromaDB)\n")
    
    if not skip_extraction:
        # 1. YouTube
        print("📺 STEP 1: YouTube")
        # yt_data = extract_youtube_transcripts(limit=100)
        with open('data/youtube_transcripts.json', 'r', encoding='utf-8') as f:
            yt_data = json.load(f)
        
        # 2. LinkedIn
        print("\n💼 STEP 2: LinkedIn")
        # li_data = extract_linkedin_posts(limit=100)
        with open('data/linkedin_posts.json', 'r', encoding='utf-8') as f:
            li_data = json.load(f)

        
        if not yt_data and not li_data:
            print("⚠️  No new data collected, will process existing data if available")
    else:
        print("⏭️  Skipping extraction, using existing data...")
    
    # 3. Process
    print("\n🔧 STEP 3: Processing")
    processor = DataProcessor()
    chunks = processor.process_all()
    
    if not chunks:
        print("❌ No chunks generated.")
        return

    # 4. Upload to ChromaDB (instead of Qdrant)
    print("\n📤 STEP 4: Uploading to ChromaDB")
    loader = ChromaLoader()
    loader.run_pipeline(recreate_collection=True)
    
    print("\n✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to skip extraction
    skip_extraction = '--skip-extraction' in sys.argv
    
    run_full_pipeline(skip_extraction=skip_extraction)
