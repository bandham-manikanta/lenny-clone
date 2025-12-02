import json
import re
import os

def is_likely_lenny(text, index, total_chunks):
    text = text.lower()
    
    # 1. The Intro (First 5% of text is usually Lenny)
    if index < (total_chunks * 0.05):
        return True
        
    # 2. Signature Phrases
    if "welcome to the podcast" in text: return True
    if "i'm lenny" in text: return True
    if "today's guest" in text: return True
    
    # 3. Short Questions (Lenny asks, Guests answer long)
    # If it ends in a question mark and is short (< 200 chars)
    if "?" in text and len(text) < 200:
        return True
        
    return False

def clean_data():
    print("🧹 Cleaning YouTube Transcripts (Separating Lenny vs Guests)...")
    
    with open('data/youtube_transcripts.json', 'r', encoding='utf-8') as f:
        videos = json.load(f)
        
    cleaned_data = []
    
    for video in videos:
        if video.get('status') != 'success': continue
        
        transcript = video['transcript']
        # Split by periods/questions to approximate turns
        chunks = re.split(r'(?<=[.?!])\s+', transcript)
        
        lenny_only_text = ""
        guest_text = ""
        
        for i, chunk in enumerate(chunks):
            if is_likely_lenny(chunk, i, len(chunks)):
                lenny_only_text += chunk + " "
            else:
                guest_text += chunk + " "
                
        # We save TWO versions. 
        # 1. Lenny's Voice (Primary Source)
        cleaned_data.append({
            "id": video['video_id'] + "_lenny",
            "text": lenny_only_text,
            "source": "youtube_lenny", # SPECIAL TAG
            "url": video['url']
        })
        
        # 2. Guest Context (Secondary Source - use only for examples)
        cleaned_data.append({
            "id": video['video_id'] + "_guest",
            "text": guest_text,
            "source": "youtube_guest", # SPECIAL TAG
            "url": video['url']
        })

    with open('data/cleaned_transcripts.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)
        
    print(f"✅ Processed {len(videos)} videos into Lenny vs Guest streams.")

if __name__ == "__main__":
    clean_data()
