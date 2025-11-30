"""
Robust YouTube Extractor using yt-dlp + WebShare Static Proxy
"""

import os
import json
import time
import random
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Try to import yt-dlp
try:
    import yt_dlp
except ImportError:
    print("❌ yt-dlp not installed. Please run: uv pip install yt-dlp")
    exit(1)

load_dotenv()

def get_channel_videos_official_api(channel_id: str, max_results: int = 100) -> List[str]:
    """Get Video List via Official API"""
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("❌ YOUTUBE_API_KEY missing")
        return []
    
    try:
        from googleapiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_ids = []
        token = None
        print("📡 Connecting to YouTube Data API v3...")
        while len(video_ids) < max_results:
            req = youtube.search().list(
                part='id', channelId=channel_id, maxResults=50,
                order='date', type='video', pageToken=token
            )
            res = req.execute()
            for item in res.get('items', []):
                video_ids.append(item['id']['videoId'])
            token = res.get('nextPageToken')
            if not token: break
        print(f"✅ Found {len(video_ids)} videos via Official API")
        return video_ids[:max_results]
    except Exception as e:
        print(f"❌ API Error: {e}")
        return []

def clean_vtt_text(vtt_content: str) -> str:
    """Clean VTT file content to plain text"""
    lines = vtt_content.splitlines()
    text_lines = []
    timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}')
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if not line or line == 'WEBVTT': continue
        if 'Kind:' in line or 'Language:' in line: continue
        if timestamp_pattern.search(line) or line.isdigit(): continue
        clean_line = re.sub(r'<[^>]+>', '', line)
        # Remove duplicate lines (common in VTT)
        if clean_line and clean_line not in seen_lines:
            text_lines.append(clean_line)
            seen_lines.add(clean_line)
    return " ".join(text_lines)

def process_video_ytdlp(video_id: str, proxy_url: str) -> Dict:
    """
    Download transcript using yt-dlp (Robust against blocking)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_filename = f"temp_{video_id}_{random.randint(1000,9999)}"
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'outtmpl': temp_filename,
        'quiet': True,
        'no_warnings': True,
        'proxy': proxy_url,  # ✅ Use the working proxy
        
        # Anti-detection options
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        # Run yt-dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Find the downloaded .vtt file
        files = glob.glob(f"{temp_filename}*.vtt")
        
        if not files:
            return {"status": "failed", "error": "No transcript available", "video_id": video_id}
            
        # Read and clean
        with open(files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            
        full_text = clean_vtt_text(content)
        
        # Cleanup temp files
        for f in files:
            try: os.remove(f)
            except: pass
            
        if len(full_text) < 50:
             return {"status": "failed", "error": "Transcript empty", "video_id": video_id}

        return {
            "video_id": video_id,
            "url": url,
            "transcript": full_text,
            "status": "success"
        }

    except Exception as e:
        # Cleanup on error
        for f in glob.glob(f"{temp_filename}*"):
            try: os.remove(f)
            except: pass
            
        return {
            "video_id": video_id,
            "url": url,
            "error": str(e),
            "status": "failed"
        }

def extract_youtube_transcripts(limit: int = 100):
    # 1. Setup Proxy
    username = os.getenv('PROXY_USERNAME')
    password = os.getenv('PROXY_PASSWORD')
    proxy_host = os.getenv('PROXY_HOST', '142.111.48.253:7030')
    
    if not username or not password:
        print("❌ Proxy credentials missing")
        return
        
    # Construct the EXACT proxy URL that works
    proxy_url = f"http://{username}:{password}@{proxy_host}/"
    print(f"🛡️  Using Proxy: {proxy_host}")

    # 2. Get Videos
    channel_id = "UC6t1O76G0jYXOAoYCm153dA"
    video_ids = get_channel_videos_official_api(channel_id, limit)
    
    if not video_ids: return []

    results = []
    print(f"\n🚀 Downloading {len(video_ids)} transcripts (Using yt-dlp)...")
    print("   (This is slower than the API but rarely gets blocked)")
    
    # 3. Run in Parallel (5 Workers)
    # We use ThreadPoolExecutor because yt-dlp is blocking
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_vid = {
            executor.submit(process_video_ytdlp, vid, proxy_url): vid 
            for vid in video_ids
        }
        
        # Simple progress counter
        total = len(video_ids)
        completed = 0
        
        for future in as_completed(future_to_vid):
            res = future.result()
            completed += 1
            results.append(res)
            
            if res['status'] == 'success':
                success_count += 1
                print(f"   [{completed}/{total}] ✅ {res['video_id']} Success")
            else:
                print(f"   [{completed}/{total}] ❌ {res['video_id']} Failed: {res.get('error')[:50]}...")
    
    print(f"\n✅ Extracted {success_count}/{len(video_ids)} transcripts")
    
    # 4. Save Results
    os.makedirs('data', exist_ok=True)
    with open('data/youtube_transcripts.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    extract_youtube_transcripts(limit=100)
