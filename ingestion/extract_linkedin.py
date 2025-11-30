"""
Extract LinkedIn posts from Lenny Rachitsky's profile using Apify
"""

import os
import json
from typing import List, Dict
from apify_client import ApifyClient
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def extract_linkedin_posts(limit: int = 100) -> List[Dict]:
    """
    Extract LinkedIn posts using Apify
    """
    
    print("💼 Extracting LinkedIn posts from Lenny Rachitsky...")
    
    apify_api_key = os.getenv('APIFY_API_KEY')
    if not apify_api_key or apify_api_key == 'PLACEHOLDER':
        print("❌ APIFY_API_KEY not set")
        return []
    
    client = ApifyClient(apify_api_key)
    profile_url = "https://www.linkedin.com/in/lennyrachitsky/"
    
    # Using the actor you found
    actor_name = "apimaestro/linkedin-profile-posts"
    
    print(f"🚀 Starting Apify scraper: {actor_name}")
    
    try:
        # Input schema for apimaestro/linkedin-profile-posts
        run_input = {
            "linkedinUrl": profile_url,
            "maxPosts": limit,
            "proxy": {"useApifyProxy": True}
        }
        
        run = client.actor(actor_name).call(run_input=run_input)
        
        # Fetch results
        posts = []
        print("📥 Fetching results...")
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            posts.append(item)
        
        if posts:
            cleaned_posts = []
            for post in tqdm(posts[:limit], desc="Processing posts"):
                cleaned_post = {
                    "post_id": post.get("urn", post.get("id", "")),
                    "url": post.get("url", ""),
                    "text": post.get("text", post.get("textContent", "")),
                    "timestamp": post.get("time", {}).get("date", ""),
                    "likes": post.get("likes", 0),
                    "status": "success"
                }
                cleaned_posts.append(cleaned_post)
            
            os.makedirs('data', exist_ok=True)
            with open('data/linkedin_posts.json', 'w', encoding='utf-8') as f:
                json.dump(cleaned_posts, f, indent=2)
            
            print(f"\n✅ Successfully extracted {len(cleaned_posts)} posts")
            return cleaned_posts
            
    except Exception as e:
        print(f"❌ Apify error: {e}")

    # Fallback if Apify fails/runs out of credits
    print("\n⚠️  Apify failed. Using sample data for testing...")
    return extract_linkedin_manual_fallback(profile_url, limit)

def extract_linkedin_manual_fallback(profile_url: str, limit: int) -> List[Dict]:
    sample_posts = []
    for i in range(10):
        sample_posts.append({
            "post_id": f"sample_{i}",
            "url": f"{profile_url}/posts/sample_{i}",
            "text": f"Sample LinkedIn post {i} about product management and growth.",
            "author": "Lenny Rachitsky",
            "status": "sample"
        })
    
    os.makedirs('data', exist_ok=True)
    with open('data/linkedin_posts.json', 'w', encoding='utf-8') as f:
        json.dump(sample_posts, f, indent=2)
    return sample_posts

if __name__ == "__main__":
    extract_linkedin_posts(limit=10)
