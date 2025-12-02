# extract_style_nim.py (Fixed Version)

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
LINKEDIN_FILE = "data/linkedin_posts.json"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def load_pure_lenny_data():
    """Load LinkedIn posts for style analysis"""
    combined_text = ""
    if os.path.exists(LINKEDIN_FILE):
        with open(LINKEDIN_FILE, 'r', encoding='utf-8') as f:
            posts = json.load(f)
            posts.sort(key=lambda x: len(x.get('text', '')), reverse=True)
            for p in posts[:15]:
                combined_text += f"{p.get('text', '')}\n\n"
    return combined_text

def generate_persona_definition(text_samples):
    print("🧬 Analyzing Lenny's Linguistic DNA...")
    
    prompt = f"""
You are an expert Computational Linguist and Prompt Engineer.

Analyze the text below and extract Lenny Rachitsky's linguistic signatures.

OUTPUT ONLY VALID JSON in this exact format:
{{
    "vocabulary_whitelist": ["word1", "word2", "word3"],
    "rhetorical_patterns": ["pattern1", "pattern2"],
    "formatting_rules": ["rule1", "rule2"],
    "tone_descriptors": ["descriptor1", "descriptor2"]
}}

Find his most distinctive words, phrases, and structural patterns.

TEXT DATA:
{text_samples[:15000]}
"""

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            stream=False
        )
        
        response = completion.choices[0].message.content
        
        # Try to extract JSON if wrapped in markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        # Validate JSON
        parsed = json.loads(response)
        return parsed
        
    except Exception as e:
        print(f"❌ Error generating DNA: {e}")
        return get_fallback_dna()

def get_fallback_dna():
    """Fallback DNA based on manual analysis"""
    return {
        "vocabulary_whitelist": [
            "sticky", "superpower", "leverage", "bucket", "iterate",
            "aha moment", "north star", "flywheel", "unlock", "double down",
            "PMF", "retention curves", "cohort", "benchmark"
        ],
        "rhetorical_patterns": [
            "Think about it:",
            "Here's the thing:",
            "What I've found:",
            "My advice?",
            "Bottom line:",
            "I loved what [X] said about..."
        ],
        "formatting_rules": [
            "Use short paragraphs (1-3 sentences)",
            "Use bold for emphasis",
            "Ask rhetorical questions",
            "End with tactical advice",
            "Use bullet points for lists"
        ],
        "tone_descriptors": [
            "enthusiastic but humble",
            "data-driven",
            "conversational",
            "tactical"
        ]
    }

if __name__ == "__main__":
    data = load_pure_lenny_data()
    
    if not data:
        print("❌ No LinkedIn data found. Using fallback DNA.")
        analysis = get_fallback_dna()
    else:
        print(f"✅ Loaded {len(data)} chars of Pure Lenny data.")
        analysis = generate_persona_definition(data)
    
    # Ensure directory exists
    os.makedirs("agent", exist_ok=True)
    
    # Save with proper formatting
    with open("agent/lenny_dna.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Analysis saved to 'agent/lenny_dna.json'")
    print(json.dumps(analysis, indent=2))
