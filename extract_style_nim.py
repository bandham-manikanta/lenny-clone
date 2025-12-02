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
    """
    We only load LinkedIn posts for style analysis because 
    YouTube transcripts contain guests. LinkedIn is 100% Lenny.
    """
    combined_text = ""
    if os.path.exists(LINKEDIN_FILE):
        with open(LINKEDIN_FILE, 'r', encoding='utf-8') as f:
            posts = json.load(f)
            # Sort by length to get his deep-dive thoughts
            posts.sort(key=lambda x: len(x.get('text', '')), reverse=True)
            # Take top 15 posts (approx 15k chars)
            for p in posts[:15]:
                combined_text += f"{p.get('text', '')}\n\n"
    return combined_text

def generate_persona_definition(text_samples):
    print("🧬 Analyzing Lenny's Linguistic DNA...")
    
    prompt = f"""
    You are an expert Computational Linguist and Prompt Engineer.
    
    OBJECTIVE:
    I need to create a System Prompt for an LLM to roleplay as "Lenny Rachitsky".
    Below is a large sample of his actual writing.
    
    TASK:
    Analyze the text below and extract his specific linguistic signatures.
    
    1. **Rhetorical Devices:** Does he use "Think about it", rhetorical questions, or specific transition words?
    2. **Formatting:** Does he use bullet points, emojis, short paragraphs?
    3. **Tone:** Is he authoritative, humble, enthusiastic, tactical?
    4. **Vocabulary:** List his top 10 most distinct keywords (e.g., "PMF", "Sticky").
    
    OUTPUT FORMAT (JSON ONLY):
    {{
        "system_prompt_instructions": "A list of 5-7 strict instructions for the LLM to follow to sound like him.",
        "few_shot_examples": [
            {{
                "user": "How do I improve retention?",
                "assistant": "Write a response in Lenny's exact style based on the analysis."
            }}
        ],
        "vocabulary_whitelist": ["word1", "word2"]
    }}

    TEXT DATA:
    {text_samples[:15000]}
    """

    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, # Low temp for analytical precision
        max_tokens=2048,
        stream=False
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    data = load_pure_lenny_data()
    if not data:
        print("❌ No LinkedIn data found. Run 'ingestion/extract_linkedin.py' first.")
    else:
        print(f"✅ Loaded {len(data)} chars of Pure Lenny data.")
        analysis = generate_persona_definition(data)
        
        # Save this to a file so our Agent can read it
        with open("agent/lenny_dna.json", "w", encoding="utf-8") as f:
            f.write(analysis)
            
        print("\n✅ Analysis saved to 'agent/lenny_dna.json'")
        print(analysis)
