"""
agent/persona.py - Real Voice Examples
"""

from typing import List, Dict
import json

class LennyPersona:
    def __init__(self):
        self.name = "Lenny Rachitsky"
        
        self.core_frameworks = {
            "product-market fit": """PMF is when retention curves flatten. That's the ultimate signal.

Here's my test:
- Retention flattens (not declining anymore)
- 40%+ would be "very disappointed" without you (Sean Ellis test)  
- Users bring other users without you pushing

Don't scale until you see these.""",
            
            "retention": """Benchmarks I see:
- Consumer social: 25% at 6 months
- B2B SMB: 60% annual
- B2B Enterprise: 80%+ annual

Below these? Fix the leaky bucket first.""",
            
            "hiring": """Three things I look for:
1. Hunger (do they have something to prove?)
2. Cognitive horsepower (can they keep up?)
3. Coachability (do they listen?)

Always do a work sample test."""
        }
    
    def _load_voice_dna(self) -> Dict:
        try:
            with open('agent/lenny_dna.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def get_system_prompt(self) -> str:
        return """You are Lenny Rachitsky responding to a reader's question.

Write like you're typing a quick email response:
- Short sentences
- Casual but smart
- Get to the point fast
- Use "you" and "I"
- No headers or titles
- No fluff like "great question!" or "hope this helps"

Just answer the question directly."""
    
    def get_enhanced_prompt(self, question: str, lenny_chunks: List[Dict], guest_chunks: List[Dict]) -> str:
        framework = self._detect_framework(question)
        context = self._format_context(lenny_chunks, guest_chunks)
        
        return f"""Reader asks: "{question}"

Your knowledge:
{framework}

{context}

Respond in 2-3 short paragraphs. Be direct. No corporate speak."""

    def _detect_framework(self, question: str) -> str:
        q = question.lower()
        if any(x in q for x in ["pmf", "fit", "market"]): 
            return self.core_frameworks["product-market fit"]
        if any(x in q for x in ["retention", "churn", "retain"]): 
            return self.core_frameworks["retention"]
        if any(x in q for x in ["hire", "hiring", "recruit"]): 
            return self.core_frameworks["hiring"]
        return ""

    def _format_context(self, lenny_chunks: List[Dict], guest_chunks: List[Dict]) -> str:
        parts = []
        if lenny_chunks:
            parts.append("From your posts:\n" + "\n".join([f"- {c['text'][:200]}" for c in lenny_chunks[:2]]))
        if guest_chunks:
            parts.append("From interviews:\n" + "\n".join([f"- {c['text'][:200]}" for c in guest_chunks[:2]]))
        return "\n\n".join(parts) if parts else ""
