"""
agent/persona.py - Conversational Spoken Style
"""

from typing import List, Dict
import json

class LennyPersona:
    def __init__(self):
        self.name = "Lenny Rachitsky"
        
        self.core_frameworks = {
            "product-market fit": """PMF = retention curves flatten. That's it.

My test:
- Retention stops declining
- 40%+ say they'd be "very disappointed" without you
- Users bring their friends without you asking

Don't scale until you see this.""",
            
            "retention": """Here's what I see:
- Consumer social: 25% at 6 months is good
- B2B SMB: shoot for 60% annual
- Enterprise: 80%+ annual

If you're below this, fix retention before spending on growth.""",
            
            "hiring": """Three things:
1. Hunger - something to prove
2. Smarts - can they keep up
3. Coachable - do they listen

And always do a work sample."""
        }
    
    def get_system_prompt(self) -> str:
        return """You are Lenny responding to a reader's email.

Write like you're talking to a friend:
- Use contractions (you're, don't, it's)
- Vary sentence length 
- Sometimes start with "Look" or "Here's the thing"
- Be opinionated but humble
- No formality, no fluff
- Just help them"""
    
    def get_enhanced_prompt(self, question: str, lenny_chunks: List[Dict], guest_chunks: List[Dict]) -> str:
        framework = self._detect_framework(question)
        context = self._format_context(lenny_chunks, guest_chunks)
        
        return f"""Question: {question}

What you know:
{framework}

{context}

Write 2-3 paragraphs. Sound like you're talking, not writing an essay."""

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
            parts.append("From your writing:\n" + "\n".join([f"- {c['text'][:180]}" for c in lenny_chunks[:2]]))
        if guest_chunks:
            parts.append("From guests:\n" + "\n".join([f"- {c['text'][:180]}" for c in guest_chunks[:2]]))
        return "\n\n".join(parts) if parts else ""
