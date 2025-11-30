"""
Evaluation metrics for synthetic chat quality
Based on BAIR and Stanford HAI research
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class PersonaEvaluator:
    """
    Evaluate persona consistency and quality
    
    Based on:
    - BAIR: Generative Agents (persona behavioral consistency)
    - Stanford HAI: Virtual persona evaluation frameworks
    """
    
    def __init__(self, reference_texts: List[str] = None):
        """
        Args:
            reference_texts: Real Lenny content for comparison
        """
        self.reference_texts = reference_texts or []
        
        # Lenny's characteristic phrases (learned from content)
        self.lenny_markers = [
            r"in my experience",
            r"what i've learned",
            r"the data shows",
            r"here's what",
            r"one framework",
            r"let me share",
            r"this is counterintuitive",
            r"the best companies",
        ]
    
    def factual_grounding_score(
        self,
        response: str,
        retrieved_chunks: List[Dict]
    ) -> float:
        """
        Measure how well response is grounded in retrieved content
        
        Returns score between 0 and 1
        """
        if not retrieved_chunks:
            return 0.0
        
        response_lower = response.lower()
        
        # Count overlapping key phrases
        total_overlap = 0
        total_chunks = len(retrieved_chunks)
        
        for chunk in retrieved_chunks:
            chunk_text = chunk['text'].lower()
            
            # Extract key phrases (simple: 3-5 word sequences)
            chunk_phrases = self._extract_key_phrases(chunk_text)
            
            # Count how many appear in response
            overlap = sum(1 for phrase in chunk_phrases if phrase in response_lower)
            total_overlap += overlap / max(len(chunk_phrases), 1)
        
        # Normalize
        score = total_overlap / total_chunks
        return min(score, 1.0)
    
    def persona_consistency_score(self, response: str) -> Dict[str, float]:
        """
        Measure consistency with Lenny's persona
        
        Returns:
            Dict with different consistency metrics
        """
        response_lower = response.lower()
        
        # 1. Linguistic marker usage
        marker_count = sum(
            1 for marker in self.lenny_markers
            if re.search(marker, response_lower)
        )
        marker_score = min(marker_count / 3, 1.0)  # Expect ~3 markers
        
        # 2. Tone analysis (practical, data-driven, humble)
        tone_indicators = {
            'practical': [r'\bactionable\b', r'\bpractical\b', r'\bspecific\b', r'\bconcrete\b'],
            'data_driven': [r'\bdata\b', r'\bresearch\b', r'\bstud(y|ies)\b', r'\bevidence\b'],
            'humble': [r'\bin my view\b', r'\bi think\b', r'\bperhaps\b', r'\bmight\b', r'\bcould\b'],
        }
        
        tone_scores = {}
        for tone, patterns in tone_indicators.items():
            count = sum(1 for pattern in patterns if re.search(pattern, response_lower))
            tone_scores[tone] = min(count / 2, 1.0)
        
        # 3. Structure (has examples, frameworks, or takeaways)
        structure_score = 0.0
        if re.search(r'for example|here\'s an example|let me share', response_lower):
            structure_score += 0.33
        if re.search(r'framework|approach|model|method', response_lower):
            structure_score += 0.33
        if re.search(r'takeaway|key point|important|remember', response_lower):
            structure_score += 0.34
        
        return {
            'linguistic_markers': marker_score,
            'tone_practical': tone_scores.get('practical', 0),
            'tone_data_driven': tone_scores.get('data_driven', 0),
            'tone_humble': tone_scores.get('humble', 0),
            'structure': structure_score,
            'overall': np.mean([
                marker_score,
                np.mean(list(tone_scores.values())),
                structure_score
            ])
        }
    
    def response_quality_score(self, response: str, question: str) -> Dict[str, float]:
        """
        Measure general response quality
        
        Returns:
            Dict with quality metrics
        """
        
        # 1. Completeness (response length relative to question complexity)
        question_words = len(question.split())
        response_words = len(response.split())
        
        # Expect ~10-20x response length
        expected_ratio = 15
        length_score = min(response_words / (question_words * expected_ratio), 1.0)
        
        # 2. Specificity (contains specific numbers, names, companies)
        specificity_count = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))  # Names
        specificity_count += len(re.findall(r'\d+%|\d+x', response))  # Percentages/multiples
        specificity_score = min(specificity_count / 5, 1.0)
        
        # 3. Coherence (simple: check for proper sentences)
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s for s in sentences if len(s.split()) >= 5]
        coherence_score = len(valid_sentences) / max(len(sentences), 1)
        
        # 4. Relevance (question keywords appear in response)
        question_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
        response_keywords = set(re.findall(r'\b\w{4,}\b', response.lower()))
        overlap = question_keywords & response_keywords
        relevance_score = len(overlap) / max(len(question_keywords), 1)
        
        return {
            'length': length_score,
            'specificity': specificity_score,
            'coherence': coherence_score,
            'relevance': relevance_score,
            'overall': np.mean([length_score, specificity_score, coherence_score, relevance_score])
        }
    
    def citation_accuracy(
        self,
        response: str,
        retrieved_chunks: List[Dict]
    ) -> float:
        """
        Check if claims in response can be traced to sources
        
        Returns score between 0 and 1
        """
        # Extract claims (sentences with factual statements)
        sentences = re.split(r'[.!?]+', response)
        factual_sentences = [
            s for s in sentences
            if any(marker in s.lower() for marker in ['research', 'study', 'data', 'companies', 'found'])
        ]
        
        if not factual_sentences:
            return 1.0  # No claims to verify
        
        # Check if each claim has support in chunks
        supported_count = 0
        
        for sentence in factual_sentences:
            sentence_lower = sentence.lower()
            
            # Check if any chunk supports this claim
            for chunk in retrieved_chunks:
                chunk_lower = chunk['text'].lower()
                
                # Simple overlap check
                key_words = set(re.findall(r'\b\w{5,}\b', sentence_lower))
                chunk_words = set(re.findall(r'\b\w{5,}\b', chunk_lower))
                
                overlap_ratio = len(key_words & chunk_words) / max(len(key_words), 1)
                
                if overlap_ratio > 0.3:  # At least 30% overlap
                    supported_count += 1
                    break
        
        return supported_count / len(factual_sentences)
    
    def _extract_key_phrases(self, text: str, n: int = 4) -> List[str]:
        """Extract key n-word phrases"""
        words = text.lower().split()
        phrases = []
        
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            phrases.append(phrase)
        
        return phrases
    
    def evaluate_response(
        self,
        question: str,
        response: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Complete evaluation of a response
        
        Returns:
            Dict with all metrics
        """
        
        return {
            'factual_grounding': self.factual_grounding_score(response, retrieved_chunks),
            'persona_consistency': self.persona_consistency_score(response),
            'response_quality': self.response_quality_score(response, question),
            'citation_accuracy': self.citation_accuracy(response, retrieved_chunks),
        }


if __name__ == "__main__":
    # Test evaluator
    evaluator = PersonaEvaluator()
    
    test_response = """
    In my experience, product-market fit is when your product solves a real problem 
    so well that people are willing to pay for it. The data shows that companies with 
    strong PMF see organic growth and high retention. Here's what I've learned: 
    you know you have PMF when users would be "very disappointed" if your product went away.
    
    One framework that's helpful is the Sean Ellis test - if 40% or more of users would be 
    very disappointed, you likely have PMF. Companies like Airbnb and Slack hit this threshold 
    early on.
    """
    
    test_question = "What is product-market fit?"
    
    test_chunks = [
        {'text': 'Product-market fit is about solving a real problem that people care about.'},
        {'text': 'The Sean Ellis test uses 40% as the threshold for strong product-market fit.'}
    ]
    
    results = evaluator.evaluate_response(test_question, test_response, test_chunks)
    
    print("📊 Evaluation Results:\n")
    print(f"Factual Grounding: {results['factual_grounding']:.2%}")
    print(f"Citation Accuracy: {results['citation_accuracy']:.2%}")
    print(f"\nPersona Consistency:")
    for key, value in results['persona_consistency'].items():
        print(f"  {key}: {value:.2%}")
    print(f"\nResponse Quality:")
    for key, value in results['response_quality'].items():
        print(f"  {key}: {value:.2%}")
