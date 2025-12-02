"""
evaluation/eval.py
Comprehensive RAG Evaluation Suite
Metrics:
1. Retrieval Confidence (Vector Similarity)
2. System Latency (Performance)
3. Generation Quality (LLM-as-a-Judge)
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.rag import LennyRAG
from agent.llm_client import NvidiaLlamaClient

# 🧪 THE GOLDEN DATASET
TEST_CASES = [
    {
        "category": "Core Framework",
        "question": "What is product-market fit?",
        "criteria": "Must mention retention curves flattening and Sean Ellis test (>40%)."
    },
    {
        "category": "Benchmarks",
        "question": "What are good retention benchmarks for B2B SMB?",
        "criteria": "Must mention 60% annual retention."
    },
    {
        "category": "Hiring",
        "question": "How should I hire my first PM?",
        "criteria": "Focus on hunger, cognitive horsepower, and mandatory work sample tests."
    },
    {
        "category": "Definitions",
        "question": "What is a North Star Metric?",
        "criteria": "Single metric that captures value delivered to customers."
    },
    {
        "category": "Tactical",
        "question": "How do I increase conversion rates?",
        "criteria": "Tactical advice on friction, value proposition, or onboarding."
    }
]

class ComprehensiveEvaluator:
    def __init__(self):
        print("⚙️ Initializing Evaluator...")
        self.rag = LennyRAG()
        self.judge = NvidiaLlamaClient()
        self.judge.model = "meta/llama-3.1-70b-instruct" # Use the big brain for judging

    def evaluate_generation(self, question, response, context, criteria):
        """LLM-as-a-Judge for Semantic Quality"""
        context_text = "\n".join([c['text'][:200] for c in context])
        
        prompt = f"""
        You are an expert AI Evaluator.
        
        USER QUESTION: "{question}"
        EXPECTED CRITERIA: "{criteria}"
        
        RETRIEVED CONTEXT:
        {context_text}
        
        AI RESPONSE:
        "{response}"
        
        -----------------------
        
        Evaluate on these 3 dimensions (Score 1-5):
        
        1. **Faithfulness**: Is the answer derived from Context OR the Persona's known frameworks? (1=Hallucination, 5=Grounded)
        2. **Style Match**: Does it sound like Lenny (Tactical, "Think about it", No fluff)? (1=Generic AI, 5=Lenny)
        3. **Accuracy**: Does it meet the EXPECTED CRITERIA? (1=Wrong, 5=Perfect)
        
        OUTPUT JSON ONLY:
        {{
            "faithfulness": <int>,
            "style": <int>,
            "accuracy": <int>,
            "reasoning": "<short text>"
        }}
        """
        
        try:
            result = self.judge.generate(prompt, temperature=0.1)
            # Clean markdown
            if "```json" in result: result = result.split("```json")[1].split("```")[0]
            elif "```" in result: result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"faithfulness": 0, "style": 0, "accuracy": 0, "reasoning": "Error"}

    def run(self):
        results = []
        print(f"\n🚀 Starting Comprehensive Evaluation on {len(TEST_CASES)} cases...\n")
        
        for test in tqdm(TEST_CASES):
            # --- METRIC 1: LATENCY ---
            start_time = time.time()
            rag_result = self.rag.query_with_metadata(test['question'], top_k=5)
            end_time = time.time()
            latency = end_time - start_time
            
            response = rag_result['response']
            chunks = rag_result['chunks']
            
            # --- METRIC 2: RETRIEVAL CONFIDENCE ---
            # Average similarity score of the top 3 chunks
            if chunks:
                top_scores = [c['score'] for c in chunks[:3]]
                retrieval_confidence = sum(top_scores) / len(top_scores)
            else:
                retrieval_confidence = 0.0

            # --- METRIC 3: GENERATION QUALITY (LLM JUDGE) ---
            judge_scores = self.evaluate_generation(
                test['question'], response, chunks, test['criteria']
            )
            
            results.append({
                "Category": test['category'],
                "Question": test['question'],
                "Latency (s)": round(latency, 2),
                "Retrieval Conf": round(retrieval_confidence, 3),
                "Faithfulness": judge_scores['faithfulness'],
                "Style Score": judge_scores['style'],
                "Accuracy": judge_scores['accuracy'],
                "Judge Reasoning": judge_scores['reasoning']
            })

        # --- REPORTING ---
        df = pd.DataFrame(results)
        
        print("\n\n📊 DETAILED RESULTS:")
        print(df.drop(columns=['Judge Reasoning']).to_markdown(index=False))
        
        print("\n📈 AGGREGATE METRICS:")
        print(f"1. ⚡ Avg Latency:        {df['Latency (s)'].mean():.2f}s")
        print(f"2. 🔍 Avg Retrieval Conf: {df['Retrieval Conf'].mean():.3f} (0-1 Scale)")
        print(f"3. 🤖 Avg Style Score:    {df['Style Score'].mean():.1f}/5.0")
        print(f"4. ✅ Avg Accuracy:       {df['Accuracy'].mean():.1f}/5.0")
        
        # Save
        df.to_csv("evaluation/comprehensive_results.csv", index=False)
        print("\n💾 Saved to evaluation/comprehensive_results.csv")

if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    evaluator.run()
