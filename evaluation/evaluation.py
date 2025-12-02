import json
import os
from dotenv import load_dotenv
from agent.rag import LennyRAG
from agent.llm_client import NvidiaLlamaClient
import pandas as pd
from tqdm import tqdm

load_dotenv()

# 1. EVALUATION DATASET (Golden QA Pairs)
EVAL_SET = [
    {
        "question": "How do I know if I have Product Market Fit?",
        "ground_truth_framework": "PMF is when retention curves flatten. 40% very disappointed test.",
        "style_markers": ["tactical", "data-driven", "short paragraphs"]
    },
    {
        "question": "What is a good retention rate for B2B?",
        "ground_truth_framework": "SMB: 60%, Enterprise: 80%+. If below, fix leaky bucket.",
        "style_markers": ["benchmarks", "direct", "no fluff"]
    },
    {
        "question": "Should I hire a PM?",
        "ground_truth_framework": "Don't hire until you have PMF. Founders should do product first.",
        "style_markers": ["contrarian", "founder-focused"]
    }
]

class LennyEvaluator:
    def __init__(self):
        self.rag = LennyRAG()
        self.judge = NvidiaLlamaClient()  # The Judge LLM
        with open('agent/lenny_dna.json', 'r') as f:
            self.dna = json.load(f)

    def evaluate_response(self, question, generated_answer, framework):
        """
        Uses LLM-as-a-Judge to score the response on 0-10 scale.
        """
        prompt = f"""
        You are an expert evaluator for a Virtual Persona. 
        Your job is to rate the following response based on how well it mimics 'Lenny Rachitsky'.

        THE PERSONA DNA:
        - Keywords: {', '.join(self.dna.get('vocabulary_whitelist', [])[:10])}
        - Tone: {', '.join(self.dna.get('tone_descriptors', []))}
        - Formatting: {', '.join(self.dna.get('formatting_rules', []))}

        GROUND TRUTH FRAMEWORK (What Lenny actually believes):
        "{framework}"

        GENERATED RESPONSE:
        "{generated_answer}"

        Evaluate on these 3 dimensions (0-10 score):
        1. **Framework Accuracy**: Does it strictly follow the ground truth framework? (Penalty if it uses Guest opinions as Lenny's facts).
        2. **Stylistic Fidelity**: Does it use short paragraphs, 'Lenny-isms' (sticky, unlock, flywheel), and avoid corporate fluff?
        3. **Distinction**: Does it separate Lenny's view from Guest views?

        Output JSON only:
        {{
            "framework_score": <float>,
            "style_score": <float>,
            "distinction_score": <float>,
            "reasoning": "<short explanation>"
        }}
        """
        
        try:
            response = self.judge.generate(prompt, temperature=0.1)
            # Naive parsing - mostly works with Llama 3.1 70B
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            return json.loads(response)
        except Exception as e:
            print(f"Error grading: {e}")
            return {"framework_score": 0, "style_score": 0, "distinction_score": 0, "reasoning": "Error"}

    def run_eval(self):
        results = []
        print("⚖️  Judge Model initializing...")
        
        for item in tqdm(EVAL_SET):
            # Generate Answer
            rag_output = self.rag.query_with_metadata(item['question'])
            answer = rag_output['response']
            
            # Grade Answer
            scores = self.evaluate_response(item['question'], answer, item['ground_truth_framework'])
            
            results.append({
                "question": item['question'],
                "answer": answer[:50] + "...",
                **scores
            })

        df = pd.DataFrame(results)
        df['Total Score'] = (df['framework_score'] + df['style_score'] + df['distinction_score']) / 3
        print("\n🏆 EVALUATION RESULTS 🏆")
        print(df.to_markdown())
        
        # Save for the Founder to see
        df.to_csv("evaluation_metrics.csv")
        print("\n✅ Metrics saved to evaluation_metrics.csv")

if __name__ == "__main__":
    evaluator = LennyEvaluator()
    evaluator.run_eval()