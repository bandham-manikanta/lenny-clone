from agent.rag import LennyRAG

rag = LennyRAG()

test_questions = [
    "What is product-market fit?",
    "How do I improve retention?",
    "What makes a great PM?"
]

for q in test_questions:
    print(f"\n{'='*60}\nQ: {q}\n{'='*60}\n")
    
    result = rag.query_with_metadata(q, top_k=5, temperature=0.3)
    print(result['response'])
    print(f"\n📚 Sources: {len(result['sources'])}")
