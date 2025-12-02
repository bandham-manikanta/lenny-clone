# IMPLEMENTATION.md

## System Architecture

### High-Level Design

```
User Query
    ↓
[Dual-Stream Retrieval]
    ├─→ LinkedIn Posts (Lenny's Core Beliefs)
    └─→ YouTube Transcripts (Guest Examples)
    ↓
[Persona Layer]
    ├─→ Framework Detection
    └─→ Context Formatting
    ↓
[LLM Generation (Streaming)]
    ↓
Response + Citations
```

**Key Innovation:** Stratified retrieval prevents guest opinion leakage into Lenny's voice.

---

## Core Technical Decisions

### 1. Dual-Stream Retrieval Architecture

**Problem:** Early prototypes attributed guest quotes to Lenny, creating inconsistent persona.

**Solution:** Separate retrieval streams with distinct semantic roles.

```python
# agent/rag.py
def _get_dual_stream_context(self, question: str, top_k: int):
    lenny_chunks = self.retriever.search_with_filters(
        query=question,
        source_filter="linkedin",  # Core beliefs
        top_k=3
    )
    
    guest_chunks = self.retriever.search_with_filters(
        query=question,
        source_filter="youtube",  # Examples only
        top_k=3
    )
    
    return lenny_chunks, guest_chunks
```

**Trade-off:** Doubles retrieval latency (100ms → 200ms), but still well below LLM generation time (2-4s).

---

### 2. Persona-Aware Prompt Engineering

**Implementation:** Three-tier prompting strategy:

#### Tier 1: System Prompt (Tone)
```python
# agent/persona.py
def get_system_prompt(self):
    return """You are Lenny responding to a reader's email.

Write like you're talking to a friend:
- Use contractions (you're, don't, it's)
- Vary sentence length 
- Be opinionated but humble
- No formality, no fluff"""
```

#### Tier 2: Framework Injection (Content)
```python
def _detect_framework(self, question: str):
    q = question.lower()
    if "pmf" in q or "fit" in q:
        return self.core_frameworks["product-market fit"]
    # ... other frameworks
```

Hard-coded frameworks ensure factual accuracy even when retrieval fails.

#### Tier 3: Context Formatting (Structure)
```python
def _format_context(self, lenny_chunks, guest_chunks):
    parts = []
    if lenny_chunks:
        parts.append("From your writing:\n" + ...)
    if guest_chunks:
        parts.append("From guests:\n" + ...)
    return "\n\n".join(parts)
```

---

### 3. Streaming Response Architecture

**Challenge:** Streamlit generators don't stream properly if returned directly.

**Root Cause:**
```python
# ❌ Broken - returns generator object, doesn't yield
def query(self, stream=True):
    return self.llm.generate(..., stream=True)

# ✅ Fixed - yields from generator
def query(self, stream=True):
    for chunk in self.llm.generate(..., stream=True):
        yield chunk
```

**Why This Matters:**
- Streamlit expects `yield` statements in the call stack
- Returning a generator breaks the async chain
- Proper yielding enables real-time token display

---

### 4. Speaker Diarization Heuristic

**Problem:** YouTube auto-transcripts lack speaker labels.

**Constraints:**
- Whisper diarization: 2min per video (too slow for 127 videos)
- Third-party APIs: $0.01/min (would cost $100+)

**Solution:** Rule-based heuristic classifier:

```python
# ingestion/clean_transcripts.py
def is_likely_lenny(text, index, total_chunks):
    # Intro detection
    if index < (total_chunks * 0.05):
        return True
    
    # Signature phrase matching
    if "welcome to the podcast" in text.lower():
        return True
    
    # Question pattern (Lenny asks, guests answer long-form)
    if "?" in text and len(text) < 200:
        return True
    
    return False
```

**Estimated Accuracy:** ~80-85% based on manual spot-checking.

---

### 5. Embedding Strategy

**Choice:** `sentence-transformers/all-mpnet-base-v2` (768 dims)

**Optimization:** In-memory caching of query embeddings:
```python
# agent/chroma_retriever.py
self._embedding_cache = {}  # MD5(query) → embedding vector
```

---

### 6. ChromaDB for Vector Storage

**Choice:** ChromaDB with local persistence

**Why:**
- Local, no API costs
- Fast enough for <100k chunks (~100ms queries)
- Persistent SQLite-based storage
- Portable for deployment

**Implementation:**
```python
# agent/chroma_retriever.py
self.client = chromadb.PersistentClient(path=self.persist_directory)
self.collection = self.client.get_collection(name=self.collection_name)
```

---

### 7. Evaluation Framework (LLM-as-a-Judge)

**Implementation:**

```python
# evaluation/evaluation.py
def evaluate_response(self, question, generated_answer, framework):
    prompt = f"""
    Evaluate on these 3 dimensions (0-10 score):
    1. Framework Accuracy
    2. Stylistic Fidelity
    3. Distinction
    
    Output JSON only: {{"framework_score": <float>, ...}}
    """
    
    response = self.judge.generate(prompt, temperature=0.1)
    return json.loads(response)
```

**Metrics Tracked:**
1. **Framework Accuracy** (0-10): Follows Lenny's actual beliefs?
2. **Stylistic Fidelity** (0-10): Sounds like Lenny?
3. **Distinction Score** (0-10): Separates Lenny's views from guest opinions?

---

## Performance Characteristics

### Latency Breakdown (p50)

```
Total Query Time: 2.3s
├─ Embedding: 50ms
├─ Dual Retrieval: 200ms (100ms × 2)
├─ Prompt Construction: 10ms
├─ LLM Generation: 2000ms (TTFB: 200ms)
└─ Source Formatting: 40ms
```

**Bottleneck:** LLM generation (87% of total time).

---

### Cost Analysis

**Per Query:**
```
Embedding (local): $0.000
Retrieval (local): $0.000
LLM (NVIDIA NIM): $0.001 per 1k tokens
Average response: 300 tokens = $0.0003

Total: ~$0.0003 per query
```

**Monthly Estimate (1000 queries):** $0.30  
**Compare to GPT-4:** $9.00 (30x more expensive)

---

## Technical Debt & Known Limitations

### 1. Stateless Conversations

**Current:** Each query is independent, no memory between turns.

**Impact:** Can't do:
- "Tell me more about that"
- "Can you elaborate on point 2?"

---

### 2. No Hybrid Search

**Current:** Pure semantic search (vector similarity).

**Problem:** Misses exact keyword matches.

Example:
- Query: "What's the 40% rule?"
- Semantic search might miss chunks containing "40%" if topic embeddings don't align

**Why Not Implemented:**
- ChromaDB doesn't support BM25 natively
- Current semantic-only approach works "well enough"

---

### 3. Speaker Diarization Accuracy

**Current:** ~80-85% precision on separating Lenny vs Guests.

**Failure Modes:**
- Lenny's long explanations → misclassified as Guest (~15%)
- Guest's short questions → misclassified as Lenny (~5%)

---

## Deployment Architecture

### Streamlit Cloud

```
GitHub Repo
    ↓
Streamlit Cloud (Serverless)
    ├─ Python 3.11 Container
    ├─ Load chroma_db/ from repo (45MB)
    ├─ Download sentence-transformers model (100MB, cached)
    └─ Run app/streamlit_app.py

Cold Start: 30s (model download)
Warm Start: 3s
```

**Key Constraint:** 1GB RAM limit on free tier.

**Workaround:** Use `all-MiniLM-L6-v2` (384 dims) instead of larger models.

---

## Key Takeaways

### What Worked

1. **Dual-stream retrieval** - Critical for persona consistency
2. **Hard-coded frameworks** - Ensures baseline accuracy (6/10 floor)
3. **LLM-as-a-Judge** - Scalable, reproducible evaluation
4. **Local embeddings** - Zero cost, good enough quality
5. **Heuristic diarization** - 80-85% accuracy at zero cost
6. **ChromaDB** - Simple, fast, portable for <100k vectors

### What's Next (Not Yet Implemented)

1. **Conversation memory** - Enable multi-turn dialogs
2. **Hybrid search** - Better exact-match retrieval (BM25 + vector fusion)
3. **Speaker embedding diarization** - Improve Lenny/Guest separation to 90%+
```