
```markdown
# 🎙️ Lenny Rachitsky Clone - RAG Agent

A RAG-powered AI agent that clones Lenny Rachitsky's persona using his YouTube podcast transcripts and LinkedIn posts.

## 🎯 Features

- **100 YouTube transcripts** from Lenny's Podcast
- **100 LinkedIn posts** from Lenny's profile
- **Persona-aware responses** using BAIR/Stanford HAI research
- **Streaming chat interface** deployed on Vercel
- **Evaluation metrics** for synthetic chat quality

## 🏗️ Architecture

```
Data Ingestion → Processing → Vector DB → RAG Agent → Streamlit UI
```

## 📦 Setup

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `NVIDIA_API_KEY` - NVIDIA NIM API
- `QDRANT_API_KEY` - Qdrant Cloud
- `APIFY_API_KEY` - Apify (LinkedIn scraping)

### 3. Run Ingestion Pipeline

```bash
cd ingestion

# Run full pipeline
uv run python run_ingestion.py

# Or run steps individually
uv run python run_ingestion.py --step extract
uv run python run_ingestion.py --step process
uv run python run_ingestion.py --step upload
```

## 🚀 Usage

### Run Streamlit App Locally

```bash
cd app
uv run streamlit run main.py
```

### Deploy to Vercel

```bash
vercel deploy
```

## 📊 Evaluation

```bash
cd evaluation
jupyter notebook synthetic_chat_eval.ipynb
```

## 🗂️ Project Structure

```
lenny-clone/
├── ingestion/          # Data pipeline
├── agent/              # RAG agent
├── app/                # Streamlit UI
├── evaluation/         # Evaluation system
├── cli/                # CLI client
└── docs/               # Documentation
```

## 📄 Documentation

- [Implementation Details](docs/implementation.md)
- [Ideal System Design](docs/design.md)
- [Research Summary](docs/research_summary.md)

## 🧪 Testing

```bash
# Test YouTube extraction
cd ingestion
uv run python extract_youtube.py

# Test LinkedIn extraction
uv run python extract_linkedin.py

# Test RAG agent
cd ../agent
uv run python rag.py
```

## 🔧 Troubleshooting

### YouTube extraction fails
- Ensure `feedparser` is installed: `uv pip install feedparser`
- Try manual video ID collection from channel page

### LinkedIn extraction fails
- Verify Apify API key and credits
- Check actor names are correct: `apify/linkedin-profile-scraper`

### Qdrant connection fails
- Verify URL and API key in `.env`
- Check collection name is correct

## 📝 License

MIT

## 🙏 Acknowledgments

- Lenny Rachitsky for the amazing content
- BAIR and Stanford HAI for virtual persona research
```

---

# ✅ **Phase 1 & 2 Complete**

Now we have:
- ✅ Complete ingestion pipeline
- ✅ Qdrant upload
- ✅ Orchestration script

