# Raleigh-Durham Housing Intelligence Assistant

A conversational RAG (Retrieval-Augmented Generation) chatbot that helps users navigate the Raleigh-Durham housing market. Built with LangChain, ChromaDB, and Google Gemini.

🏠 **[Live Demo](https://raleigh-housing-assistant.streamlit.app/)**

---

## What it does

This assistant answers questions about buying, renting, and neighborhood selection in the Raleigh-Durham area. It combines real housing market data with conversational AI to give personalized, context-aware responses.

Users complete a brief profile (budget, rent vs. buy preference, priorities) and then chat naturally with the assistant.

Example questions it can answer:
- "What is Boylan Heights like?"
- "How does Glenwood South compare to Five Points?"
- "Based on my budget, should I rent or buy right now?"
- "What are average home prices in North Raleigh?"
- "How long would I need to stay to make buying worth it?"

---

## Architecture
```
User Question
     ↓
Query Condensation (rewrites follow-ups as standalone questions)
     ↓
ChromaDB Vector Search (finds 4 most relevant chunks)
     ↓
Gemini 2.5 Flash (generates answer using retrieved context)
     ↓
Streamlit UI (renders response with conversation memory)
```

### Data Sources
| Source | Data | Update Frequency |
|---|---|---|
| FRED API | Median listing price, active listings, days on market | Monthly |
| HUD API | Fair market rents for Wake and Durham counties | Annual |
| Zillow Research CSVs | Home values and rent index by ZIP code | Monthly |
| Wikipedia | Neighborhood profiles and character descriptions | Static |
| Hardcoded guidelines | 28/36 rule, price-to-rent ratio, break-even horizon | Static |

### Tech Stack
- **LangChain** — RAG pipeline orchestration and conversation memory
- **ChromaDB** — local vector database for semantic search
- **Google Gemini** — embeddings (`gemini-embedding-001`) and generation (`gemini-2.5-flash`)
- **Streamlit** — web interface
- **FRED API / HUD API** — real-time housing market data

---

## How it works

### Ingestion (`ingest.py`)
1. Fetches data from FRED API, HUD API, Wikipedia, and Zillow CSVs
2. Chunks text into 500-character pieces with 50-character overlap
3. Converts each chunk into a vector embedding using Gemini
4. Stores vectors and text in ChromaDB

### Query (`app.py`)
1. User asks a question
2. If conversation history exists, a condense chain rewrites the question as a standalone query
3. ChromaDB finds the 4 most semantically similar chunks
4. Gemini generates a conversational answer grounded in the retrieved context
5. Answer and question are appended to conversation memory for future turns

### Query Condensation
Follow-up questions like "is it affordable for me?" are ambiguous to a vector search engine. Before hitting ChromaDB, the question gets rewritten using conversation history:

> "is it affordable for me?" → "Is Boylan Heights affordable for someone with a $1,500-$2,000/month housing budget?"

This ensures retrieval accuracy across multi-turn conversations.

---

## Running locally

### Prerequisites
- Python 3.11+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))
- FRED API key (free at [fred.stlouisfed.org](https://fred.stlouisfed.org))
- HUD API token (free at [huduser.gov](https://www.huduser.gov/portal/dataset/fmr-api.html))

### Setup
```bash
git clone https://github.com/theodore-thormann/raleigh-housing-assistant
cd raleigh-housing-assistant
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
HUD_API_TOKEN=your_token_here
```

Build the vector database:
```bash
python3 ingest.py
```

Run the app:
```bash
streamlit run app.py
```

---

## Project structure
```
raleigh-housing-assistant/
├── app.py              # Streamlit UI and RAG chain
├── ingest.py           # Data ingestion pipeline
├── query.py            # Terminal query interface
├── data/               # Filtered Zillow CSV data
├── .chroma/            # Vector database (generated, not committed)
├── .env                # API keys (not committed)
└── README.md
```

---

## Disclaimer

This tool provides general housing information and financial guidelines for educational purposes only. It is not personalized financial advice. Always consult a licensed financial advisor before making major financial decisions.

---

## Author

Theodore Thormann — [GitHub](https://github.com/theodore-thormann) · [LinkedIn](your-linkedin-url)