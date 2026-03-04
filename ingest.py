import os
import shutil
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load your API key from .env
load_dotenv()

# ---- FETCH DATA ----

def get_wikipedia_data(topic):
    """Search Wikipedia and fetch the full article text"""
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 1
    }
    headers = {"User-Agent": "RaleighHousingAssistant/1.0 (educational project)"}

    # Search for the best matching article
    search_response = requests.get(search_url, params=search_params, headers=headers).json()
    results = search_response.get("query", {}).get("search", [])

    if not results:
        return ""

    # Get the full article text instead of just the summary
    page_title = results[0]["title"]
    content_params = {
        "action": "query",
        "titles": page_title,
        "prop": "extracts",
        "explaintext": True,  # plain text, no HTML
        "exsectionformat": "plain",
        "format": "json"
    }

    content_response = requests.get(search_url, params=content_params, headers=headers).json()
    pages = content_response.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    extract = page.get("extract", "")

    if extract:
        print(f"  Found: '{page_title}' ({len(extract)} chars)")

    return extract[:5000]

# Raleigh neighborhood and housing topics to fetch
topics = [
    "Raleigh North Carolina city",
    "Durham North Carolina city",
    "Cary North Carolina",
    "Chapel Hill North Carolina",
    "Research Triangle North Carolina",
    "Boylan Heights Raleigh neighborhood",
    "Five Points Raleigh neighborhood",
    "North Hills Raleigh neighborhood",
    "Glenwood South Raleigh neighborhood",
    "Oakwood Raleigh neighborhood",
    "Trinity Park Durham neighborhood",
    "Hope Valley Durham neighborhood",
    "Ninth Street Durham neighborhood",
]

print("Fetching data...")
documents = []
for topic in topics:
    try:
        text = get_wikipedia_data(topic)
        if text:
            documents.append(text)
            print(f"✓ Fetched: {topic}")
        else:
            print(f"✗ Empty response: {topic}")
    except Exception as e:
        print(f"✗ Failed: {topic} — {e}")

# ---- CHUNK THE DATA ----

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.create_documents(documents)
print(f"\n✓ Created {len(chunks)} chunks from {len(documents)} articles")

# ---- EMBED AND STORE IN CHROMADB ----
# Clear existing ChromaDB so we start fresh
if os.path.exists(".chroma"):
    shutil.rmtree(".chroma")
    print("✓ Cleared old ChromaDB")
print("\nEmbedding and storing in ChromaDB...")
import time

if len(chunks) == 0:
    print("ERROR: No chunks to embed. Check the data fetching step.")
else:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Process in batches of 80 to avoid rate limits
    batch_size = 49
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=".chroma"
            )
        else:
            vectorstore.add_documents(batch)
        print(f"✓ Embedded chunks {i+1} to {min(i+batch_size, len(chunks))}")
        if i + batch_size < len(chunks):
            print("  Waiting 30s for rate limit...")
            time.sleep(30)

    print("✓ Done! Your vector database is ready.")