import os
import time
import shutil
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---- CONFIGURATION ----

load_dotenv()

TOPICS = [
    # Cities and regions
    "Raleigh North Carolina city",
    "Durham North Carolina city",
    "Cary North Carolina",
    "Chapel Hill North Carolina",
    "Research Triangle North Carolina",
    # Raleigh neighborhoods
    "Boylan Heights Raleigh neighborhood",
    "Five Points Raleigh neighborhood",
    "North Hills Raleigh neighborhood",
    "Glenwood South Raleigh neighborhood",
    "Oakwood Raleigh neighborhood",
    # Durham neighborhoods
    "Trinity Park Durham neighborhood",
    "Hope Valley Durham neighborhood",
    "Ninth Street Durham neighborhood",
]

FRED_SERIES = {
    "MEDLISPRI39580": "Median Listing Price in Raleigh NC",
    "AVELISPRI39580": "Average Listing Price in Raleigh NC",
    "ACTLISCOU39580": "Active Listing Count in Raleigh NC",
    "MEDDAYONMAR39580": "Median Days on Market in Raleigh NC",
    "RPPSERVERENT39580": "Regional Housing Cost Index for Raleigh NC",
    "DP04ACS037183": "Housing Cost Burdened Households in Wake County NC",
}


HUD_COUNTIES = {
    "3718399999": "Wake County (Raleigh, Cary, Apex)",
    "3706399999": "Durham County"
}

# ---- DATA FETCHING ----

def get_wikipedia_data(topic):
    """Search Wikipedia and fetch the full article text"""
    headers = {"User-Agent": "RaleighHousingAssistant/1.0 (educational project)"}
    search_url = "https://en.wikipedia.org/w/api.php"

    # Step 1: Search for the best matching article
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 1
    }
    search_response = requests.get(search_url, params=search_params, headers=headers, timeout=10).json()    
    results = search_response.get("query", {}).get("search", [])
    if not results:
        return ""

    # Step 2: Fetch the full article text
    page_title = results[0]["title"]
    content_params = {
        "action": "query",
        "titles": page_title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json"
    }
    content_response = requests.get(search_url, params=content_params, headers=headers, timeout=10).json()   
    pages = content_response.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    extract = page.get("extract", "")

    if extract:
        print(f"  Found: '{page_title}' ({len(extract)} chars)")

    return extract[:5000]


def get_fred_data():
    """Fetch Raleigh housing market data from FRED API"""
    api_key = os.getenv("FRED_API_KEY")
    all_text = []

    for series_id, description in FRED_SERIES.items():
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 24
        }
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params
        ).json()
        observations = response.get("observations", [])

        if not observations:
            print(f"  ✗ No data for {series_id}")
            continue

        lines = [f"{description} (last 24 months):"]
        for obs in observations:
            if obs["value"] != ".":
                lines.append(f"  {obs['date']}: {obs['value']}")

        all_text.append("\n".join(lines))
        print(f"  ✓ Fetched: {description} ({len(observations)} months)")

    return "\n\n".join(all_text)


def get_hud_data():
    """Fetch HUD Fair Market Rents for Raleigh and Durham via API"""
    token = os.getenv("HUD_API_TOKEN")
    all_text = []

    for fips, name in HUD_COUNTIES.items():
        response = requests.get(
            f"https://www.huduser.gov/hudapi/public/fmr/data/{fips}",
            headers={"Authorization": f"Bearer {token}"}
        ).json()

        basicdata = response.get("data", {}).get("basicdata", {})
        if not basicdata:
            print(f"  ✗ No HUD data for {name}")
            continue

        # HUD returns either a list or a dict depending on the county
        latest = basicdata[0] if isinstance(basicdata, list) else basicdata
        text = f"""HUD Fair Market Rents for {name} (FY2026):
- Efficiency/Studio: ${latest.get('Efficiency', 'N/A')}/month
- 1 Bedroom: ${latest.get('One-Bedroom', 'N/A')}/month
- 2 Bedroom: ${latest.get('Two-Bedroom', 'N/A')}/month
- 3 Bedroom: ${latest.get('Three-Bedroom', 'N/A')}/month
- 4 Bedroom: ${latest.get('Four-Bedroom', 'N/A')}/month"""

        all_text.append(text)
        print(f"  ✓ Fetched HUD FMR data for {name}")

    return "\n\n".join(all_text)


# ---- MAIN INGESTION PIPELINE ----

def fetch_all_documents():
    """Fetch all data sources and return as a list of text documents"""
    documents = []

    print("\n[1/3] Fetching FRED housing market data...")
    fred_text = get_fred_data()
    if fred_text:
        documents.append(fred_text)
        print("✓ FRED data added")

    print("\n[2/3] Fetching HUD fair market rent data...")
    hud_text = get_hud_data()
    if hud_text:
        documents.append(hud_text)
        print("✓ HUD data added")

    print("\n[3/3] Fetching Wikipedia neighborhood data...")
    for topic in TOPICS:
        try:
            text = get_wikipedia_data(topic)
            if text:
                documents.append(text)
            else:
                print(f"  ✗ Empty response: {topic}")
        except Exception as e:
            print(f"  ✗ Failed: {topic} — {e}")
        time.sleep(1)  # small delay between requests

    return documents


def chunk_documents(documents):
    """Split documents into chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(documents)
    print(f"\n✓ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def embed_and_store(chunks):
    """Embed chunks and store in ChromaDB"""
    # Clear existing database
    if os.path.exists(".chroma"):
        shutil.rmtree(".chroma")
        print("✓ Cleared old ChromaDB")

    if len(chunks) == 0:
        print("ERROR: No chunks to embed.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Process in batches to avoid rate limits
    batch_size = 49
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
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

    print("\n✓ Done! Your vector database is ready.")


# ---- RUN ----

if __name__ == "__main__":
    print("Starting Raleigh Housing Assistant ingestion pipeline...")
    documents = fetch_all_documents()
    chunks = chunk_documents(documents)
    embed_and_store(chunks)