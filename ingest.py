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

def get_financial_guidelines():
    """Return financial guidelines and rules of thumb for rent vs. buy decisions"""
    
    guidelines = """
FINANCIAL GUIDELINES FOR RENTING VS. BUYING IN RALEIGH-DURHAM (2026)

DISCLAIMER: The following are general financial guidelines and rules of thumb for 
educational purposes only. This is not personalized financial advice. Always consult 
a licensed financial advisor before making major financial decisions.

--- THE 28/36 RULE ---
Spend no more than 28% of your gross monthly income on housing costs (mortgage or rent).
Spend no more than 36% of your gross monthly income on total debt (housing + car + student loans).
Example: $80,000 annual salary = $6,667/month gross income.
Maximum recommended housing payment: $1,867/month.
Maximum recommended total debt: $2,400/month.

--- THE PRICE-TO-RENT RATIO ---
Divide the home purchase price by the annual rent for a comparable home.
Below 15: Buying generally makes more financial sense.
Between 15 and 20: Could go either way depending on your situation.
Above 20: Renting likely makes more financial sense.
Raleigh's current price-to-rent ratio is approximately 18-20, making it borderline.
Example: $440,000 home vs. $24,000/year rent ($2,000/month) = ratio of 18.3.

--- THE BREAK-EVEN HORIZON ---
Buying only makes financial sense if you plan to stay long enough to recoup closing costs.
General rule: Plan to stay at least 5 years before buying.
Closing costs in North Carolina typically run 2-5% of the purchase price.
On a $440,000 Raleigh home, expect $8,800 to $22,000 in closing costs.
Current break-even horizon in Raleigh is estimated at 6-7 years given current mortgage rates.

--- THE 20% DOWN PAYMENT RULE ---
Putting less than 20% down triggers PMI (Private Mortgage Insurance).
PMI typically adds $100-200/month to your payment on a $440,000 home.
First-time buyer programs in NC allow as little as 3-5% down.
NC Housing Finance Agency offers down payment assistance for qualifying buyers.
20% down on a $440,000 Raleigh home = $88,000 required upfront.

--- THE EMERGENCY FUND RULE ---
Have 3-6 months of living expenses saved BEFORE buying a home.
This is separate from your down payment and closing costs.
For a Raleigh household spending $4,000/month, that means $12,000-$24,000 in reserves.
Buying without an emergency fund puts you at serious financial risk.

--- THE 1% MAINTENANCE RULE ---
Budget approximately 1% of your home's value per year for maintenance and repairs.
On a $440,000 Raleigh home: $4,400/year or approximately $367/month.
This is frequently overlooked when comparing mortgage payments to rent.
True monthly cost of owning a $440,000 home at 7% mortgage rate (20% down):
  - Principal + Interest: $2,340/month
  - Property taxes (Wake County ~0.65%): $238/month
  - Homeowner's insurance: ~$100/month
  - Maintenance (1% rule): ~$367/month
  - Total true cost: ~$3,045/month

--- RALEIGH-DURHAM MARKET CONTEXT (2026) ---
Raleigh median home price: approximately $440,000.
Durham median home price: approximately $380,000.
Current 30-year fixed mortgage rate: approximately 6.5-7%.
To meet the 28% rule on a $440,000 Raleigh home: need ~$100,000+ gross annual income.
Raleigh rent vs. buy break-even: approximately 6-7 years at current rates.
Wake County property tax rate: approximately 0.65% of assessed value.
Durham County property tax rate: approximately 1.08% of assessed value.

--- WHEN RENTING MAKES MORE SENSE ---
You plan to stay less than 5 years.
Your price-to-rent ratio is above 20.
You don't have 20% down plus closing costs plus emergency fund saved.
Your housing costs would exceed 28% of gross income.
You value flexibility for career or lifestyle changes.
You don't want responsibility for maintenance and repairs.

--- WHEN BUYING MAKES MORE SENSE ---
You plan to stay 5+ years in the same area.
You have 20% down payment plus closing costs saved.
Your mortgage payment stays under 28% of gross income.
You want to build equity and have stability.
You're ready for the responsibilities of homeownership.
Interest rates are favorable relative to rent costs.
"""
    print("✓ Loaded financial guidelines")
    return guidelines

def get_zillow_data():
    """Parse Zillow CSV files for Raleigh/Durham ZIP code data"""
    import csv

    # ZIP codes mapped to neighborhoods
    raleigh_zips = {
        "27601": "Downtown Raleigh",
        "27603": "Boylan Heights, South Raleigh",
        "27605": "Five Points, Glenwood South",
        "27607": "Cameron Village, West Raleigh",
        "27609": "North Hills, North Raleigh",
        "27612": "Northwest Raleigh",
        "27701": "Downtown Durham",
        "27703": "East Durham",
        "27705": "Duke University area, Trinity Park",
    }

    all_text = []

    # Process home values (ZHVI)
    zhvi_file = "data/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    if os.path.exists(zhvi_file):
        print("  Processing Zillow home value data...")
        with open(zhvi_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Get the last 12 month columns
            fieldnames = reader.fieldnames
            date_cols = [c for c in fieldnames if c.startswith("20")][-12:]

            for row in reader:
                zip_code = row["RegionName"]
                if zip_code in raleigh_zips:
                    neighborhood = raleigh_zips[zip_code]
                    lines = [f"Zillow Home Values for ZIP {zip_code} ({neighborhood}):"]
                    for date in date_cols:
                        val = row.get(date, "")
                        if val:
                            lines.append(f"  {date}: ${float(val):,.0f}")
                    all_text.append("\n".join(lines))
                    print(f"  ✓ Home values: ZIP {zip_code} ({neighborhood})")

    # Process rent index (ZORI)
    zori_file = "data/Zip_zori_uc_sfrcondomfr_sm_month.csv"
    if os.path.exists(zori_file):
        print("  Processing Zillow rent data...")
        with open(zori_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            date_cols = [c for c in fieldnames if c.startswith("20")][-12:]

            for row in reader:
                zip_code = row["RegionName"]
                if zip_code in raleigh_zips:
                    neighborhood = raleigh_zips[zip_code]
                    lines = [f"Zillow Observed Rent Index for ZIP {zip_code} ({neighborhood}):"]
                    for date in date_cols:
                        val = row.get(date, "")
                        if val:
                            lines.append(f"  {date}: ${float(val):,.0f}/month")
                    all_text.append("\n".join(lines))
                    print(f"  ✓ Rent index: ZIP {zip_code} ({neighborhood})")

    return "\n\n".join(all_text)


# ---- MAIN INGESTION PIPELINE ----

def fetch_all_documents():
    """Fetch all data sources and return as a list of text documents"""
    documents = []

    print("\n[1/5] Fetching FRED housing market data...")
    fred_text = get_fred_data()
    if fred_text:
        documents.append(fred_text)
        print("✓ FRED data added")

    print("\n[2/5] Fetching HUD fair market rent data...")
    hud_text = get_hud_data()
    if hud_text:
        documents.append(hud_text)
        print("✓ HUD data added")

    print("\n[3/5] Fetching Wikipedia neighborhood data...")
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

    print("\n[4/5] Loading financial guidelines...")
    financial_text = get_financial_guidelines()
    if financial_text:
        documents.append(financial_text)
        print("✓ Financial guidelines added")

    print("\n[5/5] Loading Zillow neighborhood data...")
    zillow_text = get_zillow_data()
    if zillow_text:
        documents.append(zillow_text)
        print("✓ Zillow data added")

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