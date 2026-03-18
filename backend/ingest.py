"""
Ingest CSV data into Qdrant Cloud with Hugging Face embeddings.
Converts housing data rows into natural language chunks and embeds in batches of 30.
"""

import os
import time
import requests
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "india_housing_prices.csv")

COLLECTION_NAME = "housing_data"
BATCH_SIZE = 30
SAMPLE_SIZE = 5000
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant cloud client."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")
    return QdrantClient(url=url, api_key=api_key, timeout=120)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Hugging Face Inference API."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be set in environment variables.")

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(3):
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # Model loading, wait and retry
            wait = response.json().get("estimated_time", 20)
            print(f"  HF model loading, waiting {wait:.0f}s...")
            time.sleep(min(wait, 30))
        else:
            raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

    raise RuntimeError("HF API failed after 3 retries.")


def row_to_text(row: dict) -> str:
    """Convert a CSV row into a descriptive natural language text chunk."""
    parts = []

    bhk = row.get("BHK", "N/A")
    prop_type = row.get("Property_Type", "Property")
    locality = row.get("Locality", "Unknown")
    city = row.get("City", "Unknown")
    state = row.get("State", "Unknown")
    parts.append(f"{bhk} BHK {prop_type} in {locality}, {city}, {state}.")

    size = row.get("Size_in_SqFt", "N/A")
    price = row.get("Price_in_Lakhs", "N/A")
    price_sqft = row.get("Price_per_SqFt", "N/A")
    parts.append(f"Size: {size} sqft. Price: ₹{price} Lakhs (₹{price_sqft}/sqft).")

    year = row.get("Year_Built", "N/A")
    age = row.get("Age_of_Property", "N/A")
    parts.append(f"Built in {year}, property age: {age} years.")

    furnished = row.get("Furnished_Status", "N/A")
    floor = row.get("Floor_No", "N/A")
    total_floors = row.get("Total_Floors", "N/A")
    parts.append(f"Furnished status: {furnished}. Floor {floor} of {total_floors}.")

    parking = row.get("Parking_Space", "N/A")
    security = row.get("Security", "N/A")
    facing = row.get("Facing", "N/A")
    parts.append(f"Parking: {parking}. Security: {security}. Facing: {facing}.")

    amenities = row.get("Amenities", "N/A")
    parts.append(f"Amenities: {amenities}.")

    schools = row.get("Nearby_Schools", "N/A")
    hospitals = row.get("Nearby_Hospitals", "N/A")
    transport = row.get("Public_Transport_Accessibility", "N/A")
    parts.append(
        f"Nearby schools: {schools}. Nearby hospitals: {hospitals}. "
        f"Public transport accessibility: {transport}."
    )

    owner = row.get("Owner_Type", "N/A")
    status = row.get("Availability_Status", "N/A")
    parts.append(f"Owner type: {owner}. Availability: {status}.")

    return " ".join(parts)


def is_already_ingested() -> bool:
    """Check if the Qdrant collection already has data."""
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return False
        count = client.count(collection_name=COLLECTION_NAME).count
        return count > 0
    except Exception:
        return False


def get_collection_count() -> int:
    """Get the number of documents in the Qdrant collection."""
    try:
        client = get_qdrant_client()
        return client.count(collection_name=COLLECTION_NAME).count
    except Exception:
        return 0


def ingest_data() -> dict:
    """
    Main ingestion function.
    - Reads CSV, samples ~5000 rows stratified by State
    - Converts each row to text
    - Embeds via HuggingFace API in batches of 30
    - Upserts vectors into Qdrant Cloud
    Returns status dict with count and message.
    """
    if is_already_ingested():
        count = get_collection_count()
        return {
            "status": "already_exists",
            "message": f"Embeddings already exist with {count} documents. Skipping ingestion.",
            "count": count,
        }

    print("Reading CSV file...")
    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    print(f"Total rows in CSV: {total_rows}")

    # Stratified sampling by State to get representative data
    print(f"Sampling {SAMPLE_SIZE} rows stratified by State...")
    sampled_dfs = []
    state_counts = df["State"].value_counts()
    for state in state_counts.index:
        state_df = df[df["State"] == state]
        n_sample = max(1, int(SAMPLE_SIZE * len(state_df) / total_rows))
        if n_sample > len(state_df):
            n_sample = len(state_df)
        sampled_dfs.append(state_df.sample(n=n_sample, random_state=42))

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Sampled {len(sampled_df)} rows across {sampled_df['State'].nunique()} states")

    # Free the full dataframe from memory immediately
    del df

    # Convert rows to text chunks
    print("Converting rows to text chunks...")
    documents = []
    metadatas = []

    for idx, row in sampled_df.iterrows():
        text = row_to_text(row.to_dict())
        documents.append(text)
        metadatas.append({
            "state": str(row.get("State", "")),
            "city": str(row.get("City", "")),
            "property_type": str(row.get("Property_Type", "")),
            "bhk": str(row.get("BHK", "")),
            "price_lakhs": str(row.get("Price_in_Lakhs", "")),
            "size_sqft": str(row.get("Size_in_SqFt", "")),
            "text": text,
        })

    # Free the dataframe from memory
    del sampled_df

    # Create Qdrant collection
    print("Connecting to Qdrant Cloud and creating collection...")
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Embed and upsert in batches of 30
    total_docs = len(documents)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {total_docs} documents in {total_batches} batches of {BATCH_SIZE}...")

    point_id = 0
    for i in range(0, total_docs, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_docs)
        batch_num = (i // BATCH_SIZE) + 1
        batch_texts = documents[i:batch_end]
        batch_metas = metadatas[i:batch_end]

        embeddings = get_embeddings(batch_texts)

        points = []
        for j, (embedding, meta) in enumerate(zip(embeddings, batch_metas)):
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=meta,
            ))
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)

        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches} done ({batch_end}/{total_docs} docs)")

    final_count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Ingestion complete! {final_count} documents stored in Qdrant Cloud.")

    return {
        "status": "success",
        "message": f"Successfully ingested {final_count} documents into Qdrant Cloud.",
        "count": final_count,
    }


if __name__ == "__main__":
    result = ingest_data()
    print(result)
