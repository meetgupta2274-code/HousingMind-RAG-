"""
Ingest CSV data into ChromaDB with persistent storage.
Converts housing data rows into natural language chunks and embeds in batches of 30.
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "india_housing_prices.csv")
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "chroma_db")
COLLECTION_NAME = "housing_data"
BATCH_SIZE = 30
SAMPLE_SIZE = 5000


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
    """Check if the ChromaDB collection already has data."""
    if not os.path.exists(CHROMA_DB_PATH):
        return False
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        for col in collections:
            if col.name == COLLECTION_NAME:
                collection = client.get_collection(COLLECTION_NAME)
                count = collection.count()
                return count > 0
        return False
    except Exception:
        return False


def get_collection_count() -> int:
    """Get the number of documents in the collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def ingest_data() -> dict:
    """
    Main ingestion function.
    - Reads CSV, samples ~5000 rows stratified by State
    - Converts each row to text
    - Embeds in batches of 30 into ChromaDB (persistent)
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
        # proportional sampling
        n_sample = max(1, int(SAMPLE_SIZE * len(state_df) / total_rows))
        if n_sample > len(state_df):
            n_sample = len(state_df)
        sampled_dfs.append(state_df.sample(n=n_sample, random_state=42))

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Sampled {len(sampled_df)} rows across {sampled_df['State'].nunique()} states")

    # Convert rows to text chunks
    print("Converting rows to text chunks...")
    documents = []
    metadatas = []
    ids = []

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
        })
        ids.append(f"doc_{idx}")

    # Create ChromaDB client with persistent storage
    print(f"Creating ChromaDB persistent client at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "India Housing Prices RAG Collection"},
    )

    # Embed in batches of 30
    total_docs = len(documents)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {total_docs} documents in {total_batches} batches of {BATCH_SIZE}...")

    for i in range(0, total_docs, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_docs)
        batch_num = (i // BATCH_SIZE) + 1

        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )

        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches} done ({batch_end}/{total_docs} docs)")

    final_count = collection.count()
    print(f"Ingestion complete! {final_count} documents stored in ChromaDB.")

    return {
        "status": "success",
        "message": f"Successfully ingested {final_count} documents into ChromaDB.",
        "count": final_count,
    }


if __name__ == "__main__":
    result = ingest_data()
    print(result)
