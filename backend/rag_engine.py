"""
RAG Engine: Query Qdrant Cloud and generate answers using Groq LLM.
"""

import os
import time
import requests
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "housing_data"
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"

# Token limit management
MAX_CONTEXT_CHARS = 6000  # ~1500 tokens for context
MAX_ANSWER_TOKENS = 1024

SYSTEM_PROMPT = """You are an expert Indian real estate assistant. You help users find information about housing prices, properties, and real estate trends across India.

You answer questions based ONLY on the provided context data from a housing prices database. If the context doesn't contain enough information to answer the question accurately, say so honestly.

Guidelines:
- Give specific data points (prices, sizes, locations) when available
- Format prices in Lakhs (₹) as used in India
- Compare properties when relevant
- Be concise but thorough
- Use bullet points for clarity when listing multiple items
- If asked about trends or averages, analyze the provided data points
- Always mention the city/state when discussing properties"""


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant cloud client."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")
    return QdrantClient(url=url, api_key=api_key, timeout=120)


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single query text from HuggingFace Inference API."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be set in environment variables.")

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    for attempt in range(3):
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            # HF returns list of token embeddings — take the mean for sentence embedding
            if isinstance(result[0], list):
                embedding = [sum(x) / len(result) for x in zip(*result)]
            else:
                embedding = result
            return embedding
        elif response.status_code == 503:
            wait = response.json().get("estimated_time", 20)
            time.sleep(min(wait, 30))
        else:
            raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

    raise RuntimeError("HF API failed after 3 retries.")


def query_vectordb(question: str, n_results: int = 5) -> list:
    """Query Qdrant for relevant documents."""
    client = get_qdrant_client()
    query_vector = get_embedding(question)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=n_results,
        with_payload=True,
    )
    return results


def build_context(results: list) -> str:
    """Build context string from Qdrant search results, respecting token limits."""
    if not results:
        return "No relevant data found."

    context_parts = []
    current_length = 0

    for i, hit in enumerate(results):
        doc = hit.payload.get("text", "")
        entry = f"Property {i + 1}: {doc}"
        if current_length + len(entry) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        current_length += len(entry)

    return "\n\n".join(context_parts)


def build_sources(results: list) -> list:
    """Extract source metadata from Qdrant results."""
    sources = []
    for hit in results:
        payload = hit.payload or {}
        sources.append({
            "city": payload.get("city", "Unknown"),
            "state": payload.get("state", "Unknown"),
            "property_type": payload.get("property_type", "Unknown"),
            "bhk": payload.get("bhk", "N/A"),
            "price_lakhs": payload.get("price_lakhs", "N/A"),
            "size_sqft": payload.get("size_sqft", "N/A"),
        })
    return sources


def query_with_llm(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question via HuggingFace API
    2. Query Qdrant for relevant documents
    3. Build context from results
    4. Send to Groq LLM for answer generation
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return {
            "answer": "⚠️ Groq API key not configured. Please add your API key to backend/.env file.",
            "sources": [],
            "error": True,
        }

    # Step 1: Query vector DB
    results = query_vectordb(question, n_results=5)

    # Step 2: Build context and sources
    context = build_context(results)
    sources = build_sources(results)

    # Step 3: Build prompt
    user_prompt = f"""Based on the following real estate data from India, answer the user's question.

--- PROPERTY DATA ---
{context}
--- END DATA ---

User Question: {question}

Provide a helpful, accurate answer based on the data above. If the data is insufficient, mention what information is available and suggest how the user might refine their question."""

    # Step 4: Call Groq LLM
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=MAX_ANSWER_TOKENS,
        )

        answer = chat_completion.choices[0].message.content
        return {
            "answer": answer,
            "sources": sources,
            "error": False,
        }

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return {
                "answer": "⚠️ Rate limit reached. Please wait a moment and try again.",
                "sources": sources,
                "error": True,
            }
        return {
            "answer": f"⚠️ Error generating answer: {error_msg}",
            "sources": sources,
            "error": True,
        }
