"""
RAG Engine: Query ChromaDB and generate answers using Groq LLM.
"""

import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "chroma_db")
COLLECTION_NAME = "housing_data"

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


def get_chroma_collection():
    """Get the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def query_vectordb(question: str, n_results: int = 5) -> dict:
    """Query ChromaDB for relevant documents."""
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    return results


def build_context(results: dict) -> str:
    """Build context string from ChromaDB query results, respecting token limits."""
    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant data found."

    documents = results["documents"][0]
    context_parts = []
    current_length = 0

    for i, doc in enumerate(documents):
        entry = f"Property {i + 1}: {doc}"
        if current_length + len(entry) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        current_length += len(entry)

    return "\n\n".join(context_parts)


def build_sources(results: dict) -> list:
    """Extract source metadata from results."""
    if not results or not results.get("metadatas") or not results["metadatas"][0]:
        return []

    sources = []
    for meta in results["metadatas"][0]:
        sources.append({
            "city": meta.get("city", "Unknown"),
            "state": meta.get("state", "Unknown"),
            "property_type": meta.get("property_type", "Unknown"),
            "bhk": meta.get("bhk", "N/A"),
            "price_lakhs": meta.get("price_lakhs", "N/A"),
            "size_sqft": meta.get("size_sqft", "N/A"),
        })
    return sources


def query_with_llm(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Query ChromaDB for relevant documents
    2. Build context from results
    3. Send to Groq LLM for answer generation
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

    # Step 2: Build context
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
        # Handle rate limiting gracefully
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
