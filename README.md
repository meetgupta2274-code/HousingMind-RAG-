# 🏠 India Housing RAG — AI Real Estate Assistant

RAG-powered Q&A over 250K Indian housing records using ChromaDB + Groq LLM + React UI.

---

## ⚡ Quick Setup

### 1. Add your Groq API key

Open `backend/.env` and replace the placeholder:
```
GROQ_API_KEY=your_actual_groq_api_key_here
```
Get your free key at: https://console.groq.com

---

### 2. Run the Backend

```powershell
# From the project root
.\venv\Scripts\activate

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: **http://localhost:8000**

---

### 3. Run the Frontend

Open a **new terminal**:

```powershell
cd frontend
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

### 4. Initialize the Database (first time only)

1. Open **http://localhost:5173** in your browser
2. Click the **"Initialize DB"** button in the top-right corner
3. Wait a few minutes while the data is embedded (**one-time only**)
4. Once done, start asking questions! 🎉

> **The vector store is persisted** in `chroma_db/` — you never need to re-embed again.

---

## 🗂️ Project Structure

```
house_price_prediction-Rag/
├── venv/                  ← Python virtual environment
├── data/
│   └── india_housing_prices.csv
├── chroma_db/             ← Persistent vector store (auto-created)
├── backend/
│   ├── .env               ← Your Groq API key
│   ├── ingest.py          ← CSV → ChromaDB (batch of 30)
│   ├── rag_engine.py      ← ChromaDB + Groq LLM
│   └── main.py            ← FastAPI server
└── frontend/
    └── src/App.jsx        ← Chat UI
```

## 🧠 How It Works

1. **Ingest**: Reads CSV, samples 5,000 representative rows (stratified by State), converts each row to a natural-language text chunk, embeds in batches of 30 into ChromaDB
2. **Query**: User question → ChromaDB semantic search → top-5 relevant property docs → Groq LLM generates answer
3. **Persist**: ChromaDB stores embeddings to disk so you only embed once

## 📌 Sample Questions

- _"What is the average price of a 3 BHK apartment in Mumbai?"_
- _"Which cities have the most affordable housing under ₹40 Lakhs?"_
- _"Show me furnished flats in Bangalore with parking"_
- _"What is the price per sqft in South Delhi?"_
