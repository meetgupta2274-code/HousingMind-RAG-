# HousingMind RAG 🏠

A full-stack Retrieval-Augmented Generation (RAG) application for querying Indian housing prices using natural language. Ask questions like *"What is the average price of a 3 BHK in Mumbai?"* and get real, data-backed answers.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Frontend (Vite + React)
     │  HTTPS (axios)
     ▼
Backend (FastAPI on Render)
     │
     ├──► HuggingFace Inference API  ──► Embedding (384-dim vector)
     │         (all-MiniLM-L6-v2)
     │
     ├──► Qdrant Cloud  ──► Vector Search (top-5 similar properties)
     │
     └──► Groq LLM API  ──► Answer Generation (llama-3.3-70b-versatile)
```

---

## ✨ Features

- **Natural language queries** over 4,000+ real Indian housing listings
- **RAG pipeline**: retrieves relevant property data before generating answers
- **Dark glassmorphism UI** with smooth animations, typing indicators, and source chips
- **Suggested questions** on the welcome screen
- **Toast notifications** and real-time status indicators
- **Cloud-native architecture**: no heavy models loaded on the server (RAM-efficient)
- **Persistent vector storage** on Qdrant Cloud (no re-embedding on server restart)
- **Uptime monitoring endpoint** at `/health`

---

## 🗂️ Project Structure

```
house_price_prediction-Rag/
├── backend/
│   ├── main.py           # FastAPI app — all API endpoints
│   ├── ingest.py         # CSV ingestion → HF embeddings → Qdrant
│   ├── rag_engine.py     # RAG pipeline: embed query → Qdrant search → Groq LLM
│   ├── requirements.txt  # Python dependencies
│   └── .env              # Environment variables (NOT committed to git)
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component (chat UI, state, API calls)
│   │   ├── index.css     # Full design system (dark theme, glassmorphism, animations)
│   │   └── main.jsx      # React entry point
│   ├── index.html        # HTML template with meta tags + Google Fonts
│   ├── vite.config.js    # Vite configuration
│   ├── package.json      # Frontend dependencies
│   └── .env              # Frontend env — VITE_API_BASE_URL (NOT committed)
├── data/
│   └── india_housing_prices.csv   # Source dataset (250,000 rows)
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vite + React, Axios, react-markdown |
| Backend | FastAPI, Uvicorn |
| Vector DB | Qdrant Cloud (managed, free tier) |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`, 384-dim) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Dataset | India Housing Prices CSV (250K rows → ~5K sampled) |
| Deployment | Render.com (backend: Web Service, frontend: Static Site) |

---

## 🔑 Environment Variables

### `backend/.env`
```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_read_token
```

### `frontend/.env`
```env
# For local development
VITE_API_BASE_URL=http://localhost:8000/api

# Change to your Render backend URL when deploying
# VITE_API_BASE_URL=https://your-backend-name.onrender.com/api
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.12
- Node.js 18+
- Free accounts on: [Groq](https://console.groq.com), [Qdrant Cloud](https://cloud.qdrant.io), [HuggingFace](https://huggingface.co)

### 1. Clone and set up the backend
```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Create backend/.env with your 4 keys (see Environment Variables above)

# Start the backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Set up the frontend
```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### 3. Initialize the database (first time only)
1. Click **"Initialize DB"** in the app header
2. Wait 5–15 minutes while ~5,000 properties are embedded and stored in Qdrant Cloud
3. Status will change to `ready` — you can now ask questions

> ⚠️ Ingestion only runs once. Qdrant Cloud stores the vectors permanently. Restarting the server does **not** require re-ingestion.

---

## 🌐 Deployment on Render

### Backend — Web Service

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Python Version | `3.12.10` (set via `PYTHON_VERSION` env var) |

**Environment Variables (set in Render dashboard):**
- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `HF_TOKEN`
- `PYTHON_VERSION` = `3.12.10`

### Frontend — Static Site

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |

**Environment Variables:**
- `VITE_API_BASE_URL` = `https://your-backend-name.onrender.com/api`

**Redirects/Rewrites** (Render dashboard → Redirects tab):
- Source: `/*` → Destination: `/index.html` → Action: `Rewrite`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Backend status, ingestion status, doc count |
| `POST` | `/api/ingest` | Trigger CSV ingestion into Qdrant Cloud |
| `POST` | `/api/query` | Query RAG pipeline with a natural language question |
| `GET/HEAD` | `/health` | Lightweight uptime monitor endpoint |

### Query Example
```bash
curl -X POST https://your-backend.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average price of a 3 BHK in Mumbai?"}'
```

---

## 📊 Data Processing

- **Source**: `india_housing_prices.csv` — 250,000 rows of Indian real estate data
- **Sampling**: Stratified by `State` — ~5,000 representative rows across all 20 states
- **Chunking**: Each row is converted to a natural language description (city, BHK, price, sqft, amenities, etc.)
- **Embedding**: Batches of 30 texts sent to HuggingFace Inference API → 384-dimensional vectors
- **Storage**: Vectors stored in Qdrant Cloud collection `housing_data`

---

## ⚠️ Known Limitations

- **HuggingFace free tier rate limits**: Ingestion may stop mid-way if rate limits are hit. Partial data already in Qdrant is fully usable.
- **Render free tier cold starts**: The backend sleeps after 15 minutes of inactivity. First request after sleep may take 30–60 seconds. Use an uptime monitor (e.g., UptimeRobot) pinging `/health` every 5 minutes.
- **Ephemeral filesystem on Render**: Do not use local ChromaDB on Render. This project uses Qdrant Cloud specifically to avoid this limitation.
- **Ingestion time**: 5–15 minutes depending on your network latency to HuggingFace and Qdrant Cloud servers.
