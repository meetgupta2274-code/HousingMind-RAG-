import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

const SUGGESTED_QUESTIONS = [
  { emoji: '💰', text: 'What is the average price of a 3 BHK apartment in Mumbai?' },
  { emoji: '🏙️', text: 'Which cities in Maharashtra have the most affordable housing?' },
  { emoji: '📐', text: 'What is the price per sqft in prime areas of Delhi?' },
  { emoji: '🛋️', text: 'Show me furnished apartments under ₹50 Lakhs in Bangalore' },
  { emoji: '🏗️', text: 'Which states have the highest housing prices on average?' },
  { emoji: '🏫', text: 'Find properties in Chennai with good nearby school access' },
]

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// ---- TypingIndicator ----
function TypingIndicator() {
  return (
    <div className="message-wrapper assistant">
      <div className="message-header">
        <div className="message-avatar ai-avatar">🏠</div>
        <span className="message-name">AI Assistant</span>
      </div>
      <div className="typing-bubble">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  )
}

// ---- ChatMessage ----
function ChatMessage({ role, content, sources, timestamp, hasError }) {
  const isUser = role === 'user'
  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-header">
        {!isUser && <div className="message-avatar ai-avatar">🏠</div>}
        <span className="message-name">{isUser ? 'You' : 'AI Assistant'}</span>
        <span className="message-time">{formatTime(timestamp)}</span>
        {isUser && <div className="message-avatar user-avatar">👤</div>}
      </div>
      <div className={`message-bubble ${isUser ? 'user' : 'assistant'} ${hasError ? 'error-bubble' : ''}`}>
        {isUser ? (
          <span>{content}</span>
        ) : (
          <ReactMarkdown>{content}</ReactMarkdown>
        )}
        {!isUser && sources && sources.length > 0 && (
          <div className="sources-strip">
            <span className="sources-label">📍 Referenced properties</span>
            {sources.map((s, i) => (
              <span key={i} className="source-chip">
                {s.bhk} BHK · {s.city}, {s.state} · ₹{s.price_lakhs}L
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ---- Toast ----
function Toast({ toast, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000)
    return () => clearTimeout(timer)
  }, [onClose])

  return (
    <div className={`toast ${toast.type}`}>
      <span>{toast.icon}</span>
      <span>{toast.message}</span>
    </div>
  )
}

// ---- Main App ----
export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isIngesting, setIsIngesting] = useState(false)
  const [dbStatus, setDbStatus] = useState({ status: 'checking', doc_count: 0 })
  const [toast, setToast] = useState(null)
  const chatEndRef = useRef(null)
  const inputRef = useRef(null)

  const showToast = useCallback((message, type = 'info', icon = 'ℹ️') => {
    setToast({ message, type, icon })
  }, [])

  // Check health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  async function checkHealth() {
    try {
      const res = await axios.get(`${API_BASE}/health`, { timeout: 15000 })
      setDbStatus(res.data)
    } catch {
      setDbStatus({ status: 'error', ingestion_status: 'offline', doc_count: 0 })
    }
  }

  async function handleIngest() {
    setIsIngesting(true)
    showToast('Starting data ingestion... this may take a few minutes.', 'info', '⏳')
    try {
      const res = await axios.post(`${API_BASE}/ingest`, {}, { timeout: 600000 })
      setDbStatus(d => ({ ...d, ingestion_status: 'ready', doc_count: res.data.count }))
      showToast(`✅ ${res.data.message}`, 'success', '🎉')
    } catch (err) {
      showToast('Ingestion failed. Check backend logs.', 'error', '❌')
    } finally {
      setIsIngesting(false)
      await checkHealth()
    }
  }

  async function sendMessage(text) {
    const question = (text || input).trim()
    if (!question || isLoading) return
    setInput('')

    const userMsg = {
      role: 'user',
      content: question,
      timestamp: new Date(),
      sources: null,
      hasError: false,
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const res = await axios.post(
        `${API_BASE}/query`,
        { question },
        { timeout: 60000 }
      )
      const aiMsg = {
        role: 'assistant',
        content: res.data.answer,
        timestamp: new Date(),
        sources: res.data.sources || [],
        hasError: res.data.error || false,
      }
      setMessages(prev => [...prev, aiMsg])
    } catch (err) {
      const errMsg = err.response?.data?.detail || 'Network error. Is the backend running?'
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ ${errMsg}`,
          timestamp: new Date(),
          sources: [],
          hasError: true,
        },
      ])
    } finally {
      setIsLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function handleClear() {
    setMessages([])
    inputRef.current?.focus()
  }

  // ---- Status badge helper ----
  const statusInfo = (() => {
    if (dbStatus.ingestion_status === 'ready')
      return { cls: 'ready', label: `${dbStatus.doc_count.toLocaleString()} docs loaded` }
    if (dbStatus.ingestion_status === 'not_ingested')
      return { cls: 'error', label: 'Not initialized' }
    if (dbStatus.status === 'error')
      return { cls: 'error', label: 'Backend offline' }
    return { cls: 'loading', label: 'Checking...' }
  })()

  const canQuery = dbStatus.ingestion_status === 'ready' && !isLoading

  return (
    <>
      {/* Animated background orbs */}
      <div className="bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <div className="app-container">
        {/* ---- Header ---- */}
        <header className="header">
          <div className="header-logo">
            <div className="logo-icon">🏠</div>
            <div>
              <div className="header-title">India Housing RAG</div>
              <div className="header-subtitle">AI Real Estate Intelligence</div>
            </div>
          </div>
          <div className="header-right">
            {messages.length > 0 && (
              <button className="clear-btn" onClick={handleClear}>
                Clear chat
              </button>
            )}
            <div className={`status-badge ${statusInfo.cls}`}>
              <span className="status-dot" />
              {statusInfo.label}
            </div>
            {dbStatus.ingestion_status !== 'ready' && (
              <button
                className="ingest-btn"
                onClick={handleIngest}
                disabled={isIngesting}
              >
                {isIngesting ? 'Initializing...' : 'Initialize DB'}
              </button>
            )}
          </div>
        </header>

        {/* ---- Chat area ---- */}
        <div className="chat-area">
          {isIngesting && (
            <div className="ingest-progress">
              <span className="spinner" />
              Embedding housing data into ChromaDB. This runs once and takes a few minutes...
            </div>
          )}

          {messages.length === 0 ? (
            /* Welcome screen */
            <div className="welcome-screen">
              <div className="welcome-icon">🏡</div>
              <div>
                <h1 className="welcome-title">India Housing Intelligence</h1>
                <p className="welcome-subtitle">
                  Ask me anything about housing prices, localities, property types, and real estate
                  trends across India — powered by RAG + Groq AI.
                </p>
              </div>
              <div className="welcome-stats">
                <div className="stat-chip"><span>🗃️</span> 250K+ properties</div>
                <div className="stat-chip"><span>🌏</span> Pan-India data</div>
                <div className="stat-chip"><span>⚡</span> Groq-powered LLM</div>
              </div>
              <p className="suggested-title">Try asking…</p>
              <div className="suggested-grid">
                {SUGGESTED_QUESTIONS.map((q, i) => (
                  <button
                    key={i}
                    className="suggestion-card"
                    onClick={() => sendMessage(q.text)}
                    disabled={!canQuery}
                  >
                    <span className="suggestion-emoji">{q.emoji}</span>
                    <span className="suggestion-text">{q.text}</span>
                  </button>
                ))}
              </div>
              {dbStatus.ingestion_status !== 'ready' && (
                <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                  ⚠️ Click <strong style={{ color: 'var(--text-secondary)' }}>Initialize DB</strong> above to set up the vector database first.
                </p>
              )}
            </div>
          ) : (
            /* Chat messages */
            <>
              {messages.map((msg, i) => (
                <ChatMessage key={i} {...msg} />
              ))}
              {isLoading && <TypingIndicator />}
            </>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* ---- Input area ---- */}
        <div className="input-area">
          <div className="input-container">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder={
                canQuery
                  ? 'Ask about Indian real estate prices, locations, property types…'
                  : dbStatus.ingestion_status === 'not_ingested'
                  ? 'Initialize the database first to start asking questions…'
                  : 'Connecting to backend…'
              }
              value={input}
              onChange={e => {
                setInput(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
              }}
              onKeyDown={handleKeyDown}
              disabled={!canQuery}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={!canQuery || !input.trim()}
              title="Send (Enter)"
            >
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
          <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
        </div>
      </div>

      {toast && <Toast toast={toast} onClose={() => setToast(null)} />}
    </>
  )
}
