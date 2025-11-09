# Claims Micro-RAG System

A conversational AI-powered Retrieval-Augmented Generation (RAG) system for insurance claims processing. Built with FastAPI, Semantic Kernel, and Azure OpenAI.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **RAG System**: Retrieval-Augmented Generation using FAISS vector database
- **Conversational Agent**: Context-aware chat using Semantic Kernel and Azure OpenAI
- **Smart Clarification**: LLM-powered detection of ambiguous queries
- **PII Protection**: Automatic Aadhaar number masking
- **Conversation History**: Persistent chat history with context maintenance
- **Grounding Score**: Cosine similarity scoring for answer confidence
- **Health Monitoring**: `/healthz` endpoint with system metrics
- **Logging Middleware**: Request/response logging with latency tracking

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌───────────┐   ┌───────────┐ │
│  │ /ask API  │   │ /agent/*  │ │
│  └─────┬─────┘   └─────┬─────┘ │
│        │               │        │
│  ┌─────▼──────────────▼─────┐  │
│  │   Retrieval Engine       │  │
│  │  (FAISS + Embeddings)    │  │
│  └──────────────────────────┘  │
│        │               │        │
│  ┌─────▼─────┐   ┌─────▼─────┐ │
│  │   Index   │   │  Semantic │ │
│  │ Metadata  │   │  Kernel   │ │
│  └───────────┘   └─────┬─────┘ │
│                        │        │
│                   ┌────▼─────┐  │
│                   │  Azure   │  │
│                   │  OpenAI  │  │
│                   └──────────┘  │
└─────────────────────────────────┘
```

## 📦 Prerequisites

- **Python**: 3.13
- **Azure OpenAI**: Access to Azure OpenAI service with GPT-4 deployment
- **Docker** (Optional) : Docker Desktop or Docker Engine for containerized deployment
- **OS**: Windows, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data Files

Create a `data/` directory and add your text files:

```bash
mkdir data
# Add your .txt files to the data/ directory
# Example: claims_processing.txt, policy_101.txt, policy_201.txt
```

## ⚙️ Configuration

### 1. Create `.env` File

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# Data Configuration
DATA_DIR=data/
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=100

# API Configuration
ASK_API_URL=http://localhost:8000/ask
EVAL_FILE=eval.jsonl
```

### 2. Verify Configuration

```bash
# Check if .env is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Endpoint:', os.getenv('AZURE_OPENAI_ENDPOINT'))"
```

## 🏃 Running the Application

### 1. Start the Server

```bash
uvicorn app:app --reload
```

**Output:**
```
INFO - ================================================================================
INFO - Starting Claims Micro-RAG Application
INFO - ================================================================================
INFO - Loading documents from data/...
INFO -   Processing: faq_claims.txt
INFO -     Created 1 chunks
INFO -   Processing: fraud_signals.txt
INFO -     Created 1 chunks
INFO -   Processing: hospitals.txt
INFO -     Created 1 chunks
INFO -   Processing: kyc_rules.txt
INFO -     Created 1 chunks
INFO -   Processing: policy_101.txt
INFO -     Created 1 chunks
INFO -   Processing: policy_201.txt
INFO -     Created 1 chunks
INFO - Encoding 6 chunks...
INFO:     ✓ Indexed 6 chunks from 6 files.
INFO:     Application ready!
```

### 2. Verify Server is Running by doing health check

### 1. Health Check

**Endpoint:** `GET /healthz`

```bash
curl http://localhost:8000/healthz
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T10:30:45.123456",
  "uptime_seconds": 123.45,
  "index_ready": true,
  "documents_indexed": 45,
  "embedding_model": "sentence-transformers/all-mpnet-base-v2",
  "chunk_size": 400,
  "chunk_overlap": 100,
  "version": "1.0.0"
}
```

### 2. Ask Endpoint (RAG Query)

**Endpoint:** `POST /ask`

**Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the TAT for claim processing?",
  "k": 3
}'
```

**Response:**
```json
{
  "answer": "How to file a claim: upload discharge summary, itemized bill, and ID proof. Typical TAT 7–10 business days. Policy 101: Sum insured ₹5,00,000. Hospital network: CareWell, HealOne. Pre-auth required for planned admissions; emergency claims allowed 24×7. Network hospitals: CareWell (Chennai), HealOne (Pune), MedAxis (Bengaluru). Non-network claims need reimbursement route.",
  "citations": [
    {
      "doc": "faq_claims.txt",
      "snippet": "How to file a claim: upload discharge summary,",
      "full_snippet": "How to file a claim: upload discharge summary, itemized bill, and ID proof. Typical TAT 7–10 business days."
    },
    {
      "doc": "policy_101.txt",
      "snippet": "Policy 101: Sum insured ₹5,00,000. Hospital network: CareWell,",
      "full_snippet": "Policy 101: Sum insured ₹5,00,000. Hospital network: CareWell, HealOne. Pre-auth required for planned admissions; emergency claims allowed 24×7."
    },
    {
      "doc": "hospitals.txt",
      "snippet": "Network hospitals: CareWell (Chennai), HealOne (Pune), MedAxis (Bengaluru).",
      "full_snippet": "Network hospitals: CareWell (Chennai), HealOne (Pune), MedAxis (Bengaluru). Non-network claims need reimbursement route."
    }
  ],
  "retrieval": {
    "k": 3,
    "latency_ms": 162,
    "grounding_score": 0.3682
  }
}
```

### 3. Agent Chat (Conversational)

**Endpoint:** `POST /agent/chat`

**Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/agent/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "What is the room rent limit under policy 201?",
  "session_id": "default",
  "k": 3
}'
```

**Response:**
```json
{
  "reply": "The room rent limit under Policy 201 is ₹5,000 per day. \n\nAdditionally, a 10% co-pay applies to claims for customers over 60 years of age. If you need help with claim submission or understanding your coverage better, feel free to reach out!",
  "citations": [
    {
      "doc": "policy_201.txt",
      "snippet": "Policy 201: Sum insured ₹10,00,000. Room rent limit"
    },
    {
      "doc": "policy_101.txt",
      "snippet": "Policy 101: Sum insured ₹5,00,000. Hospital network: CareWell,"
    },
    {
      "doc": "hospitals.txt",
      "snippet": "Network hospitals: CareWell (Chennai), HealOne (Pune), MedAxis (Bengaluru)."
    }
  ],
  "retrieval": {
    "k": 3,
    "latency_ms": 175,
    "grounding_score": 0.4328
  },
  "session_id": "default",
  "used_clarification": false,
  "confidence_score": 1
}
```

### 4. Get Chat History

**Endpoint:** `GET /agent/history/{session_id}`

```bash
curl -X 'GET' \
  'http://localhost:8000/agent/history/user1' \
  -H 'accept: application/json'
```

**Response:**
```json
{
  "session_id": "user1",
  "history": [
    {
      "role": "assistant",
      "text": "Can you please specify the policy number for which you would like to know the network hospitals?"
    },
    {
      "role": "user",
      "text": "Name two network hospitals in the policies."
    },
    {
      "role": "assistant",
      "text": "The room rent limit under Policy 201 is ₹5,000 per day. \n\nThis policy has a sum insured of ₹10,00,000 and includes a 10% co-pay requirement for customers over 60 years of age. If you have further questions about this policy or need assistance with claims, please let me know!"
    },
    {
      "role": "user",
      "text": "What is the room rent limit under policy 201?"
    }
  ]
}
```

### 5. Clear Chat History

**Endpoint:** `DELETE /agent/history/{session_id}`

```bash
curl -X DELETE http://localhost:8000/agent/history/user123
```

**Response:**
```json
{
  "status": "success",
  "message": "History cleared for session: user123"
}
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

```
## 📊 Evaluation

### Run Evaluation Script

```bash
python eval.py
```

**Output:**
```
================================================================================
EVALUATION RESULTS
================================================================================

[1] Query: What is the TAT for claim processing?
    Expected: '7-10'
    Matches: 0/3
    Status: ✗ MISS
    Retrieved snippets:
      [1] faq_claims.txt: How to file a claim: upload discharge summary,...
      [2] policy_101.txt: Policy 101: Sum insured ₹5,00,000. Hospital network: CareWel...
      [3] hospitals.txt: Network hospitals: CareWell (Chennai), HealOne (Pune), MedAx...

[2] Query: Name two network hospitals in the policies.
    Expected: 'CareWell, HealOne'
    Matches: 1/3
    Status: ✓ HIT

[3] Query: Is Aadhaar allowed as KYC and how must it appear?
    Expected: 'masked'
    Matches: 1/3
    Status: ✓ HIT

[4] Query: Room rent limit under policy 201?
    Expected: '₹5,000'
    Matches: 1/3
    Status: ✓ HIT

[5] Query: List one fraud signam mentioned.
    Expected: 'duplicate'
    Matches: 1/3
    Status: ✓ HIT

================================================================================
SUMMARY: n=5 | hit_rate=0.80 | precision@3=0.27
================================================================================
```

### Evaluation Metrics

- **hit_rate**: Percentage of queries with at least one correct answer
- **precision@k**: Average number of correct chunks in top-k results
- **grounding_score**: Average cosine similarity (answer confidence)

## 📁 Project Structure

```
project/
├── app.py                  # Main FastAPI application
├── agent.py                # Conversational agent with Semantic Kernel
├── aadhaar_mask.py         # PII masking utility
├── history_db.py           # Conversation history persistence
├── eval.py                 # Evaluation script
├── eval.jsonl              # Evaluation dataset
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── data/                   # Document corpus (create this)
│   ├── policy_101.txt
│   ├── policy_201.txt
│   └── claims_processing.txt
├── tests/                  # Unit tests
│   └── test_app.py
└── README.md              # This file
```

## 🐛 Troubleshooting

### Issue: Index Not Ready (503 Error)

**Problem:** Server returns 503 when calling `/ask`

**Solution:**
```bash
# Check server logs for index build status
# Ensure data/ directory contains .txt files
ls data/

# Restart server
uvicorn app:app --reload
```

### Issue: Low Precision/Hit Rate

**Problem:** Evaluation shows poor results

**Solution:**
1. Increase chunk size/overlap in `.env`:
   ```env
   CHUNK_SIZE=400
   CHUNK_OVERLAP=100
   ```

2. Add more data to `data/` directory

3. Try using better embedding model:
   ```env
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   ```
## 📚 API Documentation

Interactive API documentation is available at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔒 Security Notes

- Never commit `.env` file to version control
- Rotate Azure OpenAI API keys regularly
- Validate all user inputs
- Use HTTPS in production
- Implement rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review troubleshooting section

## 🎯 Next Steps

1. **Production Deployment:**
   - Use Gunicorn/Uvicorn with multiple workers
   - Add reverse proxy (nginx)
   - Implement rate limiting
   - Add authentication

2. **Enhancements:**
   - Add more evaluation metrics
   - Implement BM25 hybrid search
   - Add streaming responses
   - Create web UI

3. **Monitoring:**
   - Add Prometheus metrics
   - Implement distributed tracing
   - Set up alerting

---

**Version:** 1.0.0  
**Last Updated:** November 9, 2025