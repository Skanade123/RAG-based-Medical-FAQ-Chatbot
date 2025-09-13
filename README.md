# 🏥 Medical FAQ Assistant with RAG

[![Demo Video Thumbnail](https://drive.google.com/uc?export=view&id=1FN04MYz46PnX8Nbb6i8cQLAtzJkQChPf)](https://drive.google.com/file/d/1ZUrUgsORjv_YVDK2oWCgQAev7IxSh6AR/view?usp=sharing)

A secure, AI-powered medical FAQ assistant built with **Streamlit** and **Google's Gemini AI**.  
This application uses **Retrieval-Augmented Generation (RAG)** to provide accurate medical information based on a curated medical knowledge base.

---

## 🔍 What is RAG (Retrieval-Augmented Generation)?

RAG is an AI architecture that enhances language models by combining:

- **Retrieval**: Searching through a knowledge base to find relevant information  
- **Augmentation**: Adding retrieved context to the AI prompt  
- **Generation**: Using an LLM to generate responses based on the retrieved context  

**How RAG Works in This Application:**  
`User Query → Embed Query → Search FAISS Index → Retrieve Medical Chunks → Generate Response with Context → Return Answer + Sources`

**Benefits of RAG:**
- ✅ Provides factual, grounded responses from medical data  
- ✅ Reduces AI hallucinations by using verified sources  
- ✅ Enables domain-specific medical knowledge  
- ✅ Allows for source attribution and transparency  
- ✅ Can be updated without retraining models  

---

## 🚀 Features

- **RAG-Based Architecture**: Combines medical document retrieval with AI generation  
- **Security-First Design**: Multi-layer security with input sanitization, rate limiting, and content filtering  
- **Interactive Chat Interface**: Real-time medical Q&A with conversation history  
- **Session Management**: Secure session handling with automatic timeout  
- **Source Attribution**: Shows relevant medical sources for transparency  
- **Performance Optimized**: Uses FAISS for fast vector similarity search  
- **Data Validation**: Built-in filtering to prevent malicious content injection  

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit  
- **AI Model**: Google Gemini 2.0 Flash  
- **Embeddings**: Google Text-Embedding-004  
- **Vector Database**: FAISS (Facebook AI Similarity Search)  
- **Language**: Python 3.8+  
- **Security**: Custom input sanitization, rate limiting, and session management  

---

## 📋 Prerequisites

- Python 3.8 or higher  
- Google AI API Key from [Google AI Studio](https://aistudio.google.com)  
- CSV dataset with medical FAQ data in format: `qtype, Question, Answer`  
  - Example dataset: [Kaggle Medical Q&A](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/data)  
  - *(Using first 100 entries for free Google API version)*  

---

## 🔒 Security Architecture

This application implements multiple security layers:

### **Data Processing Security** (`utils.py`)
- Input Validation: Filters records containing suspicious patterns  
- Content Sanitization: Blocks malicious instructions in dataset  
- Column Standardization: Safe handling of different CSV formats  

### **Application Security** (`app.py`)
- Query Sanitization: Regex-based blocking of prompt injection attempts  
- Rate Limiting: 10 requests per minute per session  
- Session Management: 30-minute timeout with unique session IDs  
- Content Filtering: Removes suspicious chunks from retrieval results  
- Size Limits: Maximum 500 characters per query  

**Security Patterns Blocked:**
```python
FORBIDDEN_PATTERNS = [
    r"ignore\s+previous", r"forget", r"system\s*prompt",
    r"reveal", r"jailbreak", r"override", r"developer\s*mode",
    r"disregard\s+instructions", r"api\s*key", r"password",
    r"show\s+source", r"dump\s+data", r"admin", r"root"
]
