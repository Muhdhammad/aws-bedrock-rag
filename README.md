# Cloud-Native RAG Application

*A Retrieval-Augmented Generation system built on AWS S3, FAISS, LangChain, and HuggingFace embeddings.*

This project implements a fully functional **Retrieval-Augmented Generation (RAG)** pipeline designed for scalable, cloud-native document search and question answering.
It features **dual Streamlit applications** (Admin + User), seamless ingestion from **AWS S3**, vector search with **FAISS**, and **MiniLM embeddings** for fast, accurate retrieval.

The entire system is **containerized with Docker** for consistent deployment anywhere.

---

## 🚀 Features

### 🔹 **1. Dual Streamlit Applications**

#### **Admin Interface**

* Upload PDFs from local system or directly from AWS S3
* Process and split documents into chunks
* Generate embeddings using **HuggingFace MiniLM**
* Build and store FAISS vector index
* Manage document ingestion lifecycle

#### **User Interface**

* Query the knowledge base using natural language
* Retrieves relevant document chunks using semantic search
* Combines context + LLM to generate accurate answers
* Clean, simple user-friendly interface

---

### 🔹 **2. Cloud-Native Storage with AWS S3**

* Stores raw PDFs and processed files
* Persistent storage for uploaded documents
* Easy integration with S3 buckets (public or private)

---

### 🔹 **3. Vector Search with FAISS**

* Efficient semantic search using **FAISS Flat Index**
* Supports thousands of document embeddings
* Blazing-fast similarity search
* Stored locally for quick access

---

### 🔹 **4. Embeddings via HuggingFace MiniLM**

* Uses **sentence-transformers/all-MiniLM-L6-v2**
* Lightweight + fast + high-accuracy
* Ideal for real-time RAG applications

---

### 🔹 **5. Fully Containerized (Docker)**

* Both admin and user Streamlit apps run in separate containers
* Ensures identical behavior across environments
* Easy to deploy on:

  * AWS EC2
  * AWS ECS / Fargate
  * Docker Compose
  * Kubernetes

---

## 🧱 System Architecture

```
             ┌──────────────────────────────┐
             │            User App           │
             │     (Query & Q/A Retrieval)   │
             └───────────────▲──────────────┘
                             │ query
                             │
             ┌───────────────┴──────────────┐
             │         RAG Pipeline          │
             │  MiniLM Embeddings + FAISS    │
             └───────────────▲──────────────┘
                             │ context
                             │
             ┌───────────────┴──────────────┐
             │         Admin App             │
             │ Upload → Chunk → Embed → Save │
             └───────────────▲──────────────┘
                             │
                             │ Docs
                             │
                   ┌─────────┴────────┐
                   │     AWS S3        │
                   │  Document Storage │
                   └───────────────────┘
```

---

## 🛠️ Tech Stack

### **Languages & Frameworks**

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Sentence Transformers

### **Cloud**

* AWS S3

### **Containerization**

* Docker
* Docker Compose

---


## ▶️ Running the Project Locally

### **1. Clone the repo**

```bash
git clone https://github.com/your-username/aws-rag.git
cd aws-rag
```

### **2. Add AWS credentials**

Set environment variables or use `.env` file:

```
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
AWS_REGION=your-region
AWS_BUCKET=my-bucket
```

### **3. Build Docker Containers**

```bash
docker-compose up --build
```

---

## 📊 RAG Workflow Summary

1. **Admin uploads documents** (PDFs)
2. System **loads → splits → embeds** documents
3. **FAISS index** created and saved
4. User enters question
5. System retrieves **top-k relevant chunks**
6. LLM generates contextual answer
7. User sees final response

---

## 📌 Future Enhancements

* Option for other embedding and llm model
* Use Postgres with pgvector for better scalibilty 
* Add authentication
* Deploy on AWS ECS with load balancing

