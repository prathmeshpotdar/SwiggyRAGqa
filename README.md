# Swiggy Annual Report — RAG QA System

A Retrieval-Augmented Generation (RAG) based Question Answering system that allows users to ask natural language questions and receive accurate, context-grounded answers strictly based on Swiggy’s Annual Report.

This project uses semantic search with vector embeddings and a Large Language Model to ensure reliable, non-hallucinated responses.

---

## 🚀 Objective

The goal of this project is to:

- Build a document-grounded AI assistant
- Enable semantic search over Swiggy’s Annual Report
- Prevent hallucinations by restricting responses to retrieved context
- Provide an interactive web-based interface using Gradio

---

## 📚 Data Source

**Swiggy Annual Report (Public Document)**

Source Link:  
https://www.swiggy.com/investor-relations  

> Note: Due to copyright considerations, the PDF file is not included in this repository.  
Please download the latest Swiggy Annual Report and place it inside the `data/` folder.

---

## 🧠 System Architecture
PDF Document
↓
PDF Loader (PyPDF)
↓
Text Chunking
↓
Embedding Model (OpenAI)
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
LLM (GPT)
↓
Answer + Source Context


---

## 🛠 Tech Stack

- Python 3.10+
- LangChain
- OpenAI Embeddings
- FAISS Vector Database
- Gradio (Web UI)
- PyPDF
- dotenv

---



---

## ⚙ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/prathmeshpotdar/SwiggyRAGqa.git
cd SwiggyRAGqa

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Add Environment Variable

Create a .env file:

OPENAI_API_KEY=your_openai_api_key

4️⃣ Add PDF File

Create a folder:

data/


Place the downloaded Swiggy Annual Report PDF inside:

data/swiggy_annual_report.pdf

5️⃣ Generate Vector Database

Run ingestion pipeline:

python ingest.py


This will:

Load PDF

Chunk text

Generate embeddings

Store vectors in FAISS

6️⃣ Launch Web UI (Gradio)
python gradio_app.py


Open in browser:

http://localhost:7860

💬 Example Questions

Try asking:

What was Swiggy’s total revenue in FY23?

What are Swiggy’s main business segments?

What risks are highlighted in the annual report?

What operational achievements are mentioned?

🛡 Hallucination Prevention Strategy

The system avoids hallucinations using:

Context-only prompting

Zero temperature inference

Semantic retrieval filtering

Fallback response when answer is unavailable

Source context display for transparency

If the answer is not found in the document, the system responds:

I could not find this information in the Swiggy Annual Report.

🌐 User Interface

The application includes a browser-based interface built with Gradio:

Features:

Natural language input

Instant AI response

Supporting context display

Lightweight and fast UI

📈 Key Features

Semantic document search

Retrieval-Augmented Generation pipeline

Financial document QA support

Source grounded responses

Web UI support

Modular and extensible architecture

📌 Future Improvements

Multi-document support

Chat-style conversational interface

PDF upload via UI

Source highlighting

RAG evaluation metrics

Docker deployment
