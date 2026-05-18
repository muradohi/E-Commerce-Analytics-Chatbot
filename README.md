# 🛒 E-Commerce AI Assistant (RAG + LLM System)

A modular AI chatbot for e-commerce analytics that combines **LLM routing, RAG retrieval, and structured data analysis** in a single system.

---

# 🚀 Features

- 🧠 **LLM Router** → Automatically routes queries to:
  - Pandas (analytics)
  - RAG (product/reviews)
  - Hybrid (both)

- 📚 **RAG System**
  - ChromaDB vector search
  - Semantic retrieval over product & review data

- 📊 **Analytics Engine**
  - Pandas-based sales and product analysis
  - Aggregations and KPI computation

- 🧪 **Evaluation Layer**
  - Faithfulness checking
  - Relevance scoring
  - Hallucination detection

- 💬 **Streamlit Chat UI**
  - ChatGPT-style interface
  - Conversation memory
  - Evaluation display

- ⚙️ **Config-Driven**
  - YAML-based configuration
  - Easy model switching (GPT-4o / GPT-4o-mini)

---

# 🏗️ Architecture

User Query → Streamlit UI → LLM Router  
→ (Pandas | RAG | Hybrid) → LLM Response  
→ Evaluation Layer → Final Output

---

# ⚙️ Tech Stack

Python • LangChain • OpenAI • ChromaDB • Pandas • Streamlit • YAML

---

# 📦 Setup
'''
uv sync
streamlit run app.py
'''

# 🧠 What I Did

## 1. Built an LLM Routing System
I implemented a smart routing layer that uses an LLM to classify user queries into:
- **Pandas** → numerical analytics (sales, revenue, trends)
- **RAG** → product descriptions and customer reviews
- **Hybrid** → combines structured + unstructured reasoning

---

## 2. Developed a RAG Pipeline
- Converted product and review data into embeddings
- Stored them in a **ChromaDB vector database**
- Implemented semantic retrieval for context-aware answering
- Used MMR retrieval to improve diversity and relevance

---

## 3. Created a Structured Analytics Engine
- Built a Pandas-based backend for structured queries
- Supported aggregations like revenue, counts, and product performance
- Connected raw CSV data into a queryable analytical layer

---

## 4. Integrated an LLM Response System
- Used GPT-4o / GPT-4o-mini for answer generation
- Designed prompts for:
  - strict context grounding
  - explanation of analytics results
  - hybrid reasoning between data and text

---

## 5. Built an Evaluation Layer
- Implemented an automated evaluation system for each response
- Measured:
  - Faithfulness (is answer grounded in context?)
  - Relevance (does it answer the question?)
  - Hallucination detection
- Used this to monitor LLM output quality

---

## 6. Designed a Config-Driven Architecture
- Introduced YAML configuration system
- Allowed dynamic switching of:
  - LLM models
  - embeddings
  - system parameters
- Eliminated hardcoding for better experimentation

---

## 7. Developed a Streamlit Chat Interface
- Built a ChatGPT-style UI
- Added session-based memory
- Displayed evaluation results alongside answers
- Created interactive chatbot experience for users

---

# ⚙️ Result

This system behaves like a **real-world AI assistant for e-commerce**, capable of:
- answering business analytics questions
- explaining product insights
- combining structured and unstructured knowledge
- evaluating its own outputs for quality control

---

# 🚀 Outcome

This project demonstrates my ability to:
- design full LLM systems (not just models)
- build RAG pipelines from scratch
- integrate analytics + AI reasoning
- implement evaluation and monitoring layers
- structure production-grade AI applications
