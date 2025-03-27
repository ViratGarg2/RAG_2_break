# Retrieval-Augmented Generation (RAG) Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, and Google Gemini. It extracts text from PDFs, stores vector embeddings, and retrieves relevant chunks to improve question answering.

## Features
- PDF processing and text extraction
- Chunking and vector storage with ChromaDB
- Retrieval-based question answering using LangChain
- Comparison between RAG and LLM-only responses
- Response saving and output management

## Demo video
https://www.loom.com/share/c66e2f425bd04012be228dc9b6791c2e?sid=a5acc261-923a-4bec-8813-3c005a084d6a

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Virtual environment support (venv)

## Installation
1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install langchain langchain_community langchain_google_genai torch sentence-transformers
   ```

## Setting Up API Key for Gemini
To use Google Gemini, you need an API key:
1. Obtain a Google Gemini API key from [Google AI](https://ai.google.com/)
2. Set the API key in your environment:
   - **Linux/macOS:**
     ```bash
     export GOOGLE_API_KEY="your_api_key_here"
     ```
   - **Windows (Command Prompt):**
     ```cmd
     set GOOGLE_API_KEY="your_api_key_here"
     ```
   - **Windows (PowerShell):**
     ```powershell
     $env:GOOGLE_API_KEY="your_api_key_here"
     ```
   
Alternatively, store the key in a `.env` file and load it in Python using `dotenv`:
```bash
pip install python-dotenv
```
Then create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```
And modify the script to load it:
```python
from dotenv import load_dotenv
load_dotenv()
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

## Usage
Run the project:
```bash
python3 rag.py
```

### Options in the Menu
1. **Process a new document** - Load a new PDF and store embeddings
2. **Ask a question (with RAG)** - Retrieve context and generate an answer
3. **Ask a question (LLM only)** - Use Gemini LLM without retrieval
4. **Compare RAG vs LLM only** - Show answers from both methods
5. **Exit** - Quit the application


