# Retrieval-Augmented Generation (RAG) with LangChain

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain** to process PDFs, store document embeddings in a vector database (ChromaDB), and generate responses using Google's Gemini AI model.

### Features:
- **Load and Process PDF Documents**: Extracts text and splits it into chunks.
- **Vector Storage with ChromaDB**: Stores document embeddings for efficient retrieval.
- **Retrieval-Augmented Generation (RAG)**: Enhances responses with contextually relevant document chunks.
- **LLM-Only Response Generation**: Generates responses without document context for comparison.
- **Comparison of RAG vs. LLM-Only Responses**.
- **User-Friendly CLI Interface**: Menu-driven interaction for processing and querying.

---

## Installation Guide
### 1. Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install langchain langchain_community langchain_google_genai torch sentence-transformers
```

### 3. Run the Project
```bash
python3 rag.py
```

---

## Usage Guide
Once the program starts, it presents a menu with the following options:

1. **Process a New Document**: Load and process a new PDF file.
2. **Ask a Question (RAG)**: Retrieve context from stored documents and generate an answer.
3. **Ask a Question (LLM-Only)**: Generate an answer without retrieval.
4. **Compare RAG vs. LLM-Only Responses**.
5. **Exit**.

### Example Usage:
#### Processing a PDF
- Place your PDF file in the project directory.
- Select **Option 1** and enter the PDF filename (e.g., `document.pdf`).
- The file is processed, chunked, and stored in the vector database.

#### Asking a Question with RAG
- Select **Option 2** and enter a query.
- The system retrieves relevant document chunks and generates an informed answer.

#### Asking a Question with LLM-Only
- Select **Option 3** and enter a query.
- The system generates a response **without document retrieval**.

#### Comparing RAG vs. LLM-Only
- Select **Option 4**, enter a query, and see how both approaches differ.

---

## Project Structure
```
.
├── db/                   # ChromaDB persistent storage
├── outputs/              # Saved outputs for comparisons
├── document.pdf          # Sample PDF document
├── rag.py                # Main Python script
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
```

---

## Technologies Used
- **Python**: Programming language.
- **LangChain**: Framework for LLM applications.
- **ChromaDB**: Vector database for document storage.
- **Google Gemini AI**: Language model for generating responses.
- **Hugging Face Embeddings**: Text embeddings for document chunking.
- **PyPDF**: Extracts text from PDF files.

---

## Future Improvements
- Implement a **web-based UI** for easier interaction.
- Support **multiple document formats** (e.g., DOCX, TXT).
- Enable **fine-tuning of retrieval parameters**.
- Add **multi-language support** for diverse document processing.

---




