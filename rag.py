import os
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Configuration
PDF_PATH = "document.pdf"
DB_DIRECTORY = "db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "DocumentCollection"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs(DB_DIRECTORY, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def load_pdf_and_split(pdf_path):
    """
    Load PDF and split into documents.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Document]: Extracted and split documents
    """
    print(f"Processing PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    documents = text_splitter.split_text(text)
    
    docs = [Document(page_content=doc) for doc in documents]
    
    print(f"Created {len(docs)} document chunks")
    return docs

def create_vector_store(documents):
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): List of documents to embed
        
    Returns:
        Chroma: Initialized vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIRECTORY
    )
    
    return vector_store

def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation chain.
    
    Args:
        vector_store (Chroma): Vector store to use for retrieval
        
    Returns:
        RetrievalQA: RAG chain
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    
    prompt_template = """
    You are a helpful AI assistant designed to answer questions based on the context provided.
    If the question cannot be answered from the context, please state that clearly.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 5}
        ),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return retrieval_qa

def create_llm_only_response(query):
    """
    Generate a response using LLM without context.
    
    Args:
        query (str): User's question
        
    Returns:
        str: LLM response
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    
    # Generate response
    response = llm.invoke(query)
    return response.content

def retrieve_relevant_chunks(vector_store, query, k=5):
    """
    Retrieve relevant chunks for a given query.
    
    Args:
        vector_store (Chroma): Vector store to search
        query (str): User's question
        k (int): Number of chunks to retrieve
        
    Returns:
        List[str]: Retrieved document chunks
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in relevant_docs]

def save_output(filename, content):
    """
    Save output to a file.
    
    Args:
        filename (str): Output filename
        content (str): Content to save
    """
    with open(f"outputs/{filename}.txt", "w", encoding='utf-8') as f:
        f.write(content)

def main():
    """Run the complete RAG pipeline using LangChain."""
    ensure_directories_exist()
    documents = load_pdf_and_split(PDF_PATH)
    vector_store = create_vector_store(documents)
    rag_chain = create_rag_chain(vector_store)
    while True:
        print("\n" + "="*50)
        print("LangChain RAG (Retrieval-Augmented Generation) Demo")
        print("="*50)
        print("1. Process a new document")
        print("2. Ask a question (with RAG)")
        print("3. Ask a question (LLM only)")
        print("4. Compare RAG vs. LLM only")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            pdf_path = input("Enter the path to the PDF document: ")
            documents = load_pdf_and_split(pdf_path)
            vector_store = create_vector_store(documents)
            rag_chain = create_rag_chain(vector_store)
            
        elif choice == "2":
            query = input("\nEnter your question: ")
            
            result = rag_chain.invoke({"query": query})
            
            print("\n----- Answer (with RAG) -----")
            print(result['result'])
            
            save_option = input("\nSave this output? (y/n): ")
            if save_option.lower() == "y":
                filename = input("Enter filename: ")
                chunks = retrieve_relevant_chunks(vector_store, query)
                
                output_text = f"Query: {query}\n\n"
                output_text += "Retrieved Chunks:\n"
                for i, chunk in enumerate(chunks, 1):
                    output_text += f"\nChunk {i}:\n{chunk}\n"
                output_text += f"\nAnswer:\n{result['result']}"
                
                save_output(filename + "_with_rag", output_text)
                print(f"Output saved to outputs/{filename}_with_rag.txt")
            
        elif choice == "3":
            query = input("\nEnter your question: ")
            document_text = " ".join([doc.page_content for doc in documents])
            answer = create_llm_only_response(query + document_text)
            print("\n----- Answer (LLM only) -----")
            print(answer)
    
            
            save_option = input("\nSave this output? (y/n): ")
            if save_option.lower() == "y":
                filename = input("Enter filename: ")
                output_text = f"Query: {query}\n\nAnswer (LLM only):\n{answer}"
                save_output(filename + "_llm_only", output_text)
                print(f"Output saved to outputs/{filename}_llm_only.txt")
            
        elif choice == "4":
            query = input("\nEnter your question: ")
            
            rag_result = rag_chain.invoke({"query": query})
            document_text = " ".join([doc.page_content for doc in documents])
            llm_answer = create_llm_only_response(query+document_text)
            chunks = retrieve_relevant_chunks(vector_store, query)
            
            print("\n----- Answer (with RAG) -----")
            print(rag_result['result'])
            
            print("\n----- Answer (LLM only) -----")
            print(llm_answer)
            
            save_option = input("\nSave this comparison? (y/n): ")
            if save_option.lower() == "y":
                filename = input("Enter filename: ")
                
                output_text = f"Query: {query}\n\n"
                output_text += "Retrieved Chunks:\n"
                for i, chunk in enumerate(chunks, 1):
                    output_text += f"\nChunk {i}:\n{chunk}\n"
                output_text += f"\nAnswer (with RAG):\n{rag_result['result']}\n\n"
                output_text += f"Answer (LLM only):\n{llm_answer}"
                
                save_output(filename + "_comparison", output_text)
                print(f"Output saved to outputs/{filename}_comparison.txt")
            
        elif choice == "5":
            print("Exiting. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()