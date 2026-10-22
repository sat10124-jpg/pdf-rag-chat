
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# This is our Fast API
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class QuestionRequest(BaseModel):
    question: str
    collection_name: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = "uploads/" + file.filename
    #Read the pdf file as binary data
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # We call the functions to break down the pdfs
    chunks = load_and_chunk_pdf(file_path)
    embeddings = embed_chunks(chunks)
    # Save it as this colleciton name
    collect_name = "uploaded_pdf"
    # Store it in chromda
    store_chunks(chunks,embeddings,collect_name)
    
    return {"filename": file.filename, 
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "collection_name": collect_name
            }  
    
@app.post("/ask")
def ask(request: QuestionRequest):
    answer = answer_question(request.question,request.collection_name)
    return {"the_answer": answer}
    
@app.get("/")
def homepage():
    return FileResponse("static/index.html")
    
    
    
    




import os
import fitz
import chromadb
from chromadb.utils import embedding_functions

# chromadb's built-in embedding function
# uses the same underlying model but without needing PyTorch separately
embedding_model = embedding_functions.DefaultEmbeddingFunction()


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500) -> list[str]:
    doc = fitz.open(pdf_path)
    full_text = ""
    try:
        for page in doc:
            full_text += page.get_text()
    finally:
        doc.close()

    chunks = []
    start = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end])
        start = end

    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embeds a list of chunks using ChromaDB's built-in embedding function.
    Returns a list of embeddings.
    """
    embeddings = embedding_model(chunks)
    return embeddings


# "chrome_db" is the folder that will store everything
chroma_client = chromadb.PersistentClient(path="chroma-db")


def store_chunks(chunks: list[str], embeddings: list[list[float]], collection_name: str):
    # store chunks + embeddings in ChromaDB
    # collection_name separates different PDFs from each other
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    collection.add(
        documents=chunks,  # the actual text
        embeddings=embeddings,  # the vectors
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
    return collection


def retrieve_relevant_chunks(question: str, collection_name: str, n_results: int = 3):
    # Given a question, find the n most relevant chunks from the collection.
    question_embedding = embedding_model([question])
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
    )

    return results["documents"][0]


    
import ollama

def answer_question(question: str, collection_name: str) -> str:
    """
    Full RAG pipeline in one function:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a prompt with those chunks as context
    3. Send to Ollama and return the answer
    """
    # step 1: retrieve
    relevant_chunks = retrieve_relevant_chunks(question, collection_name)
    
    # step 2: build context string from chunks
    context = "\n\n".join(relevant_chunks)
    
    # step 3: build the prompt
    # system prompt
    prompt = f"""You are a helpful assistant. Answer the question using 
ONLY the context provided below. If the answer isn't in the context, 
say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""

    # step 4: send to ollama
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]
    

if __name__ == "__main__":
    test_path = "test.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50),
        "Dynamic arrays resize when they reach capacity. "
        "Binary search runs in O(log n) time. "
        "Python lists are implemented as dynamic arrays. "
        "Hash tables store key value pairs for O(1) lookup. "
        "Linked lists have O(n) access time but O(1) insertion. " * 20
    )
    doc.save(test_path)
    doc.close()

    chunks = load_and_chunk_pdf(test_path)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, collection_name="test_collection")

    print("\n--- Testing full RAG pipeline ---")
    question = "how do arrays grow in size?"
    print(f"Question: {question}")
    print(f"Answer: {answer_question(question, 'test_collection')}")

    # test that it says "I don't know" for out of scope questions
    question2 = "what is the capital of France?"
    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer_question(question2, 'test_collection')}")

    os.remove(test_path)