from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import uuid

app = FastAPI()

# Model for URL and Chat input
class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    chat_id: str
    question: str

# Initialize sentence-transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# In-memory storage for simplicity
data_store: Dict[str, Dict[str, str]] = {}

# FAISS index for vector search
dimension = 384  # MiniLM embedding size
index = faiss.IndexFlatL2(dimension)

# Function to clean text
def clean_text(text: str) -> str:
    return ' '.join(text.split())

# Function to extract text from URL
def extract_content_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return clean_text(soup.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error scraping the URL.")

# Function to extract text from PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return clean_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the PDF.")

# API to process URL
@app.post("/process_url")
def process_url(request: URLRequest):
    chat_id = str(uuid.uuid4())
    content = extract_content_from_url(request.url)
    # Store content and generate embeddings
    embeddings = model.encode([content])
    index.add(embeddings)
    data_store[chat_id] = {"content": content}
    return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}

# API to process PDF
@app.post("/process_pdf")
def process_pdf(file: UploadFile = File(...)):
    chat_id = str(uuid.uuid4())
    content = extract_text_from_pdf(file)
    # Store content and generate embeddings
    embeddings = model.encode([content])
    index.add(embeddings)
    data_store[chat_id] = {"content": content}
    return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}

# API to handle chat
@app.post("/chat")
def chat(request: ChatRequest):
    if request.chat_id not in data_store:
        raise HTTPException(status_code=404, detail="Chat ID not found.")

    content = data_store[request.chat_id]["content"]
    question_embedding = model.encode([request.question])

    # Use FAISS to find the most similar content
    _, indices = index.search(question_embedding, k=1)
    relevant_text = content

    return {"response": relevant_text}
