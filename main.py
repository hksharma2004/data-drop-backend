import datetime
import os
import os
import io
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.storage import Storage
from appwrite.query import Query

# Load env 
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")
APPWRITE_DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID")
APPWRITE_FILES_COLLECTION_ID = os.getenv("APPWRITE_FILES_COLLECTION_ID")
APPWRITE_BUCKET_ID = os.getenv("APPWRITE_BUCKET_ID")


client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(APPWRITE_PROJECT_ID)
client.set_key(APPWRITE_API_KEY)

databases = Databases(client)
storage = Storage(client)

# fast api initialize
app = FastAPI()

# configuring cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class ChatRequest(BaseModel):
    file_ids: List[str]
    question: str

# rag functionssss

def get_pdf_text_from_buffers(pdf_buffers: List[bytes]) -> str:
    """Extracts text from a list of PDF file buffers."""
    text = ""
    for buffer in pdf_buffers:
        pdf_reader = PdfReader(io.BytesIO(buffer))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: List[str]):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """Creates a conversational QA chain with a specific prompt using modern LangChain approach."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # initializing model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # prompt chaining here
    def qa_chain(docs, question):
        # formatting the context from docs
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Formatting the prompt
        formatted_prompt = prompt.format(context=context, question=question)
        
        # response from model
        response = model.invoke(formatted_prompt)
        
        # extracting content from response
        if hasattr(response, 'content'):
            return {"output_text": response.content}
        else:
            return {"output_text": str(response)}
    
    return qa_chain

# fastapi Endpoints 

@app.get("/ping")
async def ping():
   
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "status": "alive",
        "timestamp": now,
        "source": "appwrite-cron"
    }


@app.get("/list-pdfs")
async def list_pdfs(owner: str):

    try:
        # adding query to filter documents by owner
        response = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_FILES_COLLECTION_ID,
            queries=[
                Query.equal("owner", [owner]),
                Query.equal("type", ["document"]),
            ],
        )
        # filter all files for ".pdf" extension
        pdf_files = [
            {"id": doc["bucketFileId"], "name": doc["name"]}
            for doc in response["documents"]
            if doc["name"].lower().endswith(".pdf")
        ]
        return {"files": pdf_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDF list from Appwrite: {str(e)}")

@app.post("/chat-with-pdfs")
async def chat_with_pdfs(request: ChatRequest):
    """Handles the chat request by processing PDFs and answering the question."""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided.")
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided.")

    try:
        # 1. getting pdf files from appwrite storage
        pdf_buffers = []
        for file_id in request.file_ids:
            file_content = storage.get_file_download(
                bucket_id=APPWRITE_BUCKET_ID,
                file_id=file_id,
            )
            pdf_buffers.append(file_content)

        # 2.processing pdfs
        raw_text = get_pdf_text_from_buffers(pdf_buffers)
        if not raw_text.strip():
            return {"answer": "Could not extract any text from the selected PDF(s). Please check the documents."}
            
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)

        if not vector_store:
            return {"answer": "Failed to create a vector store from the PDF content. The document might be empty or unreadable."}


        docs = vector_store.similarity_search(request.question)
        

        if docs and isinstance(docs[0], str):
            docs = [Document(page_content=doc) for doc in docs]
        
        chain = get_conversational_chain()
        response = chain(docs, request.question)

        return {"answer": response["output_text"]}

    except Exception as e:

        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")