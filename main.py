import datetime as dt
import io
import os
from typing import Optional
from urllib.parse import urlparse

from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.storage import Storage
from appwrite.services.tables_db import TablesDB
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from PyPDF2 import PdfReader

# Load env
load_dotenv()

config = {
    "APPWRITE_ENDPOINT": os.getenv("APPWRITE_ENDPOINT"),
    "APPWRITE_PROJECT_ID": os.getenv("APPWRITE_PROJECT_ID"),
    "APPWRITE_API_KEY": os.getenv("APPWRITE_API_KEY"),
    "APPWRITE_DATABASE_ID": os.getenv("APPWRITE_DATABASE_ID"),
    "APPWRITE_FILES_COLLECTION_ID": os.getenv("APPWRITE_FILES_COLLECTION_ID"),
    "APPWRITE_BUCKET_ID": os.getenv("APPWRITE_BUCKET_ID"),
}

PROMPT_TEMPLATE = (
    "Answer the question as detailed as possible from the provided context. "
    "If the answer is not in the provided context, say "
    '"answer is not available in the context". Do not provide a wrong answer.\n\n'
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

# Appwrite
client = Client()
for key, setter in (
    ("APPWRITE_ENDPOINT", client.set_endpoint),
    ("APPWRITE_PROJECT_ID", client.set_project),
    ("APPWRITE_API_KEY", client.set_key),
):
    if config[key]:
        setter(config[key])

tables_db = TablesDB(client)
storage = Storage(client)

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    file_ids: list[str]
    question: str


def as_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.to_dict()


def bucket_file_id_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    path_parts = urlparse(url).path.split("/")
    try:
        files_index = path_parts.index("files")
        return path_parts[files_index + 1]
    except (ValueError, IndexError):
        return None


def normalized_file_data(doc):
    doc = as_dict(doc)
    return doc.get("data") or doc.get("_data") or doc


def pdf_metadata(doc):
    data = normalized_file_data(doc)
    bucket_file_id = data.get("bucketFileId") or bucket_file_id_from_url(data.get("url"))
    name = data.get("name") or ""
    extension = (data.get("extension") or name.rsplit(".", 1)[-1]).lower()
    if (
        not bucket_file_id
        or extension != "pdf"
        or data.get("type") != "document"
    ):
        return None
    return {"id": bucket_file_id, "name": name}


def pdf_rejection_reason(doc):
    data = normalized_file_data(doc)
    bucket_file_id = data.get("bucketFileId") or bucket_file_id_from_url(data.get("url"))
    name = data.get("name") or ""
    extension = (data.get("extension") or name.rsplit(".", 1)[-1]).lower()
    if not bucket_file_id:
        return "missing bucketFileId and no storage file id in url"
    if extension != "pdf":
        return f"extension is {extension or 'empty'}, not pdf"
    if data.get("type") != "document":
        return f"type is {data.get('type') or 'empty'}, not document"
    return "accepted"


def extract_pdf_text(file_ids: list[str]) -> str:
    pages = []
    for file_id in file_ids:
        file_content = storage.get_file_download(
            bucket_id=config["APPWRITE_BUCKET_ID"],
            file_id=file_id,
        )
        pdf_reader = PdfReader(io.BytesIO(file_content))
        pages.extend(page.extract_text() or "" for page in pdf_reader.pages)
    return "".join(pages)


def create_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    if not chunks:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.from_texts(chunks, embedding=embeddings)


def ask_gemini(docs, question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    response = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3).invoke(
        PROMPT_TEMPLATE.format(context=context, question=question)
    )
    return response.content if hasattr(response, "content") else str(response)


def answer_question_from_pdfs(file_ids: list[str], question: str) -> str:
    raw_text = extract_pdf_text(file_ids)
    if not raw_text.strip():
        return "Could not extract any text from the selected PDF(s). Please check the documents."

    vector_store = create_vector_store(raw_text)
    if not vector_store:
        return "Failed to create a vector store from the PDF content. The document might be empty or unreadable."

    return ask_gemini(vector_store.similarity_search(question), question)


# Endpoints
@app.get("/")
async def home():
    return {"status": "alive", "docs": "/docs", "health": "/ping"}


@app.get("/ping")
async def ping():
    return {
        "status": "alive",
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "source": "appwrite-cron",
    }


@app.get("/list-pdfs")
async def list_pdfs(owner: str, debug: bool = False):
    try:
        response = tables_db.list_rows(
            database_id=config["APPWRITE_DATABASE_ID"],
            table_id=config["APPWRITE_FILES_COLLECTION_ID"],
            queries=[
                Query.equal("owner", [owner]),
                Query.equal("type", ["document"]),
            ],
        )
        rows = as_dict(response).get("rows", [])
        files = [pdf for row in rows if (pdf := pdf_metadata(row))]
        payload = {
            "files": files,
            "owner": owner,
            "totalDocuments": len(rows),
            "matchedPdfs": len(files),
            "filterVersion": "pdf-filter-v3",
        }
        if debug:
            payload["debugRows"] = [
                {
                    "name": normalized_file_data(row).get("name"),
                    "type": normalized_file_data(row).get("type"),
                    "extension": normalized_file_data(row).get("extension"),
                    "hasBucketFileId": bool(normalized_file_data(row).get("bucketFileId")),
                    "hasUrlStorageId": bool(bucket_file_id_from_url(normalized_file_data(row).get("url"))),
                    "reason": pdf_rejection_reason(row),
                }
                for row in rows
            ]
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDF list from Appwrite: {str(e)}")


@app.post("/chat-with-pdfs")
async def chat_with_pdfs(request: ChatRequest):
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided.")
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided.")

    try:
        return {"answer": answer_question_from_pdfs(request.file_ids, request.question)}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")
