# Data Drop Backend

A FastAPI-powered backend service that enables intelligent document interaction through Retrieval Augmented Generation (RAG). Users can upload PDF documents and engage in conversational queries about their content using Google's Generative AI.

##  Features

- **PDF Document Processing**: Extract and process text from PDF files
- **Intelligent Document Chat**: Ask questions about your PDF documents using natural language
- **Vector Search**: Powered by FAISS vector store for efficient document similarity search
- **Cloud Storage Integration**: Seamless integration with Appwrite for document storage
- **CORS Support**: Cross-origin resource sharing enabled for web applications
- **Health Check**: Built-in ping endpoint for service monitoring

## Technology Stack

- **Web Framework**: FastAPI
- **AI/ML**: Google Generative AI (Gemini 1.5 Flash), LangChain, FAISS
- **PDF Processing**: PyPDF2
- **Database & Storage**: Appwrite
- **Environment Management**: python-dotenv

## Prerequisites

- Python 3.7+
- Google Generative AI API key
- Appwrite instance with configured database and storage

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/hksharma2004/data-drop-backend.git
cd data-drop-backend
```

2. **Install dependencies**
```bash
pip install fastapi uvicorn python-dotenv PyPDF2 langchain langchain-google-genai langchain-community google-generativeai appwrite faiss-cpu pydantic
```

3. **Environment Configuration**
Create a `.env` file in the root directory:

```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_generative_ai_api_key

# Appwrite Configuration
APPWRITE_ENDPOINT=your_appwrite_endpoint
APPWRITE_PROJECT_ID=your_project_id
APPWRITE_API_KEY=your_api_key
APPWRITE_DATABASE_ID=your_database_id
APPWRITE_FILES_COLLECTION_ID=your_files_collection_id
APPWRITE_BUCKET_ID=your_storage_bucket_id
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **Endpoint**: `GET /ping`
- **Description**: Check if the service is running
- **Response**: Service status and timestamp

### List PDF Documents
- **Endpoint**: `GET /list-pdfs`
- **Parameters**: `accountId` (query parameter)
- **Description**: Retrieve all PDF documents associated with an account
- **Response**: List of PDF files with ID and name

### Chat with PDFs
- **Endpoint**: `POST /chat-with-pdfs`
- **Request Body**:
```json
{
  "file_ids": ["file_id_1", "file_id_2"],
  "question": "Your question about the documents"
}
```
- **Description**: Process PDF documents and answer questions using RAG
- **Response**: AI-generated answer based on document content

## How It Works

1. **Document Processing**: PDF files are retrieved from Appwrite storage and text is extracted
2. **Text Chunking**: Extracted text is split into manageable chunks (10,000 characters with 1,000 overlap)
3. **Vector Embedding**: Text chunks are converted to embeddings using Google's embedding model
4. **Similarity Search**: User questions are matched against document chunks using FAISS
5. **Response Generation**: Relevant chunks provide context for Gemini model to generate accurate answers

##  Configuration

- **Text Chunking**: 10,000 character chunks with 1,000 character overlap
- **AI Model**: Gemini 1.5 Flash with 0.3 temperature
- **Embedding Model**: Google's embedding-001 model

## Security Features

- Environment variable configuration for sensitive data
- API key authentication for external services
- CORS middleware for controlled cross-origin access
