import os
import boto3
import streamlit as st
import uuid
from dotenv import load_dotenv

load_dotenv()

# s3 client container
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION")
    )

# bucket
BUCKET_NAME = os.environ.get("BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME not found")

# Huggingface embeddings
# Depreciated: from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_hf = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)

# OpenAi embeddings


# langsmith tracing
# os.environ["LANGSMITH_TRACING"] = "true"

# pdf loader
from langchain_community.document_loaders.pdf import PyPDFLoader

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# vector store
from langchain_community.vectorstores import FAISS

def get_uuid():
    return str(uuid.uuid4())

import glob
from langchain_community.document_loaders.pdf import PyPDFLoader

def load_pdfs_from_folder(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found in {folder_path}")
        return []

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        # Skip empty pages
        pages = [p for p in pages if p.page_content.strip()]
        all_docs.extend(pages)

    st.write(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")
    return all_docs


def split_text(pages, chunk_size, chunk_overlap):
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_split.split_documents(pages)
    return docs

def create_vector_store(request_id, docs, save_path=None):

    faiss_vectorstore = FAISS.from_documents(docs, embeddings_hf)

    file_name = "f{request_id}"
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        faiss_vectorstore.save_local(folder_path=save_path, index_name=file_name)

        # upload to S3
        # For faiss index file
        s3_client.upload_file(
            Filename=os.path.join(save_path, f"{file_name}.faiss"),
            Bucket=BUCKET_NAME,
            Key=f"vector-stores/index.faiss"
        )
        
        # For faiss pkl metadata file
        s3_client.upload_file(
            Filename=os.path.join(save_path, f"{file_name}.pkl"),
            Bucket=BUCKET_NAME,
            Key=f"vector-stores/index.pkl"
        )

        return True


def main():
    st.write("PaperForge — RAG Assitant ")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    
    if uploaded_file is not None:
        request_id = get_uuid()
        st.write(f"Request id: {request_id}")

        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, "wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total pages: {len(pages)}")

        # Split text
        splitted = split_text(pages, 1200, 300)
        st.write(f"Total splitted docs {len(splitted)}")

        # Logging
        st.write(f"first split {splitted[0]}")
        st.write("=======================")
        st.write(f"Second split {splitted[1]}")

        st.write("Creating vector store")

        folder_path = os.path.join(os.getcwd(), "temp")
        result = create_vector_store(request_id, splitted, folder_path)

        if result:
            st.write("Success")
        else:
            st.write("Error, check logs")

    # NEW: Batch load all PDFs from folder
    """
    if st.button("Load all research papers"):
        request_id = get_uuid()
        folder_path = os.path.join(os.getcwd(), "temp")

        # Load all PDFs from folder
        docs = load_pdfs_from_folder("./researchpapers")


        # Split text 
        with st.spinner("Splitting PDFs into chunks..."):
            splitted_docs = split_text(docs, chunk_size=1200, chunk_overlap=300)
        st.write(f"Total splitted docs: {len(splitted_docs)}")

        # Create vector store
        result = create_vector_store(request_id, splitted_docs, folder_path)

        if result:
            st.success("Vector store created successfully for all research papers!")
        else:
            st.error("Error creating vector store")
    """

if __name__ == "__main__":
    main()