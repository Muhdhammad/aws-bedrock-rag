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

from langchain_huggingface import HuggingFaceEmbeddings
embeddings_hf = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)

# OpenAi embeddings


# langsmith tracing
# os.environ["LANGSMITH_TRACING"] = "true"

# pdf loader
from langchain_community.document_loaders import PyPDFLoader

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# vector store
from langchain_community.vectorstores import FAISS

def get_uuid():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_split.split_documents(pages)
    return docs

def create_vector_store(request_id, docs, save_path=None):

    faiss_vectorstore = FAISS.from_documents(docs, embeddings_hf)

    file_name = "f{request_id}.bin"
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        faiss_vectorstore.save_local(folder_path=save_path, index_name=file_name)

        # upload to S3
        # For faiss index file
        s3_client.upload_file(
            Filename=save_path + "/" + file_name + ".faiss",
            Bucket=BUCKET_NAME,
            Key="my_faiss.faiss"
        )
        
        # For faiss pkl metadata file
        s3_client.upload_file(
            Filename=save_path + "/" + file_name + ".pkl",
            Bucket=BUCKET_NAME,
            Key="my_faiss.pkl"
        )

        return True


def main():
    st.write("Hello")
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
        st.write(f"first split {splitted[0]}")
        st.write("=======================")
        st.write(f"Second split {splitted[1]}")

        st.write("Creating vector store")

        folder_path = "/temp/"
        r = create_vector_store(request_id, splitted, folder_path)

        if r:
            st.write("Success")
        else:
            st.write("Error, check logs")

if __name__ == "__main__":
    main()