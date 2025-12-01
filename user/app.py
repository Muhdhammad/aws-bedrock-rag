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
print(BUCKET_NAME)

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME not found")

res = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="vector-store/")
if 'Contents' in res:
    print("objects in buck")
    for obj in res["contents"]:
        print(f"{obj['Key']}")
else:
    print("bucket empty?")

# List available bedrock models for region
def list_available_bedrock_models():
    bedrock = boto3.client('bedrock', region_name='eu-north-1')
    
    try:
        response = bedrock.list_foundation_models()
        print("Available models in eu-north-1:")
        for model in response['modelSummaries']:
            print(f"- {model['modelId']}")
    except Exception as e:
        print(f"Error: {e}")

list_available_bedrock_models()

# Huggingface embeddings
# Depreciated: from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# bedrock embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION"))

def get_embeddings():

    # Huggingface embeddings
    embeddings_hf = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )
        
    '''
    # bedrock titan embeddings
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock_client
    )
    '''

    return embeddings_hf

# Groq 
from langchain_groq import ChatGroq
 

def initialize_llm():

    """
    llm = Bedrock(
        credentials_profile_name="default",
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 512}
    )

    """
    # initializing Groq llm
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    return llm

# Prompt and Chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_response(llm, vector_store, question):

    prompt_template = """
        human: You are a helpful assistant. Use ONLY the context provided below to answer the question.
        If the answer is not in the context, say: "I don't know."

        <context>
        {context}
        </context>

        Question: {question}

        Assistant:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4}   # 4 most relevant chunks
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    answer = qa({"query": question})
    return answer['result']

# vector store
from langchain_community.vectorstores import FAISS

def get_uuid():
    return str(uuid.uuid4())

def create_faiss_index(save_path):
    faiss_index = FAISS.load_local(
        folder_path=save_path,
        # index_name="my_faiss",
        embeddings=get_embeddings(),
        # allow_dangerous_deserialization=True
    )

    return faiss_index

def load_faiss_index(save_path):

    os.makedirs(save_path, exist_ok=True)

    s3_client.download_file(
        Filename=os.path.join(save_path, "index.faiss"),
        Bucket=BUCKET_NAME,
        Key="vector-stores/index.faiss"
    )
    
    s3_client.download_file(
        Filename=os.path.join(save_path, "index.pkl"),
        Bucket=BUCKET_NAME,
        Key="vector-stores/index.pkl"
    )


def main():
    st.header("Hello")

    folder_path = os.path.join(os.getcwd(), "temp")
    load_faiss_index(save_path=folder_path)

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)


    faiss_index = create_faiss_index(save_path=folder_path)
    if faiss_index:
        st.write("Index is ready")

    question = st.text_input("Ask your question")

    if st.button("Ask Question"):
        with st.spinner("Analyzing.."):

            llm = initialize_llm()

            # get the response
            result = get_response(llm, faiss_index, question)
            st.write(result)
            st.success("Done")

if __name__ == "__main__":
    main()