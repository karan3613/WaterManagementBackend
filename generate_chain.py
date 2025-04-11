import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
import os 


load_dotenv()
def generate_chain():
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_api_key)
    model = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001" , 
        google_api_key = gemini_api_key
        )
    vector_embeddings = FAISS.load_local(
        "vector_embedding",
        embeddings= embeddings ,
        allow_dangerous_deserialization=True
        )
    chain = RetrievalQA.from_chain_type(
        llm = model , 
        chain_type = "stuff" , 
        retriever = vector_embeddings.as_retriever())
    return chain 
