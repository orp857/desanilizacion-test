import pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import streamlit as st
import os
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")
INDEX_PINECONE = pinecone.Index("desanilizacion")

# Initialize embeddings
EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define the language model
LLM = ChatOpenAI(
    temperature=0.3,
    model="gpt-4",
    openai_api_key=OPENAI_API_KEY,
    max_tokens=2048
)

# Define prompt template
PROMPT = PromptTemplate(
    template="""
    Eres un experto en regulación de la industria...
    QUESTION: {question}
    =========
    {summaries}
    =========
    """,
    input_variables=["summaries", "question"]
)

# Define RetrievalQA
QA = RetrievalQA(
    retriever=INDEX_PINECONE.as_retriever(),
    llm=LLM,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)

# Function to refine query
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Refine user queries based on conversation history."},
            {"role": "user", "content": f"CONVERSATION LOG:\n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.3,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

# Function to get similar docs from Pinecone
def get_similar_docs_pinecone(query, k=10):
    query_embedding = EMBEDDINGS.embed_query(query)
    result_query = INDEX_PINECONE.query(vector=query_embedding, top_k=k, include_metadata=True)
    matches = result_query.get("matches", [])
    return transform_dict_to_document(matches)

# Transform function and other components remain the same.
