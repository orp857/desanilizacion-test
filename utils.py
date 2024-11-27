import json
import os
import openai
import pinecone
from langchain.chains import StuffDocumentsChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")
INDEX_PINECONE = pinecone.Index("desanilizacion")

# Embedding model
EMBEDDINGS = OpenAIEmbeddings()

# Helper class for documents
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

# Helper functions
def transform_dict_to_document(dict_list):
    document_list = []
    for dict_obj in dict_list:
        page_content = dict_obj["metadata"]["text"].replace("\n", "")
        metadata = {
            "page": dict_obj["metadata"]["page"],
            "source": dict_obj["metadata"]["source"],
        }
        doc = Document(page_content=page_content, metadata=metadata)
        document_list.append(doc)
    return document_list

def get_similar_docs_pinecone(query, k=10):
    query_embedding = EMBEDDINGS.embed_query(query)
    result_query = INDEX_PINECONE.query(vector=query_embedding, top_k=k, include_metadata=True)
    result_query_json = json.dumps(result_query.to_dict())

    def json_to_list(json_string):
        try:
            json_dict = json.loads(json_string)
            return json_dict["matches"]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise

    similar_docs = transform_dict_to_document(json_to_list(result_query_json))
    return similar_docs

def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Refine the user's query based on the conversation history."},
            {"role": "user", "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}"},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response["choices"][0]["message"]["content"]

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Bot: {st.session_state['responses'][i+1]}\n"
    return conversation_string

def get_answer(query):
    similar_docs = get_similar_docs_pinecone(query)
    prompt_template = """
    Use the following documents to answer the question:
    =========
    {summaries}
    =========
    Question: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["summaries", "question"])
    llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa_chain = StuffDocumentsChain(llm_chain=llm, prompt=PROMPT)
    return qa_chain.invoke({"input_documents": similar_docs, "question": query})

