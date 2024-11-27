# Import necessary libraries and modules
import json
import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import pinecone  # Use pinecone-client==2.2.1

# Load environment variables and secrets
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = "us-east-1"
MODEL_NAME = "text-embedding-ada-002"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Connect to the existing Pinecone index or create it if it doesn't exist
index_name = 'desalinizacion'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric='cosine')
# Connect to the index
INDEX_PINECONE = pinecone.Index(index_name)

# Initialize the OpenAI embeddings
EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Class to represent a document
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

# Function to transform a list of dictionaries into Document objects
def transform_dict_to_document(dict_list):
    document_list = []
    for dict_obj in dict_list:
        # Extract page content and metadata from the dictionary
        page_content = dict_obj['metadata'].get('text', '').replace('\n', '')
        metadata = {
            'page': dict_obj['metadata'].get('page', ''),
            'source': dict_obj['metadata'].get('source', '')
        }
        # Create a Document object
        doc = Document(page_content=page_content, metadata=metadata)
        # Add the Document to the list
        document_list.append(doc)
    return document_list

# Function to retrieve similar documents from Pinecone based on the query
def get_similar_docs_pinecone(query, k=10):
    # Generate the query embedding
    query_embedding = EMBEDDINGS.embed_query(query)
    # Query Pinecone for similar documents
    result_query = INDEX_PINECONE.query(
        vector=query_embedding, 
        top_k=k, 
        include_metadata=True
    )
    # Transform the result into a list of Document objects
    similar_docs = transform_dict_to_document(result_query['matches'])
    return similar_docs

# Function to refine the user's query based on the conversation history
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica centrada en el área de regulación."},
            {"role": "user", "content": f"CONVERSATION LOG:\n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.3,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

# Function to get the conversation history as a string
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Assistant: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# Function to get an answer to the user's query
def get_answer(query):
    similar_docs = get_similar_docs_pinecone(query)
    # Use the RetrievalQA chain with the OpenAI LLM and Pinecone retriever
    qa_chain = RetrievalQA(
        llm=OpenAI(temperature=0.3, model_name="gpt-4"), 
        retriever=Pinecone(index=INDEX_PINECONE, embedding_function=EMBEDDINGS)
    )
    answer = qa_chain.run({"input_documents": similar_docs, "question": query})
    return answer

# Initial prompt template for the assistant
INITIAL_TEMPLATE = """
Eres un experto en regulación de la industria del mercado eléctrico y la industria de la desalinización en las regiones de España, Australia, Israel, Arabia Saudita y California, creado por un ingeniero eléctrico.
Para generar tus respuestas y propuestas, debes seguir y guiarte por los principios regulatorios definidos por Colbún los cuales son: eficiencia económica; no arriesgar la seguridad de abastecimiento del sistema; se debe evitar la utilización de metodologías complejas y cambios abruptos; debe seguir un tratamiento no discriminatorio entre los distintos agentes del mercado; asegurar transparencia; las reglas del mercado deben garantizar actividades productivas rentables; se deben enviar las señales económicas correctas para fomentar la eficiencia en el suministro.
Responsabilidades Específicas que tienes que seguir:
- Recepción y análisis de documentos regulatorios.
- Identificación y extracción de información relevante.
- Generación de informes personalizados.
- Responder preguntas específicas sobre regulaciones.
- Proporcionar explicaciones claras y precisas.
- Conservar el aprendizaje de respuestas previas y sugerir correlaciones con otros documentos de la base de datos.
Reglas de Respuesta:
- Nunca repetir la pregunta en la respuesta.
- Responder con amabilidad y precisión.
Analiza y comprende cada documento de tu base de datos para proporcionar las respuestas más precisas, identificando instituciones, régimen legal, roles del Estado, barreras regulatorias, régimen de propiedad, obligaciones de suministro, permisos y competencias territoriales, modelos de participación privada, estructuras de financiamiento y condiciones, gestión de proyectos y contexto histórico y geográfico.
Identifica términos clave y definiciones.
Usa lenguaje técnico adecuado.
Desarrolla resúmenes técnicos incluyendo estándares, regulaciones y normativas mencionadas.
Enfócate en puntos clave, evitando opiniones no verificadas.
En caso que el usuario te lo pida, a partir de las preguntas realizadas y tus respuestas, genera un texto compilado en formato de informe con todas las respuestas de la conversación.
QUESTION: {question}
=========
{summaries}
========="""

PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["summaries", "question"])

