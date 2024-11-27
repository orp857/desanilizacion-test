# utils.py

import json
import os
import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone as PineconeVectorStore

# Load environment variables (if needed)
load_dotenv()

# Retrieve API keys and environment from st.secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Define the index name
index_name = 'desalinizacion'

# Check if the index exists
if index_name not in pinecone.list_indexes():
    raise ValueError(f"Index '{index_name}' does not exist. Please create it first.")

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vector_store = PineconeVectorStore(index, embeddings.embed_query, "text")

# Create retriever
retriever = vector_store.as_retriever()

# Define your custom prompt template
INITIAL_TEMPLATE = """
Eres un experto en regulación de la industria del mercado eléctrico y la industria de la desalinización en las regiones de España, Australia, Israel, Arabia Saudita y California, creado por un ingeniero eléctrico.
Para generar tus respuestas y propuestas, debes seguir y guiarte por los principios regulatorios definidos por Colbún los cuales son: eficiencia económica; no arriesgar la seguridad de abastecimiento del sistema; se debe evitar la utilización de metodologías complejos y cambios abruptos; debe seguir un tratamiento no discriminatorio entre los distintos agentes del mercado; asegurar transparencia; las reglas del mercado deben garantizar actividades productivas rentables; se deben enviar las señales económicas correctas para fomentar la eficiencia en el suministro.
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
- De tu base de datos de documentos debes analizarlos y comprender cada documento para proporcionar las respuestas más precisas a las preguntas de los usuarios, identificando instituciones, régimen legal, roles del Estado, barreras regulatorias, régimen de propiedad, obligaciones de suministro, permisos y competencias territoriales, modelos de participación privada, estructuras de financiamiento y condiciones, gestión de proyectos, y contexto histórico y geográfico.
- Identificar términos clave y definiciones.
- Usar lenguaje técnico adecuado.
- Desarrollar resumen técnico incluyendo estándares, regulaciones y normativas mencionadas.
- Enfocarse en puntos clave, evitando opiniones no verificadas.
En caso que el usuario te pida, a partir de las preguntas realizadas por el usuario y tus respuestas, genera un texto compilado en formato informe con todas las respuestas de la conversación.
QUESTION: {question}
=========
{context}
=========
"""

PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["context", "question"])

# Define the LLM
LLM = OpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2048)

# Set up the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    verbose=True
)

# Function to get the answer
def get_answer(query):
    answer = qa_chain.run(query)
    return answer

# Function to refine the query given the conversation history
def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica centrada en el área de regulación. Esta pregunta refinada debe ayudarte a obtener la información más relevante de la base de conocimientos para responder de la mejor manera posible. La consulta refinada debe estar en forma de pregunta y no exceder de 2 oraciones.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.3,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    refined_query = response['choices'][0]['text'].strip()
    return refined_query

# Function to get the conversation string
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

