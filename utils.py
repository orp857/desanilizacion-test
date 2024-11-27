# Importaciones de servicios y bibliotecas
import json
import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Configuración y constantes globales
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = "us-east-1"  # specify the environment
MODEL_NAME = "text-embedding-ada-002"

# Inicializa Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Check if the index exists and handle the exception if it already exists
index_name = 'desalinizacion'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )

INDEX_PINECONE = pc.get_index(index_name)
EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Clase para representar un documento
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

# Función para transformar un diccionario en un Document
def transform_dict_to_document(dict_list):
    document_list = []

    for dict_obj in dict_list:
        # Extraer el contenido de la página y los metadatos del diccionario
        page_content = dict_obj['metadata']['text']
        page_content = page_content.replace('\n', '')  # Elimina los saltos de línea

        metadata = {'page': dict_obj['metadata']['page'], 'source': dict_obj['metadata']['source']}

        # Crear un Document con el contenido de la página y los metadatos
        doc = Document(page_content=page_content, metadata=metadata)

        # Añadir el Document a la lista
        document_list.append(doc)

    # Devolver la lista de Documents
    return document_list


def get_similar_docs_pinecone(query, k=10, score=False):
    query_embedding = EMBEDDINGS.embed_query(query)
    result_query = INDEX_PINECONE.query(vector=query_embedding, top_k=k, include_metadata=True)
    result_query_json = json.dumps(result_query.to_dict())

    def json_to_list(json_string):
        try:
            json_dict = json.loads(json_string)
            return json_dict['matches']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"JSON string: {json_string[:500]}...")  # Print first 500 chars for inspection
            raise

    similar_docs = transform_dict_to_document(json_to_list(result_query_json))
    return similar_docs

# Función para refinar una consulta dada una conversación anterior
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica centrada en el área de regulación."},
            {"role": "user", "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.3,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

# Función para obtener el historial de conversación
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# Función para obtener una respuesta a una consulta
def get_answer(query):
    similar_docs = get_similar_docs_pinecone(query)
    qa_chain = RetrievalQA(llm=OpenAI(temperature=0.3, model_name="gpt-4"), retriever=Pinecone(index=INDEX_PINECONE, embedding_function=EMBEDDINGS))
    answer = qa_chain.run({"input_documents": similar_docs, "question": query})
    return answer

# Plantilla de aviso inicial
INITIAL_TEMPLATE = """
Eres un experto en regulación de la industria del mercado eléctrico y la industria de la desalinización en las regiones de España, Australia, Israel, Arabia Saudita y California, creado por un ingeniero eléctrico.
Para generar tus respuestas y propuestas, debes seguir y guiarte por los principios regulatorios definidos por Colbún los cuales son: eficiencia económica; no arriesgar la seguridad de abastecimiento del sistema; se debe evitar la utilización de metodologías complejos y cambios abruptos; debe seguir un tratamiento no discriminatorio entre los distintos agentes del mercado; asegurar transparencia; las reglas del mercado deben garantizar actividades productivas rentables; se deben enviar las señales económicas correctas para fomentar la eficiencia en el suministro.
Responsabilidades Específicas que tienes que seguir
Recepción y análisis de documentos regulatorios.
Identificación y extracción de información relevante.
Generación de informes personalizados.
Responder preguntas específicas sobre regulaciones.
Proporcionar explicaciones claras y precisas.
Conservar el aprendizaje de respuestas previas y sugerir correlaciones con otros documentos de la base de datos.
Reglas de Respuesta
Nunca repetir la pregunta en la respuesta.
Responder con amabilidad y precisión.
De tu base de datos de documentos debes analizarlos y comprender cada documento para proporcionar las respuestas más precisas a las preguntas de los usuarios, identificando instituciones, régimen legal, roles del Estado, barreras regulatorias, régimen de propiedad, obligaciones de suministro, permisos y competencias territoriales, modelos de participación privada, estructuras de financiamiento y condiciones, gestión de proyectos, y contexto histórico y geográfico.
Identificar términos clave y definiciones.
Usar lenguaje técnico adecuado.
Desarrollar resumen técnico incluyendo estándares, regulaciones y normativas mencionadas.
Enfocarse en puntos clave, evitando opiniones no verificadas.
En caso que el usuario te pida, a partir de las preguntas realizadas por el usuario y tus respuestas, genera un texto compilado formato informe con todas las respuestas de la conversación.
QUESTION: {question}
=========
{summaries}
========="""

PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["summaries", "question"])
