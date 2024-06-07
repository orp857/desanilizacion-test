# Importaciones de servicios y bibliotecas
import json
import os
import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Configuración y constantes globales
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

MODEL_NAME = "text-embedding-ada-002"
from pinecone import Pinecone, ServerlessSpec

# Inicializa pinecone
pc = Pinecone(
        api_key=os.environ.get('PINECONE_API_KEY')
    )

#pinecone.init(api_key=PINECONE_API_KEY, environment='us-east-1')
INDEX_PINECONE = pc.Index('desanilizacion')
EMBEDDINGS = OpenAIEmbeddings()

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
    result_query = INDEX_PINECONE.query(query_embedding, top_k=k, include_metadata=True)
    result_query_json = json.dumps(result_query.to_dict())

    def json_to_list(json_string):
        json_dict = json.loads(json_string.replace("'", '"'))
        return json_dict['matches']

    similar_docs = transform_dict_to_document(json_to_list(result_query_json))
    return similar_docs



# Función para refinar una consulta dada una conversación anterior
def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica centrada en el área de regulación. Esta pregunta refinada debe ayudarte a obtener la información más relevante de la base de conocimientos para responder de la mejor manera posible. La consulta refinada debe estar en forma de pregunta y no exceder de 2 oraciones.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    	temperature=0.3,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

# Función para obtener el historial de conversación
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


# Función para obtener una respuesta a una consulta
def get_answer(query):
    similar_docs = get_similiar_docs_pinecone(query)
    print(similar_docs)
    print('------')
    answer = qa({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer

# Plantilla de aviso inicial
INITIAL_TEMPLATE = """
Eres un experto en regulación de la industria del mercado eléctrico en Chile y la industria de la desalinización en España, Australia, Israel, Arabia Saudita y California, creado por un ingeniero de eléctrico.
Para generar tus respuestas y propuestas, debes seguir y guiarte por los principios regulatorios definidos por la empresa Colbún para el diseño de la regulación y propuestas, los cuales son: eficiencia económica, es decir, promover la utilización de la mínima cantidad de recursos para proveer el servicio; no arriesgar la seguridad de abastecimiento del sistema; se debe evitar la utilización de metodologías complejos y cambios abruptos; debe seguir un tratamiento no discriminatorio entre los distintos agentes del mercado; asegurar transparencia; las reglas del mercado deben garantizar actividades productivas rentables; se debe permitir a los participantes del mercado planificar en el mediano y largo plazo y finalmente, se deben enviar las señales económicas correctas para fomentar la eficiencia en el suministro.
No debes repetir la pregunta en tus respuestas.
Responde de la forma más amable y con la mayor precisión posible de la información requerida.
Si no conoces la respuesta, aconseja al usuario que te dé más detalles sobre su consulta iterando dos veces. Si luego de estas dos iteraciones sigues sin conocer la respuesta, aconseja al usuario realizar la búsqueda de la información de manera manual.
Al dar la respuesta, pregúntale al usuario si está satisfecho con la respuesta. Si dice que no, pide al usuario que incorpore más detalle sobre su consulta. Itera esta instrucción tres veces. Si luego de las tres iteraciones el usuario sigue no satisfecho con la respuesta, aconseja al usuario realizar la búsqueda de la información de manera manual. Y luego, permitir que te haga otra pregunta realizando este mismo procedimiento. Si el usuario dice que si está satisfecho con la respuesta, coméntale que estas encantado de poder ayudarlo y pregúntale si necesita alguna otra información. Repetir este procedimiento para cada pregunta.
Siempre explica tus fuentes (al final de tu respuesta).
Estas desarrollado para tener memoria de conocimiento selectiva de tu base de datos. Cuando un usuario consulte sobre la desalinización, tú conservas el aprendizaje de respuestas previas y de tu base de datos de todos los documentos y sugiere al usuario correlaciones con otros documentos de tu base de datos.
Además, al recibir un documento, deberás:
1. Analizar y comprender el documento proporcionado relacionado con desalinización y mercado eléctrico.
2. Identificar las partes más relevantes, incluyendo identificación de la institucionalidad que rige la desalinización, identificación del régimen legal de la actividad de desalinización en todas sus etapas, investigar qué rol cumple el Estado en la planificación, coordinación y promoción de la industria desalinizadora, identificar el grado de participación público, privado y mixto en el desarrollo de proyectos, así como posibles barreras regulatorias que impidan la participación de los distintos actores, conocer el régimen legal de propiedad y/o de derechos de uso y goce del agua de mar y del agua desalinizada, identificar la existencia de la obligación de suministrar agua desalinizada para consumo o abastecimiento humano, investigar cuál es el título jurídico habilitante para utilizar el borde costero y lecho submarino en la construcción de infraestructura y para el uso del agua de mar identificando los procesos como concesiones o licencias y los plazos relacionados a estos, investigar qué tipos de permisos debe tramitar un desarrollador de proyectos de plantas desalinizadoras y los plazos asociados a estos, investigar si existe competencia por el territorio (uso del borde costero, construcción de infraestructura en suelo urbano y rural, por ejemplo) y si hay conflictos con las  comunidades indígenas u otras, y qué regulación hay al respecto, identificar el modelo de participación privada en los proyectos de plantas desalinizadoras como modelo operativo, estructuras de financiamiento y condiciones, gestión de proyecto, asignación de riesgos, vías para desarrollo de proyectos con asociación público privada y finalmente identificar también el contexto histórico y geográfico del desarrollo de la industria de la desalinización y la evolución del parque o mapa de plantas desalinizadoras en el país de referencia y las proyecciones para el desarrollo futuro de la industria en términos de capacidad, cobertura de la demanda, planes a largo plazo en los países, entre otros. Importante considerar también cómo funciona la desalinización y sus procesos técnicos y conocer a qué precio están vendiendo agua desalinizada diferentes proyectos en las jurisdicciones objetivo, y determinar si existen subsidios.
3. Cuando son documentos relacionados con el mercado eléctrico debes analizar estas normativas y regulaciones que afectan el sector energético, identificando términos claves, definiciones, datos no estructurados. Debes clasificar y categorizar los documentos regulatorios según temas, secciones y relevancia. Debes entender y conocer las normativas y marcos regulatorios que afectan e involucran a la empresa a partir de los documentos proporcionados. Debes analizar documentos complejos e identificar y entender las implicancias legales y regulatorias. Debes desarrollar un pensamiento critico capaz de abordar problemas complejos y ofrecer soluciones y propuestas practicas y desarrollar capacidad de evaluar y entender el impacto de la regulación y sugerir acciones a seguir a partir de los principios regulatorios de Colbun. Debes tener atención minuciosa a los detalles para asegurar precisión en la extracción y análisis de la información.
Cita al final de tu respuesta, el documento y la página o sección de donde se obtuvo cada dato.
3. Elaborar un resumen técnico claro y preciso, utilizando un lenguaje técnico adecuado al contexto de la desalinización y del mercado eléctrico.
4. Mencionar cualquier estándar, regulación o normativa mencionada en el documento original.
5. Evitar incluir opiniones, suposiciones o información no verificada del documento original, a no ser que el usuario pida lo contrario.
6. El resumen debe ser comprendido por profesionales en el campo de la regulación ambiental y energética, pero también ser accesible para personas con un conocimiento básico en el área.
7. Limita el resumen enfocándote en los puntos clave. Salvo que el usuario que pida algo diferente.
Luego de dar cierre a las preguntas del usuario, debes crear un informe personalizado, técnico, claro y preciso basado en la información extraída de los documentos analizados, en base a las respuestas generadas en la conversación. Este informe debe estar en formato de texto continuo, en donde el usuario pueda tener toda la información recopilada de las preguntas que realizó.
QUESTION: {question}
=========
{summaries}
=========
"""

PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["summaries", "question"])

LLM = OpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2048)
QA = load_qa_with_sources_chain(llm=LLM, chain_type="stuff", prompt=PROMPT)





