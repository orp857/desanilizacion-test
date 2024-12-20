import streamlit as st
from langchain.chat_models import ChatOpenAI  # Updated import
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from utils import *

import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Define the Document class
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

# Display the title and logo
st.markdown("<h2 style='text-align: center;'>Asistente Regulatorio Virtual</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <img src="https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png" width="100" align="middle">
    """,
    unsafe_allow_html=True,
)

# Initialize session states if they don't exist
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hola, soy tu asistente regulatorio virtual, ¿En qué puedo ayudarte hoy?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the ChatOpenAI model correctly
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define the system and human message templates
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Responde la pregunta con la mayor veracidad posible utilizando el contexto proporcionado. 
    Si no conoce la respuesta, iterar dos veces para pedir más detalles.
    Preguntar siempre si el usuario está satisfecho con la respuesta.
    Iterar hasta tres veces si el usuario no está satisfecho.
    Proporcionar fuentes al final de cada respuesta. 
    Al final de tu respuesta, sugiere correlaciones con otros documentos de la misma temática.
    Cuando el usuario lo pida, ofrece la mejor recomendación basada en tu conocimiento general de las regulaciones y mejores prácticas en la industria, 
    incluyendo nuevas ideas y propuestas que puedan ser beneficiosas. Solo si no puedes ofrecer ninguna recomendación útil, entonces sugiere al usuario que 
    'Podrías preguntarle al equipo de Regulación, seguramente ellos podrán orientarte'."""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# Set up the conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Containers for chat history and text input
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            context = get_answer(refined_query)
            print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
