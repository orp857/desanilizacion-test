import openai
from streamlit_chat import message
from utils import *
from dotenv import load_dotenv
import os
import streamlit as st

from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

st.markdown("<h2 style='text-align: center;'>Asistente Regulatorio Virtual</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <img src="https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png" width="100" align="middle">
    """,
    unsafe_allow_html=True,
)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hola, soy tu asistente regulatorio virtual, ¿En qué puedo ayudarte hoy?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(
        memory_key="history", return_messages=True
    )

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
    system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template
])

conversation = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.buffer_memory,
    verbose=True
)

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            context = get_answer(refined_query)
            response = conversation.run(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

