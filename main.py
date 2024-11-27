from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory

import streamlit as st
from streamlit_chat import message
from utils import *

# Streamlit configuration
st.markdown("<h2 style='text-align: center;'>Asistente Regulatorio Virtual</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <img src="https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png" width="100" align="middle">
    """,
    unsafe_allow_html=True,
)

# Initialize session states
if "responses" not in st.session_state:
    st.session_state["responses"] = ["Hola, soy tu asistente regulatorio virtual, ¿En qué puedo ayudarte hoy?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define prompt templates
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

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=st.secrets["OPENAI_API_KEY"])

conversation = RunnableWithMessageHistory(
    memory=st.session_state.buffer_memory, llm=llm, prompt=prompt_template
)

# Streamlit UI components
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
            response = conversation.invoke(input=f"Context:\n{context}\n\nQuery:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + "_user")

