# Import necessary libraries and modules
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import openai
from streamlit_chat import message
from utils import *  # Import all functions and variables from utils.py
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables and secrets
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set up the Streamlit app title and logo
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
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(return_messages=True)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Define the conversation prompts
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Responde la pregunta con la mayor veracidad posible utilizando el contexto proporcionado. 
    Si no conoces la respuesta, pide más detalles hasta dos veces.
    Pregunta siempre si el usuario está satisfecho con la respuesta.
    Itera hasta tres veces si el usuario no está satisfecho.
    Proporciona fuentes al final de cada respuesta.
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

# Initialize the conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory, 
    prompt=prompt_template, 
    llm=llm, 
    verbose=True
)

# Containers for chat history and input text box
response_container = st.container()
textcontainer = st.container()

# Input text box for user queries
with textcontainer:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo..."):
            # Get the conversation history as a string
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            # Refine the user's query based on the conversation history
            refined_query = query_refiner(conversation_string, query)
            # Get the context from Pinecone based on the refined query
            context = get_answer(refined_query)
            # Generate the response using the conversation chain
            response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        # Update session state with the new query and response
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

# Display the chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            # Display the assistant's response
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                # Display the user's query
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

