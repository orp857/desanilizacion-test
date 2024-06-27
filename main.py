from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import openai
from streamlit_chat import message
from utils import *


from dotenv import load_dotenv
import os


import streamlit as st

# Define las variables de entorno y otros secrets


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__


st.markdown("<h2 style='text-align: center;'>Asistente Regulatorio Virtual</h2>", unsafe_allow_html=True)

# st.markdown(
#     """
#     <img src="https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png" width="100" align="middle">
#     """,
#     unsafe_allow_html=True,
# )

#st.markdown("<h6 style='text-align: center;'>Creado por: Robinson Cornejo</h6>", unsafe_allow_html=True)


# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state['responses'] = ["Hola, soy tu asistente regulatorio virtual, ¿En qué puedo ayudarte hoy?"]
    st.session_state['requests'] = []
    st.session_state['initialized'] = True  # Flag to check if initialization has occurred

# Path to the custom bot image
logo_path = 'https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png'

# Custom function to display bot messages with custom logo
def display_bot_message(content, logo_path):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="{logo_path}" width="50" height="50" style="margin-right: 10px;">
            <div style="background-color: #f1f0f0; border-radius: 5px; padding: 10px; margin: 5px;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Custom function to display user messages without logo
def display_user_message(content):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-end;">
            <div style="background-color: #dcf8c6; border-radius: 5px; padding: 10px; margin: 5px;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display bot responses with custom logo
for response in st.session_state['responses']:
    display_bot_message(response, logo_path=logo_path)

# Display user requests without logo
for request in st.session_state['requests']:
    display_user_message(request)
    
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


# system_msg_template = SystemMessagePromptTemplate.from_template(
#     template="""Responda la pregunta con la mayor veracidad posible utilizando el contexto proporcionado,
# y si la respuesta no está contenida en el texto a continuación, diga 'Podrias preguntarle al equipo de Regulación, seguramente ellos podran orientarte' """)

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Responde la pregunta con la mayor veracidad posible utilizando el contexto proporcionado. 
    Si no conoce la respuesta, iterar dos veces para pedir más detalles.
Preguntar siempre si el usuario está satisfecho con la respuesta.
Iterar hasta tres veces si el usuario no está satisfecho.
Proporcionar fuentes al final de cada respuesta. 
Al final de tu respuesta, sugiere correlaciones con otros documentos de la misma temática.
Cuando el usuario lo pida, ofrece la mejor recomendación basada en tu conocimiento general de las regulaciones y mejores prácticas en la industria, 
    incluyendo nuevas ideas y propuestas que puedan ser beneficiosas. Solo si no puedes ofrecer ninguna recomendación útil, entonces sugiere al usuario que 
    'Podrías preguntarle al equipo de Regulación, seguramente ellos podrán orientarte'.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            #user_entities = extract_entities(query)
            #print("#######")
            #print(user_entities)
            refined_query = query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = get_answer(refined_query)
            print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
