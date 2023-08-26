import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import requests

st.set_page_config(page_title = "Mozi Slack Bot Knowledge Management Dashboard", page_icon="⭕️",
   layout="centered",  initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets.openai_key
st.title ("Upload your Docs, powered by AIsimp and LlamaIndex!")
st.info ("For any partnering questions or technical support Call Goose 🪿")

@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(text="Loading and Indexing New Mozi Data - hang tight! This Should take 1-2minutes."):
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
      system_prompt="You are Alex Harmozi, user you knowledge to answer questions. Don't Hallucinate."))
    return index
    
index = load_data()

chat_engine = index.as_chat_engine(chat_model="condense_question", verbose=True)

if prompt := st.chat_input("Your Question"): 
    st.session_state.message.append({"role": "user", "content": prompt})


for message in st.session_state.messages: #Display the prior chat messages
  with st.chat_message(message["role"]):
    st.write(message["content"])

if st.session_state.messages[-1]["role"] !="assistant":
  with st.chat_message("assisstant"):
    with st.spinner("Thinking ... "):
      response = chat_engine.chat(prompt)
      st.write(response.response)
      message = {"role": "assisstant", "content": response.response}
      st.session_state.messages.append(message) #add respond to message history