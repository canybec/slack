import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import csv
import json

st.set_page_config(page_title="Document Uploader for LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Upload documents for LlamaIndex 🦙")
st.info("Upload your document to manage what data is in the vector store. Each new document will be indexed using a namespace equal to the document's name.")

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the existing docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

# Uploader UI
uploaded_file = st.file_uploader("Choose a document to upload", type=['txt', 'md', 'rst', 'csv', 'json'])

if uploaded_file:
    file_content = uploaded_file.read().decode()
    document_name = uploaded_file.name

    # Process the file based on its type
    if document_name.endswith('.csv'):
        csv_content = []
        csv_reader = csv.reader(file_content.splitlines(), delimiter=',')
        for row in csv_reader:
            csv_content.append(", ".join(row))
        file_content = "\n".join(csv_content)
    elif document_name.endswith('.json'):
        json_content = json.loads(file_content)
        file_content = json.dumps(json_content, indent=4)

    # Create a new Document with the processed content and its name as the namespace
    new_doc = Document(namespace=document_name, content=file_content)
    index.add_documents([new_doc])  # Add the new document to the index

    st.success(f"Document '{document_name}' has been uploaded and indexed!")
