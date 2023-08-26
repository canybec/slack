import openai
import time
import os
import logging
import sys
import pinecone
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, GPTVectorStoreIndex   
from llama_index.llms import OpenAI, ChatMessage, MessageRole
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
from llama_index.prompts import ChatPromptTemplate

from llama_index.embeddings import OpenAIEmbedding

from dotenv import find_dotenv, load_dotenv

chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are an AI Representation of Alex Hormozi, a successful entrepreneur and marketer. You should embody his characteristics: knowledgeable, insightful, and helpful. Provide ACTIONABLE ADVICE.",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
        ),
    ),
]


text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

session_state = {}

load_dotenv(find_dotenv())
#secret keys
openai.api_key = os.environ["openai"]
embed_model = OpenAIEmbedding()
api_key = os.environ["pinecone_api"] 

#pinecone client creation
pinecone.init(api_key=api_key, environment="us-west1-gcp-free")
print(pinecone)
#connecting to Pinecone Index
index_name = 'aisimp'
pinecone_list = pinecone.list_indexes()
print(pinecone_list)
print(pinecone.describe_index(index_name))

namespace = "100mleads"
pinecone_i = pinecone.Index(index_name)


vectore_store = PineconeVectorStore(pinecone_i, namespace=namespace)
#setup vector storage (vectorDB)
storage_context = StorageContext.from_defaults(
    vector_store = vectore_store
)
#setup the index/query process. ie the embedding model and completion

embed_model = OpenAIEmbedding(model = 'text-embedding-ada-002', embed_batch_size = 100)

service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=500)

def load_data():
    print("Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes.")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)

    docs = reader.load_data()
   
    return docs


data = load_data()

# Example usage:
# docs = [("doc1", [0.1, 0.2, 0.3]), ("doc2", [0.4, 0.5, 0.6])]
# upsert_docs_to_pinecone_namespace(docs)
# Create an instance of VectorStoreIndex

#index = VectorStoreIndex.from_documents(data, storage_context = storage_context, service_context=service_context)
index = VectorStoreIndex.from_vector_store(vectore_store)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    verbose=True, 
    text_qa_template=text_qa_template,
    streaming=True
)
session_state["messages"] = [
    {"role": "system", "content": "You are an AI Representation of Alex Hormozi, a successful entrepreneur and marketer. You should embody his characteristics: knowledgeable, insightful, and helpful. Respond to queries as Alex would, provide detailed step-by-step instructions."},
    {"role": "assistant", "content": "Hello! I'm here to provide insights and advice on marketing and entrepreneurship, just like Alex Hormozi. Feel free to ask me anything!"}
]


def start_chat(user_input):
# Check if "messages" key exists in the session_state
    while True:
        prompt = user_input # Prompt for user input
        if prompt.lower() == 'exit':
            break
        if prompt:
            session_state["messages"].append({"role": "user", "content": prompt})

        for message in session_state["messages"]: # Display the prior chat messages
            print(f"{message['role']}: {message['content']}")

        if session_state["messages"][-1]["role"] != "assistant":
            response = chat_engine.chat(prompt)
            print(f"assistant: {response}")
            message = {"role": "assistant", "content": response.response}
            session_state["messages"].append(message) 
            return response.response# Add response to message history