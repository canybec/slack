import openai
import time
import os
import logging
import sys
import pinecone
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, GPTVectorStoreIndex   
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from dotenv import find_dotenv, load_dotenv
from datetime import datetime

session_state = {}

load_dotenv(find_dotenv())
#secret keys
openai.api_key = os.environ["openai"]
embed_model = OpenAIEmbedding()
api_key = os.environ["pinecone_api"] 

#pinecone client creation
pinecone.init(api_key=api_key, environment="us-west1-gcp-free")
#connecting to Pinecone Index
index_name = 'aisimp'
pinecone_list = pinecone.list_indexes()
print(pinecone_list)
print(pinecone.describe_index(index_name))

namespace = "moziSpace"
pinecone_i = pinecone.Index("index_name")



vectore_store = PineconeVectorStore(pinecone_i)
#setup vector storage (vectorDB)
storage_context = StorageContext.from_defaults(
    vector_store = vectore_store
)
#setup the index/query process. ie the embedding model and completion
embed_model = OpenAIEmbedding(model = 'text-embedding-ada-002', embed_batch_size = 100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

def load_data():
    print("Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes.")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    return docs

# Load the timestamp of the last upload
try:
    with open('last_upload.txt', 'r') as f:
        last_upload = datetime.strptime(f.read(), '%Y-%m-%d %H:%M:%S.%f')
except FileNotFoundError:
    last_upload = datetime.min

# Get the current timestamp
current_time = datetime.now()

# Load the data
data = []
for filename in os.listdir('./data'):
    # Get the modification time of the file
    mod_time = datetime.fromtimestamp(os.path.getmtime('./data/' + filename))
    # If the file has been modified after the last upload, add it to the data
    if mod_time > last_upload:
        data.append(load_data(filename))

# If there is new data, upload it
if data:
    # Create an instance of VectorStoreIndex
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    # Now you can call the as_chat_engine method
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Save the current timestamp for the next run
with open('last_upload.txt', 'w') as f:
    f.write(str(current_time))

# Check if "messages" key exists in the session_state
if "messages" not in session_state:
    session_state["messages"] = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

# Initial questions
initial_questions = ["What does harmozi do?", "What about after that?"]

for question in initial_questions:
    response = chat_engine.chat(question)
    print(f"assistant: {response.response}")
    message = {"role": "assistant", "content": response.response}
    session_state["messages"].append(message)

while True:
    prompt = input("Your question: ") # Prompt for user input
    if prompt.lower() == 'exit':
        break
    if prompt:
        session_state["messages"].append({"role": "user", "content": prompt})

    for message in session_state["messages"]: # Display the prior chat messages
        print(f"{message['role']}: {message['content']}")

    if session_state["messages"][-1]["role"] != "assistant":
        response = chat_engine.chat(prompt)
        print(f"assistant: {response.response}")
        message = {"role": "assistant", "content": response.response}
        session_state["messages"].append(message) # Add response to message history