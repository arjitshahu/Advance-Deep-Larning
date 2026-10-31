import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
import os
# from langchain_ollama import ChatOllama -> Use advaced methods for answer setups
from langchain_ollama import OllamaLLM

st.title("C++ RAG Chatbot with Ollama")


#Load and Process Data 
def load_vectorstore():
    #Load Document
    loader = TextLoader("CPP.txt",encoding="utf-8")
    documents = loader.load()
    
    #Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20)
    final_documents=text_splitter.split_documents(documents)
    
    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #Vector Storing
    db = FAISS.from_documents(final_documents, embeddings)
    return db
db=load_vectorstore()

#Load LLM
#Ollama Model
# --- LLM CONFIGURATION ---
# FIXED: Switched from legacy 'Ollama' to modern 'ChatOllama'.
# FIXED: The modern class uses the parameter 'model=' instead of the old 'model_name='.

# llm = ChatOllama(model="gemma2:2b") -> Use advaced methods for answer setups with ChatOllama Library
llm=OllamaLLM(model="gemma2:2b")
#Chat Interface
user_question = st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking.."):
        docs=db.similarity_search(user_question)
        context = "\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Answer the wuestion using only the context below.
        
        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)  