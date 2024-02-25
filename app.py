!pip install python-dotenv
!pip install gradio
!pip install langchain_google_genai
!pip install langchain
!pip install datasets
!pip install PyPDF2
!pip install streamlit
!pip install faiss-cpu

#import the necessary libraries
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import getpass
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
import logging
import gradio as gr
from transformers import pipeline

# Apply the Gemini API Key: 

api_key = getpass.getpass("Please enter your Google API Key:")
os.environ["GOOGLE_API_KEY"] = api_key

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            logging.error(f"Error processing PDF file '{pdf}': {str(e)}")
            continue
    return text

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        
        # Additional code for successful initialization of embeddings and Faiss index
    except Exception as e:
        logging.error(f"Error loading embeddings or Faiss index: {str(e)}")
        # Handle the error appropriately.
        return None  # Return None in case of an error

    # Check if embeddings is not empty and has a valid 'embed_documents' method
    if not embeddings or not hasattr(embeddings, 'embed_documents') or not callable(getattr(embeddings, 'embed_documents', None)):
        logging.error("Error: Invalid embeddings object.")
        return None  # Return None if embeddings is invalid

    # Use the language model to generate embeddings for text chunks
    embeddings_result = embeddings.embed_documents(text_chunks)

    # Check if the embeddings result is not empty
    if not embeddings_result or not embeddings_result[0]:
        logging.error("Error: Empty or invalid embeddings result.")
        return None  # Return None if embeddings result is empty

    # Create a Faiss index from the embeddings result
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    try:
        vector_store.save_local("faiss_index")
    except Exception as e:
        logging.error(f"Error saving Faiss index: {str(e)}")
    return vector_store


def get_conversational_chain():

    prompt_template = """
    Provide a detailed response to the question using the context that has been supplied.
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer: rewrite the context in coherent english language
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = None
    try:
      response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
      print(response
)
      # st.write("Reply: ", response["output_text"])
    except Exception as e:
      logging.error(f"Error in user_input: {str(e)}")
      # Handle the error appropriately.
      print(docs)
      st.write("Reply: ", response["output_text"])

pdf_folder = '/content/drive/MyDrive/pdf_files'
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
text = get_pdf_text(pdf_files)
text_chunks = get_chunks(text)
get_vector_store(text_chunks)

# print(text_chunks)
