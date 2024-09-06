import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Load your dataset
def load_data(file_path):
    loader = CSVLoader(file_path=file_path, encoding='utf8')
    data = loader.load()
    return data

# Set up text splitter
def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=15)
    return text_splitter.split_documents(data)

# Set up the RAG model (retriever + LLM)
def setup_rag_model(data):
    # Initialize embeddings and retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(data, embeddings)
    retriever = vector_store.as_retriever()

    # Initialize LLM (e.g., GPT-3 or other)
    llm = OpenAI(mocel_name = 'gpt-4',temperature=0)

    # Set up the prompt template
    template = """
    You are an AI assistant. Provide concise and accurate answers based on the following context.
    Use 5 sentences max. Always end the answer with 'Thanks for asking!'.
    If you don't know, say 'I don't know'.
    Context: {context}
    Question: {question}
    Answer: """
    
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": qa_prompt})
    
    return qa_chain

# Function to query the model
def qa_chainH(query, qa_chain):
    question = query['query']
    result = qa_chain({"query": question})
    return result