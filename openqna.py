import openai
import uvicorn
from fastapi import FastAPI, Request, Form
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
import os

app = FastAPI()

from dotenv import load_dotenv

load_dotenv()

# .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def setup_chain():
    # Define file path and template
    file = r"C:\Users\Anuj Bohra\Desktop\Arogo\Mental_Health_FAQ.csv"
    template = """You are a mental health assistant providing empathetic, accurate, and helpful responses to mental health-related questions. 
Use the following retrieved information as context to generate an informative response.

Context:
{context}

User's Question:
{question}

Answer in a professional yet understanding tone, ensuring clarity and support. If the context does not contain enough information, say, "I'm sorry, I don't have enough information to answer that."
"""

    # Initialize embeddings, loader, and prompt
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    loader = CSVLoader(file_path=file, encoding="utf-8")
    docs = loader.load()

    # Extract text from documents
    texts = [doc.page_content for doc in docs]

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create DocArrayInMemorySearch and retriever
    db = DocArrayInMemorySearch.from_texts(texts, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt": prompt}

    # Initialize ChatOpenAI with API key
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Setup RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    return chain


agent = setup_chain()


@app.get("/")
def read_root(request: Request):
    return {"message": "Welcome to the Mental Health Chatbot API!"}


@app.post("/prompt")
def process_prompt(prompt: str = Form(...)):
    response = agent.invoke({"query": prompt})  # Use invoke() instead of run()
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
