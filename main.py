from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import os, csv
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str

# Load docs
with open("data/docs.txt", encoding="utf-8") as f:
    text = f.read()
docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([text])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")),
    retriever=vectorstore.as_retriever()
)

def log_question(q: str):
    os.makedirs("logs", exist_ok=True)
    with open("logs/questions.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), q])

@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
async def ask(req: Question):
    log_question(req.question)
    return {"answer": qa.run(req.question)}
