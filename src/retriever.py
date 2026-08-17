import os
import re
import glob

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_store"


def load_transcripts():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line =="WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))
        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

def build_retriever():
    return load_store().as_retriever(search_kwargs={"k":5})


if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.invoke("What is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")