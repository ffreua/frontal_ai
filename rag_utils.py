# rag_utils.py
import os
import tempfile
from typing import List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_llm(api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.2):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model = model or DEFAULT_CHAT_MODEL
    return ChatOpenAI(
        model=model,
        temperature=temperature,
    )

def get_embeddings(api_key: Optional[str] = None, model: Optional[str] = None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model = model or DEFAULT_EMBEDDING_MODEL
    return OpenAIEmbeddings(model=model)

def save_upload_to_temp(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    return tmp_path

def load_docs_from_files(files) -> List:
    docs = []
    for f in files:
        ext = f.name.lower().split(".")[-1]
        tmp_path = save_upload_to_temp(f)
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        elif ext in ("docx", "doc"):
            loader = Docx2txtLoader(tmp_path)
            docs.extend(loader.load())
        elif ext in ("txt",):
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs.extend(loader.load())
        else:
            # Ignora tipos não suportados
            pass
    return docs

def split_documents(docs, chunk_size=1200, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def ensure_vectorstore(docs, embeddings, persist_dir="vectorstore") -> FAISS:
    os.makedirs(persist_dir, exist_ok=True)
    index_path = os.path.join(persist_dir, "faiss_index")
    if docs:
        if os.path.exists(index_path):
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vs.add_documents(docs)
            vs.save_local(index_path)
        else:
            vs = FAISS.from_documents(docs, embeddings)
            vs.save_local(index_path)
    else:
        if os.path.exists(index_path):
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            vs = None
    return vs

def reset_vectorstore(persist_dir="vectorstore"):
    index_path = os.path.join(persist_dir, "faiss_index")
    if os.path.isdir(index_path):
        # Remove diretório do índice
        for root, dirs, files in os.walk(index_path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
        try:
            os.rmdir(index_path)
        except Exception:
            pass

def retrieve_context(vs: Optional[FAISS], query: str, k: int = 5) -> Tuple[str, List]:
    if vs is None:
        return "", []
    docs = vs.similarity_search(query, k=k)
    context_texts = []
    for d in docs:
        meta = d.metadata or {}
        title = meta.get("source", "Documento")
        page = meta.get("page", None)
        tag = f"{title}" + (f" (p. {page})" if page is not None else "")
        context_texts.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(context_texts), docs